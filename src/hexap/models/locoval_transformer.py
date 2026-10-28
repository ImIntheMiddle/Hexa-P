"""The Hexa-P network: Physical Core, adapters, TaPE, Path-Head and LocoVal-Head.

The same network is trained in both stages. In Stage 1 (physics pre-training)
the Physical Core (`_cross_modality_transformer`), the Path-Head
(`predict_head`) and the LocoVal-Head (`reward_head`) are trained on simulated
locomotion of a single agent. In Stage 2 the Physical Core and the LocoVal-Head
are frozen, bottleneck adapters are added to each layer of the
Physical Core, and the social module with Target-relative Positional Encoding
(TaPE) is trained on a real multi-agent dataset.

Inference has two modes. Prediction Mode runs the Physical Core with the
adapters and predicts the path hypotheses. Evaluation Mode runs it without the
adapters and scores a given path with the LocoVal-Head; LocoVal filtering uses
this score.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# Joints of the 23-joint local pose that each dataset provides (the pelvis is the path token).
AVAILABLE_JOINTS = {
    "jta": [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 16, 19, 20, 21],
    "vru": [0, 1, 2, 4, 5, 6, 12, 14, 15, 16, 19, 20, 21],
    "jrdb": list(range(23)),
    "amass": list(range(23)),
}


class Attention(nn.Module):
    """Multi-head self-attention with an additive mask, written out explicitly."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """x: (L, B, D); key_padding_mask: (B, L), True for padded tokens."""
        L, B, D = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(L, B, 3, self.num_heads, self.head_dim)
            .permute(2, 1, 3, 0, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, head_dim)
        bias = torch.zeros(B, self.num_heads, L, L, device=q.device, dtype=q.dtype)
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, L)
            mask = key_padding_mask.view(B, 1, 1, L).expand(-1, self.num_heads, -1, -1)
            mask = mask.reshape(B * self.num_heads, 1, L).view(B, self.num_heads, -1, L)
            bias.masked_fill_(mask, float("-1e9"))
        weight = torch.softmax(q @ k.transpose(-2, -1) * self.scale + bias, dim=-1)
        out = self.dropout(weight) @ v  # (B, H, L, head_dim)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D).transpose(0, 1))


class EncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer with optional bottleneck adapters on the attention and FFN outputs."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        enable_adapter: bool = False,
        adapter_bottleneck_ratio: int = 4,
        adapter_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.enable_adapter = enable_adapter
        self.self_attn = Attention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        if enable_adapter:
            hidden = d_model // adapter_bottleneck_ratio
            self.adapter_scale = adapter_scale_init
            self.adapter1_down = nn.Linear(d_model, hidden)
            self.adapter1_up = nn.Linear(hidden, d_model)
            self.adapter2_down = nn.Linear(d_model, hidden)
            self.adapter2_up = nn.Linear(hidden, d_model)
            for up in (self.adapter1_up, self.adapter2_up):
                nn.init.zeros_(up.weight)
                nn.init.zeros_(up.bias)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        with_adapter: bool,
    ) -> torch.Tensor:
        use_adapter = self.enable_adapter and with_adapter
        x_sa = self.dropout1(self.self_attn(x, key_padding_mask))
        if use_adapter:
            x_sa = x_sa + self.adapter_scale * self.adapter1_up(
                F.relu(self.adapter1_down(x_sa))
            )
        x = self.norm1(x + x_sa)
        x_ff = self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )
        if use_adapter:
            x_ff = x_ff + self.adapter_scale * self.adapter2_up(
                F.relu(self.adapter2_down(x_ff))
            )
        return self.norm2(x + x_ff)


class Encoder(nn.TransformerEncoder):
    """A stack of `EncoderLayer`s (kept as nn.TransformerEncoder for the parameter names)."""

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        with_adapter: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask, with_adapter)
        return x


class BiLearnedTrajEncoding(nn.Module):
    """Learned temporal encoding of path tokens, indexed by time at the pre-training frame rate."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_obs_len: int = 18,
        max_pred_len: int = 24,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True)
        self.learned_encoding_pred = nn.Embedding(max_pred_len, d_model, max_norm=True)

    def forward(
        self, x: torch.Tensor, in_F: int, out_F: int, stride: int
    ) -> torch.Tensor:
        """x: (B, in_F + out_F, N, D)."""
        obs_idx = torch.arange(in_F - 1, -1, -1, device=x.device) * stride
        pred_idx = (stride - 1) + torch.arange(out_F, device=x.device) * stride
        x[:, :in_F] = x[:, :in_F] + self.learned_encoding_obs(obs_idx).unsqueeze(
            0
        ).unsqueeze(2)
        x[:, in_F : in_F + out_F] = x[
            :, in_F : in_F + out_F
        ] + self.learned_encoding_pred(pred_idx).unsqueeze(0).unsqueeze(2)
        return self.dropout(x)


class BiLearnedPoseEncoding(nn.Module):
    """Learned encoding of pose tokens (frame-major, joint-minor), indexed like the path tokens."""

    def __init__(self, d_model: int, dropout: float, max_obs_len: int = 414) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True)

    def forward(self, x: torch.Tensor, num_tokens: int, stride: int) -> torch.Tensor:
        """x: (B, num_tokens, N, D)."""
        idx = torch.arange(num_tokens - 1, -1, -1, device=x.device) * stride
        x[:, :num_tokens] = x[:, :num_tokens] + self.learned_encoding_obs(
            idx
        ).unsqueeze(0).unsqueeze(2)
        return self.dropout(x)


class LearnedIDEncoding(nn.Module):
    """Learned embedding of the agent index, added to every frame of that agent."""

    def __init__(self, d_model: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.person_encoding = nn.Embedding(1000, d_model, max_norm=True)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, num_people: int) -> torch.Tensor:
        """x: (N, F, B, D)."""
        ids = (
            torch.arange(num_people, device=x.device)
            .unsqueeze(1)
            .repeat_interleave(self.seq_len, dim=1)
        )
        return self.dropout(x + self.person_encoding(ids).unsqueeze(2))


class TargetRelativePositionEncoding(nn.Module):
    """TaPE: encodes the position of every agent relative to the target agent at each frame."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.nhid = d_model
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.relative_valid_encoder = nn.Linear(4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, rel: torch.Tensor, padding_mask: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """rel: (B, in_F, N, 4) offsets from the target (NaN where unobserved).

        Returns (N * num_frames, B, D); future frames are encoded as unobserved.
        """
        B, in_F, N, K = rel.shape
        rel = F.pad(rel, (0, 0, 0, 0, 0, num_frames - in_F), "constant", 0)[
            :, :num_frames
        ]
        valid = ~torch.isnan(rel)
        rel = torch.nan_to_num(rel)
        padded = (
            padding_mask.unsqueeze(1).unsqueeze(3).expand(-1, num_frames, -1, K).bool()
        )
        valid = valid & ~padded
        valid[:, in_F:] = False
        feature = self.relative_pos_encoder(rel) + F.gelu(
            self.relative_valid_encoder(valid.float())
        )
        feature = self.dropout(self.norm(feature)).permute(2, 1, 0, 3)  # (N, F, B, D)
        return feature.reshape(-1, B, self.nhid)


class RewardHead(nn.Module):
    """LocoVal-Head: scores the plausibility of future paths from the features of the target agent."""

    def __init__(self, d_model: int, max_pred_len: int = 24) -> None:
        super().__init__()
        self.nhid = d_model
        self.max_pred_len = max_pred_len
        self.fc_in_future_traj = nn.Linear(2, d_model // 2)
        self.fc_temp = nn.Linear(max_pred_len, 1)
        self.fc_out_rew = nn.Linear(d_model * 2, 1)
        self.norm = nn.LayerNorm(d_model * 2)
        self.dropout = nn.Dropout(0.1)
        self.fc_future_valid = nn.Linear(1, d_model // 2)

    def forward(
        self,
        feature: torch.Tensor,
        paths: torch.Tensor,
        stride: int,
        masking: bool,
    ) -> torch.Tensor:
        """feature: (out_F, B, D) future-frame features; paths: (B, out_F, n, 2). Returns (B, n) in [0, 1]."""
        B, out_F, n, _ = paths.shape
        device = paths.device
        valid = torch.zeros(self.max_pred_len, dtype=torch.bool, device=device)
        idx = (stride - 1) + torch.arange(out_F, device=device) * stride
        valid[idx[idx < self.max_pred_len]] = True

        paths_up = torch.zeros(B, self.max_pred_len, n, 2, device=device)
        paths_up[:, valid] = paths
        feature_up = torch.zeros(B, self.max_pred_len, self.nhid, device=device)
        feature_up[:, valid] = feature.permute(1, 0, 2)
        if masking:
            valid = valid & ~(torch.rand(self.max_pred_len, device=device) < 0.3)
            paths_up[:, ~valid] = 0.0
            feature_up[:, ~valid] = 0.0

        valid_in = valid.unsqueeze(-1).unsqueeze(0).expand(B, -1, -1).float()
        path_feature = F.relu(self.fc_in_future_traj(paths_up))
        valid_feature = (
            F.relu(self.fc_future_valid(valid_in)).unsqueeze(2).expand(-1, -1, n, -1)
        )
        x = torch.cat(
            (
                feature_up.unsqueeze(2).expand(-1, -1, n, -1),
                path_feature,
                valid_feature,
            ),
            dim=-1,
        )
        x = self.fc_temp(x.permute(0, 2, 3, 1)).squeeze(-1)  # (B, n, 2D)
        x = self.dropout(self.norm(F.gelu(x)))
        return torch.sigmoid(self.fc_out_rew(x).squeeze(-1))


class CrossModalityTransformer(nn.Module):
    """Physical Core: per-agent Transformer over path tokens and 3D pose tokens."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.nhid = config.dim_hidden
        self.obs = config.input_track_size
        self.pred = config.output_track_size
        self.token_num = config.token_num
        self.joints_3dpose = config.joints_3dpose
        self.stride = int(max(1, config.inputs.max_fps // config.inputs.fps))
        self.mask_ratio_modality = config.inputs.mask_ratio_modality
        self.mask_ratio_traj = config.inputs.mask_ratio_traj
        self.mask_ratio_frame = config.inputs.mask_ratio_frame
        self.mask_ratio_joints = config.inputs.mask_ratio_joints
        available = AVAILABLE_JOINTS[config.inputs.pose_mask]
        self.available_joints = torch.tensor(available)
        self.unavailable_joints = torch.tensor(
            [j for j in range(self.joints_3dpose) if j not in available]
        )

        self.fc_in_traj = nn.Linear(2, self.nhid)
        self.traj_encoder = BiLearnedTrajEncoding(self.nhid, config.dropout)
        self.fc_in_3dpose = nn.Linear(3, self.nhid)
        self.pose3d_encoder = BiLearnedPoseEncoding(self.nhid, config.dropout)
        layer = EncoderLayer(
            self.nhid,
            config.num_heads,
            config.dim_feedforward,
            config.dropout,
            config.activation,
            enable_adapter=config.peft_type == "adapter",
            adapter_bottleneck_ratio=config.adapter_bottleneck_ratio,
            adapter_scale_init=config.adapter_scale_init,
        )
        self.local_former = Encoder(
            layer, num_layers=config.num_layers_local, enable_nested_tensor=False
        )

    def forward(
        self,
        tgt: torch.Tensor,
        padding_mask: torch.Tensor,
        random_masking: bool,
        limit_obs: int,
        with_adapter: bool,
    ) -> torch.Tensor:
        """tgt: (B, in_F, N * token_num, 4) normalized tokens. Returns (L, B * N, D) local features."""
        B, in_F, NJ, K = tgt.shape
        out_F = self.pred
        num_frames = in_F + out_F
        J = self.token_num
        N = NJ // J
        device = tgt.device

        if random_masking:
            mask_ratio_traj, mask_ratio_modality = (
                self.mask_ratio_traj,
                self.mask_ratio_modality,
            )
            mask_ratio_joints = (
                torch.rand(1).to(device)
                if self.mask_ratio_joints == 1.0
                else self.mask_ratio_joints
            )
            mask_ratio_frame = (
                torch.rand(1).to(device)
                if self.mask_ratio_frame == 1.0
                else self.mask_ratio_frame
            )
        else:
            mask_ratio_traj = mask_ratio_modality = mask_ratio_joints = (
                mask_ratio_frame
            ) = 0

        # Future frames repeat the last observed frame.
        idx = np.append(np.arange(0, in_F), np.repeat([in_F - 1], out_F))
        tgt = torch.nan_to_num(tgt[:, idx].reshape(B, num_frames, N, J, K))

        tgt_traj = tgt[:, :, :, 0, :2]
        traj_mask = (
            torch.rand((B, in_F - 1, N), device=device).float() > mask_ratio_traj
        )
        tgt_traj[:, : in_F - 1] = tgt_traj[:, : in_F - 1] * traj_mask.unsqueeze(
            3
        ).repeat_interleave(2, dim=-1)

        modality = (
            torch.rand((B, 1, N), device=device).float() > mask_ratio_modality
        ).repeat(1, num_frames, 1)
        tgt_vis = tgt[:, :, :, 1:] * (
            modality.unsqueeze(3)
            .unsqueeze(4)
            .repeat_interleave(self.joints_3dpose, dim=3)
            .repeat_interleave(K, dim=4)
        )
        tgt_pose = tgt_vis[:, :, :, : self.joints_3dpose, :3]

        if limit_obs != 0:
            tgt_traj[:, : in_F - limit_obs] = 0
            tgt_pose = tgt_pose[:, in_F - limit_obs : in_F]
            frames = limit_obs
        elif mask_ratio_frame > 0:
            num_false = int((in_F - 1) * mask_ratio_frame)
            false_idx = torch.randperm(in_F - 1, device=device)[:num_false]
            true_idx, _ = torch.sort(
                torch.tensor(
                    [i for i in range(in_F) if i not in false_idx], device=device
                )
            )
            tgt_traj[:, false_idx] = 0
            tgt_pose = tgt_pose[:, true_idx]
            frames = in_F - num_false
        else:
            tgt_pose = tgt_pose[:, :in_F]
            frames = in_F

        tgt_traj = self.traj_encoder(
            self.fc_in_traj(tgt_traj), in_F, out_F, self.stride
        )
        tgt_pose = tgt_pose.transpose(2, 3).reshape(B, -1, N, 3)
        tgt_pose = self.pose3d_encoder(
            self.fc_in_3dpose(tgt_pose), frames * self.joints_3dpose, self.stride
        )

        # Drop the pose tokens of joints the dataset lacks and of randomly masked joints.
        num_false = int(len(self.available_joints) * mask_ratio_joints)
        joint_mask = torch.ones(self.joints_3dpose, dtype=torch.bool, device=device)
        false_idx = self.available_joints.to(device)[
            torch.randperm(len(self.available_joints), device=device)[:num_false]
        ]
        if len(self.unavailable_joints) > 0:
            false_idx = torch.cat(
                (false_idx, self.unavailable_joints.to(device)), dim=0
            )
        joint_mask[false_idx] = False
        kept = self.joints_3dpose - false_idx.shape[0]
        tgt_pose = tgt_pose[:, joint_mask.unsqueeze(0).repeat(frames, 1).reshape(-1)]
        tgt_pose = (
            tgt_pose.view(B, frames, kept, N, self.nhid).transpose(1, 2).contiguous()
        )

        tgt_traj = torch.transpose(tgt_traj, 0, 1).reshape(num_frames, -1, self.nhid)
        tgt_pose = torch.transpose(tgt_pose, 0, 1).reshape(frames * kept, -1, self.nhid)
        tokens = torch.cat((tgt_traj, tgt_pose), 0)  # (L, B * N, D)
        key_padding = (
            padding_mask.reshape(-1)
            .unsqueeze(1)
            .repeat_interleave(tokens.shape[0], dim=1)
            .bool()
        )
        out = self.local_former(
            tokens.clone(), src_key_padding_mask=key_padding, with_adapter=with_adapter
        )
        return out + tokens


class LocoValTransformer(nn.Module):
    """Physical Core + social module + Path-Head + LocoVal-Head."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        assert config.peft_type in ("adapter", "fix_local", "fft", "none"), (
            config.peft_type
        )
        assert config.st_type in ("scratch", "none"), config.st_type
        assert config.position_encoding in ("relative", "none"), (
            config.position_encoding
        )
        self.config = config
        self.input_track_size = config.input_track_size
        self.output_track_size = config.output_track_size
        self.obs_and_pred = self.input_track_size + self.output_track_size
        self.output_scale = config.output_scale
        self.nhid = config.dim_hidden
        self.nmode = config.nmode
        self.stride = int(max(1, config.inputs.max_fps // config.inputs.fps))

        self._cross_modality_transformer = CrossModalityTransformer(config)
        layer = EncoderLayer(
            self.nhid,
            config.num_heads,
            config.dim_feedforward,
            config.dropout,
            config.activation,
        )
        self.global_former = Encoder(
            layer, num_layers=config.num_layers_global, enable_nested_tensor=False
        )
        self.id_encoder = LearnedIDEncoding(
            self.nhid, config.dropout, seq_len=self.obs_and_pred
        )
        self.relative_encoder = TargetRelativePositionEncoding(
            self.nhid, config.dropout
        )
        self.reward_head = RewardHead(self.nhid)
        self.reward_masking = config.reward_masking
        self.predict_head = nn.ModuleList(
            [nn.Linear(self.nhid, 2) for _ in range(self.nmode)]
        )

    def normalize_input(
        self, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """tgt: (B, in_F, N * (joints + 2), 4) world tokens -> normalized tokens and target-relative offsets."""
        B, in_F, NJ, K = tgt.shape
        J = self.config.joints_3dpose + 1
        N = (NJ + J) // (J + 1)
        tgt = tgt.reshape(B, in_F, N, J + 1, K).permute(
            0, 2, 1, 3, 4
        )  # (B, N, in_F, J+1, K)
        if self.config.inputs.agent_relative:
            rel = (tgt[:, :, :, 0] - tgt[:, 0:1, :, 0]).permute(
                0, 2, 1, 3
            )  # (B, in_F, N, K)
            traj = tgt[:, :, :, 0:1] - tgt[:, :, in_F - 1 : in_F, 0:1]
        else:
            rel = None
            traj = tgt[:, :, :, 0:1] - tgt[:, 0:1, in_F - 1 : in_F, 0:1]
        pose = tgt[:, :, :, 2:] - tgt[:, :, :, 1:2]
        tgt = (
            torch.cat((traj, pose), dim=3)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, in_F, N * J, K)
        )
        return tgt, rel, N

    def encode(
        self,
        tgt: torch.Tensor,
        padding_mask: torch.Tensor,
        limit_obs: int,
        random_masking: bool,
        with_adapter: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Physical Core features (with the adapters in Prediction Mode, without in Evaluation Mode)."""
        tgt, rel, N = self.normalize_input(tgt)
        feature = self._cross_modality_transformer(
            tgt, padding_mask, random_masking, limit_obs, with_adapter
        )
        return feature, rel, N

    def forward(
        self,
        tgt: torch.Tensor,
        padding_mask: torch.Tensor,
        nmode: int,
        limit_obs: int = 0,
        random_masking: bool = False,
    ) -> torch.Tensor:
        """Prediction Mode. Returns path hypotheses (B, in_F + out_F, nmode, 2) relative to the last observation."""
        feature, rel, N = self.encode(
            tgt, padding_mask, limit_obs, random_masking, with_adapter=True
        )
        return self.predict_traj(
            feature,
            padding_mask,
            self.mask_relative(rel, tgt, N, limit_obs, random_masking),
            nmode,
        )

    def plausibility(
        self,
        tgt: torch.Tensor,
        padding_mask: torch.Tensor,
        paths: torch.Tensor,
        limit_obs: int = 0,
    ) -> torch.Tensor:
        """Evaluation Mode. Scores paths (B, out_F, n, 2) of the target agent; returns (B, n)."""
        feature, _, _ = self.encode(
            tgt, padding_mask, limit_obs, random_masking=False, with_adapter=False
        )
        return self.estimate_reward(feature, paths)

    def pretraining_outputs(
        self,
        tgt: torch.Tensor,
        padding_mask: torch.Tensor,
        paths: torch.Tensor,
        random_masking: bool,
        nmode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage-1 outputs: predicted paths (B, F, nmode, 2), their scores (B, nmode),
        the scores of the executed paths `paths` (B, out_F, 1, 2) as (B, 1), and the
        Evaluation-Mode features used to score other paths."""
        feature_eval, _, _ = self.encode(
            tgt, padding_mask, 0, random_masking, with_adapter=False
        )
        feature_pred, rel, N = self.encode(
            tgt, padding_mask, 0, random_masking, with_adapter=True
        )
        cond_rew = self.estimate_reward(feature_eval, paths)
        rel = self.mask_relative(rel, tgt, N, 0, random_masking)
        pred = self.predict_traj(feature_pred, padding_mask, rel, nmode)
        pred_rew = self.estimate_reward(feature_eval, pred[:, self.input_track_size :])
        return pred, pred_rew, cond_rew, feature_eval

    def mask_relative(
        self,
        rel: Optional[torch.Tensor],
        tgt: torch.Tensor,
        N: int,
        limit_obs: int,
        random_masking: bool,
    ) -> torch.Tensor:
        """Marks the target-relative offsets of unobserved frames as NaN."""
        if rel is None:
            return torch.full(
                (tgt.shape[0], self.input_track_size, N, 4), np.nan, device=tgt.device
            )
        if limit_obs != 0:
            rel[:, : self.input_track_size - limit_obs] = np.nan
        elif random_masking:
            ratio = self.config.inputs.mask_ratio_frame
            ratio = torch.rand(1).to(tgt.device) if ratio == 1.0 else ratio
            keep = (
                torch.rand((tgt.shape[0], self.input_track_size - 1), device=tgt.device)
                > ratio
            )
            keep = keep.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, 4).bool()
            rel[:, : self.input_track_size - 1][~keep] = np.nan
        return rel

    def predict_traj(
        self,
        feature: torch.Tensor,
        padding_mask: torch.Tensor,
        rel: torch.Tensor,
        nmode: int,
    ) -> torch.Tensor:
        """Social module and Path-Head on top of the Physical Core features."""
        B, N = padding_mask.shape
        num_frames = self.obs_and_pred
        local = (
            feature[:num_frames]
            .reshape(num_frames, B, N, self.nhid)
            .permute(2, 0, 1, 3)
        )  # (N, F, B, D)
        if self.config.st_type != "none":
            x = self.id_encoder(local, num_people=N).reshape(-1, B, self.nhid)
            if self.config.position_encoding == "relative":
                x = x + self.relative_encoder(rel, padding_mask, num_frames)
            key_padding = padding_mask.repeat_interleave(num_frames, dim=1).bool()
            out = self.global_former(x, src_key_padding_mask=key_padding)
            out = out * self.output_scale + local.reshape(-1, B, self.nhid)
        else:
            out = local
        primary = out.reshape(N, num_frames, B, self.nhid)[0]
        assert 1 <= nmode <= self.nmode
        return torch.stack(
            [self.predict_head[i](primary) for i in range(nmode)], dim=2
        ).transpose(0, 1)

    def estimate_reward(self, feature: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """LocoVal-Head on the future-frame features of the target agent; paths: (B, out_F, n, 2)."""
        B = paths.shape[0]
        primary = (
            feature[self.input_track_size : self.obs_and_pred]
            .reshape(self.output_track_size, B, -1, self.nhid)
            .permute(2, 0, 1, 3)[0]
        )
        return self.reward_head(
            primary, paths, self.stride, self.reward_masking
        )


def create_model(config: DictConfig, device: str) -> LocoValTransformer:
    return LocoValTransformer(config).float().to(device)
