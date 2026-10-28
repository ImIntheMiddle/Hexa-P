"""Social-Transmotion (Saadatnejad et al., ICLR 2024) on SMPL-24 pose tokens with K path heads.

This is the baseline network of the JRDB comparison. It follows the original
architecture (https://github.com/vita-epfl/social-transmotion): a per-agent
cross-modality Transformer over path and pose tokens, followed by a social
Transformer over the path tokens of all agents.

This file is derived from Social-Transmotion and is distributed under the GNU
AGPL-3.0 (src/baselines/LICENSE.social_transmotion).
"""

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


class Encoder(nn.TransformerEncoder):
    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=None, src_key_padding_mask=src_key_padding_mask)
        return x


class LearnedTrajAndIDEncoding(nn.Module):
    """Time embedding on even channels and agent embedding on odd channels."""

    def __init__(self, d_model: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned_encoding = nn.Embedding(seq_len, d_model // 2, max_norm=True)
        self.person_encoding = nn.Embedding(1000, d_model // 2, max_norm=True)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, num_people: int) -> torch.Tensor:
        """x: (B, F, N, D)."""
        half = x.size(3) // 2
        t = torch.arange(self.seq_len, device=x.device)
        x[:, :, :, 0 : half * 2 : 2] = x[
            :, :, :, 0 : half * 2 : 2
        ] + self.learned_encoding(t).unsqueeze(1).unsqueeze(0)
        ids = (
            torch.arange(num_people, device=x.device)
            .unsqueeze(0)
            .repeat_interleave(self.seq_len, dim=0)
        )
        x[:, :, :, 1 : half * 2 : 2] = x[
            :, :, :, 1 : half * 2 : 2
        ] + self.person_encoding(ids).unsqueeze(0)
        return self.dropout(x)


class LearnedPoseEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.arange(self.seq_len, device=x.device)
        return self.dropout(x + self.learned_encoding(t).unsqueeze(1).unsqueeze(0))


class CrossModalityTransformer(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.seq_len = config.seq_len
        self.nhid = config.dim_hidden
        self.output_scale = config.output_scale
        self.obs = config.input_track_size
        self.obs_and_pred = config.input_track_size + config.output_track_size
        self.token_num = config.token_num
        self.joints_3dpose = config.joints_3dpose
        self.fc_in_traj = nn.Linear(2, self.nhid)
        self.double_id_encoder = LearnedTrajAndIDEncoding(
            self.nhid, config.dropout, seq_len=self.obs_and_pred
        )
        self.fc_in_3dpose = nn.Linear(3, self.nhid)
        self.pose3d_encoder = LearnedPoseEncoding(
            self.nhid, config.dropout, seq_len=self.joints_3dpose * self.obs
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.nhid,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
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
    ) -> torch.Tensor:
        B, in_F, NJ, K = tgt.shape
        F = self.obs_and_pred
        J = self.token_num
        N = NJ // J
        device = tgt.device
        idx = np.append(np.arange(0, in_F), np.repeat([in_F - 1], F - in_F))
        tgt = torch.nan_to_num(tgt[:, idx].reshape(B, F, N, J, K))

        mask_ratio_traj = 0.1 if random_masking else 0
        mask_ratio_joints = 0.1 if random_masking else 0
        mask_ratio_modality = 0.3 if random_masking else 0

        tgt_traj = tgt[:, :, :, 0, :2]
        traj_mask = (
            torch.rand((B, F, N), device=device).float() > mask_ratio_traj
        ).unsqueeze(3)
        tgt_traj = tgt_traj * traj_mask.repeat_interleave(2, dim=-1)
        modality = (
            torch.rand((B, 1, N, 1), device=device).float() > mask_ratio_modality
        ).unsqueeze(4)
        tgt_vis = tgt[:, :, :, 1:] * modality.repeat(1, F, 1, self.joints_3dpose, 4)
        tgt_pose = tgt_vis[:, :, :, : self.joints_3dpose, :3]
        joint_mask = (
            torch.rand((B, F, N, self.joints_3dpose), device=device).float()
            > mask_ratio_joints
        )
        tgt_pose = tgt_pose * joint_mask.unsqueeze(4).repeat_interleave(3, dim=-1)
        if limit_obs != 0:
            limit = torch.ones((B, F, N), device=device).float()
            limit[:, : in_F - limit_obs] = 0
            tgt_traj = tgt_traj * limit.unsqueeze(3).repeat_interleave(2, dim=-1)
            tgt_pose = tgt_pose * limit.unsqueeze(3).repeat_interleave(
                self.joints_3dpose, dim=-1
            ).unsqueeze(4).repeat_interleave(3, dim=-1)

        tgt_traj = self.double_id_encoder(self.fc_in_traj(tgt_traj), num_people=N)
        tgt_pose = tgt_pose[:, :in_F].transpose(2, 3).reshape(B, -1, N, 3)
        tgt_pose = self.pose3d_encoder(self.fc_in_3dpose(tgt_pose))
        key_padding = (
            padding_mask.reshape(-1)
            .unsqueeze(1)
            .repeat_interleave(self.seq_len, dim=1)
            .float()
        )
        tgt_traj = torch.transpose(tgt_traj, 0, 1).reshape(F, -1, self.nhid)
        tgt_pose = torch.transpose(tgt_pose, 0, 1).reshape(
            in_F * self.joints_3dpose, -1, self.nhid
        )
        tokens = torch.cat((tgt_traj, tgt_pose), 0)
        return (
            self.local_former(tokens, src_key_padding_mask=key_padding)
            * self.output_scale
            + tokens
        )


class SocialTransmotion(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.obs_and_pred = config.input_track_size + config.output_track_size
        self.output_scale = config.output_scale
        self.nhid = config.dim_hidden
        self.nmode = config.nmode
        self._cross_modality_transformer = CrossModalityTransformer(config)
        layer = nn.TransformerEncoderLayer(
            d_model=self.nhid,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.global_former = Encoder(
            layer, num_layers=config.num_layers_global, enable_nested_tensor=False
        )
        self.predict_head = nn.ModuleList(
            [nn.Linear(self.nhid, 2) for _ in range(self.nmode)]
        )

    def normalize_input(self, tgt: torch.Tensor) -> torch.Tensor:
        B, in_F, NJ, K = tgt.shape
        J = self.config.joints_3dpose + 1
        N = NJ // J
        tgt = tgt.reshape(B, in_F, N, J, K).permute(0, 2, 1, 3, 4)
        tgt[:, :, :, 0] = tgt[:, :, :, 0] - tgt[:, 0:1, in_F - 1 : in_F, 0]
        tgt[:, :, :, 1:] = tgt[:, :, :, 1:] - tgt[:, :, in_F - 1 : in_F, 1:2]
        return tgt.transpose(1, 2).reshape(B, in_F, NJ, K)

    def forward(
        self,
        tgt: torch.Tensor,
        padding_mask: torch.Tensor,
        nmode: int,
        limit_obs: int = 0,
        random_masking: bool = False,
    ) -> torch.Tensor:
        """Returns path hypotheses (B, in_F + out_F, nmode, 2) relative to the last observation."""
        local = self._cross_modality_transformer(
            self.normalize_input(tgt), padding_mask, random_masking, limit_obs
        )
        B, N = padding_mask.shape
        F = self.obs_and_pred
        local = (
            local[:F]
            .reshape(F, B, N, self.nhid)
            .permute(2, 0, 1, 3)
            .reshape(-1, B, self.nhid)
        )
        key_padding = padding_mask.repeat_interleave(F, dim=1).float()
        out = (
            self.global_former(local, src_key_padding_mask=key_padding)
            * self.output_scale
            + local
        )
        primary = out.reshape(N, F, B, self.nhid)[0]
        assert 1 <= nmode <= self.nmode
        return torch.stack(
            [self.predict_head[i](primary) for i in range(nmode)], dim=2
        ).transpose(0, 1)


def create_model(config: DictConfig, device: str) -> SocialTransmotion:
    return SocialTransmotion(config).float().to(device)
