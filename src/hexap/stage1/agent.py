"""Agents of the physics pre-training (Stage 1).

A frozen path-following controller (an AMP policy trained with
`exp=hexap/controller`) walks along random target paths in IsaacGym. After each
rollout, `HexaPPretrainer` trains the Hexa-P network on the executed locomotion:
the Path-Head regresses the executed path from the observed motion, and the
LocoVal-Head regresses the normalized discounted return of the rollout given
its target path. `EmLocoPretrainer` trains only the plausibility estimator of
the EmLoco baseline on the same data.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from hydra.utils import instantiate
from lightning.fabric import Fabric
from protomotions.agents.amp.agent import AMP
from protomotions.agents.common.common import weight_init
from protomotions.agents.utils.data_utils import DictDataset
from protomotions.envs.base_env.env import BaseEnv
from protomotions.utils.running_mean_std import RunningMeanStd
from torch import nn
from torchmetrics.regression import PearsonCorrCoef

from baselines.emloco_mlp import EmLocoMLP
from hexap.models.locoval_transformer import LocoValTransformer
from hexap.stage1.utils import (
    CosineAnnealingLR,
    PCGradOptimizer,
    QuantileNormalizer,
    UncertaintyWeighting,
    variety_loss,
)

log = logging.getLogger(__name__)


class HexaPPretrainer(AMP):
    def __init__(self, fabric: Fabric, env: BaseEnv, config) -> None:
        super().__init__(fabric, env, config)
        self.running_return_norm = RunningMeanStd(shape=(1,), device=self.device)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.force_full_restart = True

    def build_network(self) -> nn.Module:
        return LocoValTransformer(self.config.locoval.config)

    def setup(self) -> None:
        model = instantiate(self.config.model)
        locoval = self.build_network().to(self.device)
        model.apply(weight_init)
        locoval.apply(weight_init)
        cfg = self.config.locoval.config
        params: Union[List, List[Dict]] = list(locoval.parameters())
        if self.config.enable_unc_weighting:
            locoval.add_module("unc_weight", UncertaintyWeighting().to(self.device))
            params = [
                {"params": locoval.unc_weight.parameters(), "lr": 1e-3},
                {
                    "params": [
                        p
                        for n, p in locoval.named_parameters()
                        if "unc_weight" not in n
                    ],
                    "lr": cfg.locoval_optimizer.lr,
                },
            ]
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.locoval_optimizer.lr,
            weight_decay=cfg.locoval_optimizer.weight_decay,
        )
        if self.config.enable_pcgrad:
            optimizer = PCGradOptimizer(optimizer)
        scheduler = CosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.locoval_scheduler.warmup_epochs,
            max_epochs=cfg.locoval_scheduler.max_epochs,
        )
        self.pearson = PearsonCorrCoef()
        # Fabric wraps only the frozen controller; the pre-trained network is optimized directly.
        self.model = self.fabric.setup(model)
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")
        self.locoval, self.locoval_optimizer, self.locoval_scheduler = locoval, optimizer, scheduler
        self.disc_reward_norm = QuantileNormalizer().to(self.device)

    # -----------------------------
    # Checkpoints
    # -----------------------------
    def load(self, checkpoint: Path) -> None:
        """Loads the frozen path-following controller, and optionally a Stage-1 network to continue from."""
        if checkpoint is not None:
            state = torch.load(Path(checkpoint).resolve(), map_location=self.device)
            self.model.load_state_dict(state["model"])
            if self.config.normalize_values:
                self.running_val_norm.load_state_dict(state["running_val_norm"])
        if self.config.get("locoval_checkpoint") is not None:
            self.fabric.load(
                Path(self.config.locoval_checkpoint).resolve(),
                {
                    "locoval": self.locoval,
                    "running_return_norm": self.running_return_norm,
                    "disc_reward_norm": self.disc_reward_norm,
                    "locoval_optimizer": self.locoval_optimizer,
                },
                strict=False,
            )

    def get_state_dict(self, state_dict: Dict) -> Dict:
        state_dict.update(
            {
                "epoch": self.current_epoch,
                "locoval": self.locoval.state_dict(),
                "locoval_optimizer": self.locoval_optimizer.state_dict(),
                "running_return_norm": self.running_return_norm.state_dict(),
                "best_evaluated_score": self.best_evaluated_score,
                "disc_reward_norm": self.disc_reward_norm.state_dict(),
            }
        )
        return state_dict

    def save(
        self, path=None, name: str = "locoval_last", new_high_score: bool = False
    ) -> None:
        """Writes `<name>_<epoch>epoch.ckpt` into the experiment directory."""
        save_dir = Path.cwd() / Path(
            self.fabric.loggers[0].root_dir if path is None else path
        )
        self.fabric.save(
            save_dir / f"{name}_{self.current_epoch}epoch.ckpt", self.get_state_dict({})
        )

    # -----------------------------
    # Rollouts
    # -----------------------------
    def register_extra_experience_buffer_keys(self) -> None:
        super().register_extra_experience_buffer_keys()
        self.experience_buffer.register_key("env_ids")

    def handle_reset(self, done_indices=None):
        if self.force_full_restart:
            # Every epoch starts all episodes together; remember which motion and time each one starts from.
            obs = self.env.reset(None)
            self.force_full_restart = False
            self.locoval_motion_ids = self.env.motion_manager.motion_ids
            self.locoval_motion_times = self.env.motion_manager.motion_times
            return obs
        return self.env.reset(done_indices)

    def post_train_env_step(self, rewards, dones, done_indices, extras, step) -> None:
        self.experience_buffer.update_data("env_ids", step, self.env_ids)
        if step >= self.num_steps - 1:
            done_indices = torch.arange(
                self.num_envs, device=self.env.device, dtype=torch.long
            )
            dones = torch.ones_like(done_indices)
        super().post_train_env_step(rewards, dones, done_indices, extras, step)

    # -----------------------------
    # Optimization
    # -----------------------------
    def max_num_batches(self) -> int:
        return math.ceil(self.num_envs * self.num_mini_epochs / self.config.batch_size)

    def optimize_model(self) -> Dict[str, float]:
        self.train()
        tensor_dict = self.experience_buffer.make_dict()
        self.calc_disc_cum_rewards(tensor_dict["total_rewards"], tensor_dict["dones"])
        dataset = self.process_dataset(tensor_dict)
        self.pred_stack, self.gt_stack = [], []
        sums: Dict[str, List[float]] = {}
        for batch_idx in range(self.max_num_batches()):
            dataset_idx = batch_idx % len(dataset)
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch = dataset[dataset_idx]
            for key, value in batch.items():
                if torch.isnan(value).any():
                    raise ValueError(f"NaN in {key}")
            loss, log_dict = self.locoval_step(batch, random_masking=True)
            if self.config.enable_pcgrad:
                self.locoval_optimizer.zero_grad()
                self.locoval_optimizer.pc_backward(loss)
            else:
                self.locoval_optimizer.zero_grad(set_to_none=True)
                self.fabric.backward(loss)
            log_dict.update(
                self.handle_model_grad_clipping(
                    self.locoval, self.locoval_optimizer, "locoval"
                )
            )
            self.locoval_optimizer.step()
            for k, v in log_dict.items():
                sums.setdefault(k, []).append(float(v))
        out = {k: sum(v) / len(v) for k, v in sums.items()}
        self.locoval_scheduler.step()
        self.corr_pred_gt = self.pearson(
            torch.cat(self.pred_stack).flatten(), torch.cat(self.gt_stack).flatten()
        ).item()
        out["corr_pred_gt"] = self.corr_pred_gt
        self.eval()
        self.force_full_restart = True
        return out

    def locoval_step(
        self, batch: Dict, random_masking: bool
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict]:
        cfg = self.config.locoval.config
        in_F, out_F = cfg.input_track_size, cfg.output_track_size
        env_ids = batch["env_ids"].to(torch.int64)
        tgt, padding_mask, target_path = self.get_locoval_obs(batch)
        B = tgt.shape[0]
        pred, pred_rew, cond_rew, feature_eval = self.locoval.pretraining_outputs(
            tgt,
            padding_mask,
            target_path.reshape(B, out_F, 1, 2),
            random_masking,
            cfg.nmode,
        )
        executed = self.create_gt_traj(batch)  # (B, out_F, 1, 3)
        returns = self.disc_cum_rewards[env_ids]
        ok = (
            (executed[..., 2] == 1).squeeze(dim=2).all(dim=1)
        )  # rollouts without a fall

        # Path loss on the executed paths of the rollouts without a fall, weighted by their return.
        traj_loss = variety_loss(pred[ok, in_F:], executed[ok])[0]
        traj_loss = (
            (returns[ok] ** self.config.locoval_weight_beta) * traj_loss
        ).mean()
        locoval_loss = nn.functional.mse_loss(cond_rew.squeeze(), returns) * 1000
        emloco_loss = nn.functional.mse_loss(pred_rew, torch.ones_like(pred_rew))
        if self.config.enable_unc_weighting:
            traj_loss, locoval_loss = self.locoval.unc_weight.compute_loss(
                traj_loss, locoval_loss
            )
        gt_rew = self.locoval.estimate_reward(
            feature_eval[:, ok],
            executed[ok, ..., 0, :2].reshape(int(ok.sum()), out_F, 1, 2),
        )
        barrier_loss = -torch.log(
            torch.sigmoid(gt_rew + self.config.locoval_barrier_b - cond_rew[ok]).clamp(
                min=1e-4
            )
        ).mean()
        w = self.config.locoval_loss_w
        losses = [
            locoval_loss * w,
            traj_loss * (1 - w),
            emloco_loss * self.config.locoval_emloco_w * (1 - w),
            barrier_loss * self.config.locoval_barrier_w * w,
        ]
        self.pred_stack.append(cond_rew.detach().cpu())
        self.gt_stack.append(returns.detach().cpu())
        log_dict = {
            "locoval_loss": locoval_loss.item(),
            "traj_loss": traj_loss.item(),
            "emloco_loss": emloco_loss.item(),
            "barrier_loss": barrier_loss.item(),
            "cond_rew_mean": cond_rew.mean().item(),
            "num_fall": B - int(ok.sum()),
        }
        return (losses if self.config.enable_pcgrad else sum(losses)), log_dict

    # -----------------------------
    # Observations and targets
    # -----------------------------
    def create_gt_traj(self, batch: Dict) -> torch.Tensor:
        """Executed path relative to its start; samples after a fall repeat the last reached position."""
        wp = batch["actual_waypoints"]  # (B, waypoints, 3)
        unreached = torch.where(wp[:, :, 2] != 1)
        last = (wp[:, :, 2] == 1).sum(dim=1) - 1
        wp[unreached[0], unreached[1], :2] = wp[unreached[0], last[unreached[0]], :2]
        wp[..., :2] = wp[..., :2] - wp[:, :1, :2]
        return wp[:, 1:].reshape(
            self.config.batch_size, self.config.locoval.config.output_track_size, 1, 3
        )

    def get_locoval_obs(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Observed motion of the reference clip before the rollout, and the target path relative to its start."""
        cfg = self.config.locoval.config
        B = self.config.batch_size
        tgt = torch.zeros(
            (B, cfg.input_track_size, self.motion_lib.robot_config.num_bodies + 1, 4)
        )
        past_traj, past_poses = self.get_expert_historical_locoval_obs(batch)
        target_path = batch["path_non_rotated"].reshape(
            B, cfg.output_track_size + 1, -1
        )
        target_path = (target_path[:, 1:, :2] - target_path[:, :1, :2]).reshape(
            B, cfg.output_track_size * 2
        )
        tgt[:, :, 0, :2] = past_traj
        tgt[:, :, 1:, :3] = past_poses
        return (
            tgt.to(self.device),
            torch.zeros((B, 1), dtype=torch.bool).to(self.device),
            target_path.to(self.device),
        )

    def get_expert_historical_locoval_obs(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        env_ids = batch["env_ids"].to(torch.int64).to(self.env.device)
        motion_ids = self.locoval_motion_ids[env_ids]
        motion_times = self.locoval_motion_times[env_ids]
        steps = self.config.locoval_historical_steps
        stride = steps // self.config.locoval_historical_conditioned_steps
        offsets = (
            (torch.arange(steps, device=self.env.device) * self.env.dt)[::stride]
            .unsqueeze(0)
            .expand(len(env_ids), -1)
        )
        past_times = (motion_times.unsqueeze(1) - offsets).flip(dims=[1])
        valid = (
            past_times >= 0
        )  # earlier times are clipped to the first frame of the clip
        flat_ids = motion_ids.unsqueeze(-1).tile([1, past_times.shape[1]]).view(-1)
        bodies = self.env.motion_lib.get_motion_state(
            flat_ids, past_times.view(-1)
        ).rigid_body_pos
        bodies = bodies.view(len(env_ids), past_times.shape[1], -1, 3)
        past_root = bodies[:, :, 0, :2].masked_fill(~valid.unsqueeze(-1), 0)
        past_poses = bodies[:, :, :, :3].masked_fill(
            ~valid.view(len(env_ids), -1, 1, 1), 0
        )
        return past_root, past_poses

    @torch.no_grad()
    def process_dataset(self, dataset: Dict) -> DictDataset:
        """Keeps the first step of every episode, and the waypoints recorded at its end."""
        first = torch.arange(self.num_envs, device=self.device) * self.num_steps
        last = first + self.first_done
        out = {}
        for key, tensor in dataset.items():
            if key == "actual_waypoints":
                wp = tensor[last].reshape(self.num_envs, -1, 3)
                nz = torch.where(torch.norm(wp[..., :2], dim=-1) > 1e-6)
                wp[nz[0], nz[1], :2] -= wp[nz[0], 0, :2]
                out[key] = wp
            else:
                out[key] = tensor[first]
        return DictDataset(self.config.batch_size, out, shuffle=True)

    def calc_disc_cum_rewards(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Discounted return of each episode up to its first termination, normalized to [0, 1]."""
        rewards = rewards.view(self.num_envs, self.num_steps)
        dones = dones.view(self.num_envs, self.num_steps).bool()
        self.first_done = torch.full(
            (self.num_envs,), self.num_steps, device=rewards.device, dtype=torch.long
        )
        for env_idx in range(self.num_envs):
            idx = dones[env_idx].nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                self.first_done[env_idx] = idx[0]
        steps = torch.arange(self.num_steps, device=rewards.device).unsqueeze(0)
        masked = rewards * (steps <= self.first_done.unsqueeze(1)).float()
        disc = torch.sum(masked * self.gamma**steps, dim=1)
        self.running_return_norm.update(disc)
        self.disc_cum_rewards = self.disc_reward_norm(disc)

    def train(self) -> None:
        self.model.eval()  # the controller only rolls out
        self.locoval.train()
        self.disc_reward_norm.train()

    def eval(self) -> None:
        self.model.eval()
        self.locoval.eval()
        self.disc_reward_norm.eval()

    @torch.no_grad()
    def calc_eval_metrics(self):
        return {"corr_pred_gt": self.corr_pred_gt}, self.corr_pred_gt

    def post_epoch_logging(self, training_log_dict: Dict) -> None:
        training_log_dict["rewards/disc_cum_rewards_mean"] = (
            self.disc_cum_rewards.mean()
        )
        training_log_dict["rewards/disc_reward_norm_low"] = (
            self.disc_reward_norm.running_low.item()
        )
        training_log_dict["rewards/disc_reward_norm_high"] = (
            self.disc_reward_norm.running_high.item()
        )
        super().post_epoch_logging(training_log_dict)


class EmLocoPretrainer(HexaPPretrainer):
    """Trains the plausibility estimator of the EmLoco baseline on the same rollouts."""

    def build_network(self) -> nn.Module:
        return EmLocoMLP(self.config.locoval.config)

    def setup(self) -> None:
        model = instantiate(self.config.model)
        locoval = self.build_network().to(self.device)
        model.apply(weight_init)
        locoval.apply(weight_init)
        cfg = self.config.locoval.config
        optimizer = torch.optim.AdamW(
            locoval.parameters(),
            lr=cfg.locoval_optimizer.lr,
            weight_decay=cfg.locoval_optimizer.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.locoval_scheduler.warmup_epochs,
            max_epochs=cfg.locoval_scheduler.max_epochs,
        )
        self.pearson = PearsonCorrCoef()
        self.model = self.fabric.setup(model)
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")
        self.locoval, self.locoval_optimizer, self.locoval_scheduler = locoval, optimizer, scheduler
        self.disc_reward_norm = QuantileNormalizer().to(self.device)

    def save(
        self, path=None, name: str = "emloco_last", new_high_score: bool = False
    ) -> None:
        super().save(path, name, new_high_score)

    def locoval_step(
        self, batch: Dict, random_masking: bool
    ) -> Tuple[torch.Tensor, Dict]:
        cfg = self.config.locoval.config
        env_ids = batch["env_ids"].to(torch.int64)
        tgt, _, target_path = self.get_locoval_obs(batch)
        B = tgt.shape[0]
        score = self.locoval(
            tgt, target_path.reshape(B, cfg.output_track_size, 1, 2)
        )  # (B, 1)
        returns = self.disc_cum_rewards[env_ids]
        loss = nn.functional.mse_loss(score.squeeze(), returns) * 1000
        self.pred_stack.append(score.detach().cpu())
        self.gt_stack.append(returns.detach().cpu())
        return loss, {"locoval_loss": loss.item(), "cond_rew_mean": score.mean().item()}
