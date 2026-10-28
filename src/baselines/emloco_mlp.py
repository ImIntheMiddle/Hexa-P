"""Plausibility estimator of EmLoco (Taketsugu et al., CVPR 2025) used by the EmLoco baseline.

The MLP scores a future path from the current pose and velocity of the target
agent, after rotating everything so that the path starts along the x axis. It is
trained in the physics simulation (`pretraining=True`, at the simulation frame
rate) and frozen when it supervises the EmLoco baseline in Stage 2.
"""

from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from hexap.models.locoval_transformer import AVAILABLE_JOINTS


class EmLocoMLP(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.output_track_size = config.output_track_size
        self.joints_3dpose = config.joints_3dpose
        self.stride = (
            int(max(1, config.inputs.max_fps // config.inputs.fps))
            if config.pretraining
            else 1
        )
        self.available_joints = torch.tensor(AVAILABLE_JOINTS[config.inputs.pose_mask])
        size = (
            self.output_track_size // self.stride * 2
            + len(self.available_joints) * 3
            + 2
        )
        self._network = nn.Sequential()
        self._network.add_module("fc1", nn.Linear(size, size // 2))
        self._network.add_module("relu1", nn.ReLU())
        self._network.add_module("fc2", nn.Linear(size // 2, size // 4))
        self._network.add_module("relu2", nn.ReLU())
        self._network.add_module("fc3", nn.Linear(size // 4, 1))
        self._network.add_module("sigmoid", nn.Sigmoid())

    def normalize_input(
        self, tgt: torch.Tensor, paths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """tgt: (BM, in_F, N * 25, 4); paths: (BM, out_F, 2). Returns pose, velocity and path in the path frame."""
        BM, in_F, NJ, K = tgt.shape
        J = self.joints_3dpose + 1
        N = (NJ + J) // (J + 1)
        tgt = tgt.reshape(BM, in_F, N, J + 1, K).permute(0, 2, 1, 3, 4)
        last = tgt[:, 0, -1]
        pose = (last[:, 2:, :3] - last[:, 1:2, :3])[:, self.available_joints]
        vel = tgt[:, 0, -1, 0, :2] - tgt[:, 0, -2, 0, :2]
        idx = (self.stride - 1) + torch.arange(
            self.output_track_size // self.stride, device=paths.device
        ) * self.stride
        paths = paths[:, idx, :2]

        eps = 1e-8
        x = paths[:, 1, 0]
        x = torch.where(
            x.abs() < eps, torch.sign(x) * eps + (x.abs() < 1e-10).float() * eps, x
        )
        angle = torch.atan2(paths[:, 1, 1], x)
        c, s = torch.cos(angle), torch.sin(angle)
        rot = torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2
        )  # (BM, 2, 2)
        pose_rot = pose.clone()
        pose_rot[..., :2] = torch.bmm(pose[:, :, :2], rot)
        return pose_rot, torch.bmm(vel.unsqueeze(1), rot)[:, 0], torch.bmm(paths, rot)

    def forward(self, tgt: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """tgt: (B, in_F, N * 25, 4); paths: (B, out_F, M, 2). Returns plausibility scores (B, M)."""
        B, out_F, M, K = paths.shape
        tgt = (
            torch.nan_to_num(tgt)
            .unsqueeze(1)
            .expand(-1, M, -1, -1, -1)
            .reshape(B * M, *tgt.shape[1:])
        )
        pose, vel, paths = self.normalize_input(
            tgt, paths.transpose(1, 2).reshape(B * M, out_F, K)
        )
        x = torch.cat(
            [paths.reshape(B * M, -1), pose.reshape(B * M, -1), vel.reshape(B * M, -1)],
            dim=-1,
        )
        return self._network(x).reshape(B, M)
