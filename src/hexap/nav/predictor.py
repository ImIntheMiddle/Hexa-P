"""Path prediction of the humans in the replayed scene.

`PathPredictor` keeps the last 9 frames (2.5 fps) of the position and the 3D pose of
every human, and once they are full it predicts M path hypotheses for each human with
a Stage-2 checkpoint, optionally with their LocoVal scores.
"""

from collections import deque
from typing import Deque, NamedTuple, Optional, Tuple

import numpy as np
import torch

from hexap.models import load_checkpoint
from hexap.nav.env import ReplayEnv

# Predicted frame the planner avoids: the 5th future frame, 2.0 s ahead at 2.5 fps.
LOOKAHEAD = 4
# Humans farther than this from the target human are padded out of its context.
CONTEXT_RADIUS = 6.0
# Tokens per human: path, pelvis and 23 joints.
TOKENS = 25


class Prediction(NamedTuple):
    positions: np.ndarray  # (humans, M, 2) predicted position at the lookahead frame
    scores: Optional[np.ndarray]  # (humans, M) LocoVal score of each hypothesis


class PathPredictor:
    def __init__(
        self, checkpoint: str, device: str, max_human_num: int, with_scores: bool
    ) -> None:
        self.model, config = load_checkpoint(checkpoint, device)
        assert config.model.joints_3dpose == 23
        self.device = device
        self.max_human_num = max_human_num
        self.with_scores = with_scores
        self.obs_len = config.model.input_track_size
        self.nmode = config.model.nmode
        self.traj_buf: Deque[np.ndarray] = deque(maxlen=self.obs_len)
        self.mask_buf: Deque[np.ndarray] = deque(maxlen=self.obs_len)
        self.pose_buf: Deque[np.ndarray] = deque(maxlen=self.obs_len)

    def reset(self) -> None:
        self.traj_buf.clear()
        self.mask_buf.clear()
        self.pose_buf.clear()

    def update(self, env: ReplayEnv) -> None:
        """Appends the current frame of the replay."""
        N = self.max_human_num
        pos = np.zeros((N, 2), np.float32)
        mask = np.zeros((N, 1), np.float32)
        pose = np.zeros((N, 24, 3), np.float32)
        n = min(env.human_num, N)
        for i in range(n):
            pos[i, 0] = env.humans[i].px
            pos[i, 1] = env.humans[i].py
        mask[:n] = 1.0
        for i in range(n):
            joints = env.current_scene[i][0]
            pose[i] = joints[env.current_frame, 1:25, :3].float().numpy()
        self.traj_buf.append(pos)
        self.mask_buf.append(mask)
        self.pose_buf.append(pose)

    def predict(self) -> Optional[Prediction]:
        """None until the buffer holds the observed frames."""
        if len(self.traj_buf) < self.obs_len:
            return None
        traj = torch.tensor(np.stack(self.traj_buf)).unsqueeze(0).permute(0, 2, 1, 3)
        mask = torch.tensor(np.stack(self.mask_buf)).unsqueeze(0).permute(0, 2, 1, 3)
        pose = torch.tensor(np.stack(self.pose_buf)).unsqueeze(0).permute(0, 2, 1, 3, 4)
        positions, scores = self._predict(traj, mask, pose)
        return Prediction(
            positions.cpu().numpy(),
            scores.cpu().numpy() if scores is not None else None,
        )

    @torch.no_grad()
    def _predict(
        self, traj: torch.Tensor, mask: torch.Tensor, pose: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """traj (1, N, T, 2), mask (1, N, T, 1), pose (1, N, T, 24, 3) -> (N, M, 2), (N, M)."""
        traj, mask, pose = (x.to(self.device) for x in (traj, mask, pose))
        _, N, T, _ = traj.shape
        assert T == self.obs_len
        # Row i of the batch has human i as the target human.
        perm = torch.tensor(
            [[(i + j) % N for j in range(N)] for i in range(N)],
            dtype=torch.long,
            device=self.device,
        )
        traj = traj[:, perm].reshape(N, N, T, 2)
        mask = mask[:, perm].reshape(N, N, T, 1)
        pose = pose[:, perm].reshape(N, N, T, 24, 3)

        tokens = torch.zeros(N, T, N * TOKENS, 4, device=self.device)
        view = tokens.view(N, T, N, TOKENS, 4)
        view[:, :, :, 0, :2] = traj.permute(0, 2, 1, 3)
        view[:, :, :, 1, :3] = pose[:, :, :, 0].permute(0, 2, 1, 3)
        view[:, :, :, 2:, :3] = pose[:, :, :, 1:].permute(0, 2, 1, 3, 4)

        padding = self._padding_mask(mask, traj)
        future = self.model(tokens, padding, nmode=self.nmode, limit_obs=0)[:, T:]
        M, F = self.nmode, future.shape[1]

        scores = None
        if self.with_scores:
            # Score each hypothesis of each target human on its own with the features
            # of the Prediction Mode.
            feature, _, _ = self.model.encode(
                tokens.repeat_interleave(M, dim=0),
                padding.repeat_interleave(M, dim=0),
                limit_obs=0,
                random_masking=False,
                with_adapter=True,
            )
            paths = future.permute(0, 2, 1, 3).reshape(N * M, F, 1, 2)
            scores = self.model.estimate_reward(feature, paths).reshape(N, M)

        last = traj[:, 0, -1, :]
        absolute = future + last[:, None, None, :]
        return absolute.permute(0, 2, 1, 3)[:, :, LOOKAHEAD], scores

    @staticmethod
    def _padding_mask(mask: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        """1 for the humans that are unobserved or farther than the context radius."""
        padding = (~mask[:, :, :, 0].any(dim=2)).float()
        padding[:, 0] = 0.0
        distance = torch.norm(traj[:, :, -1, :] - traj[:, 0:1, -1, :], dim=-1)
        padding = torch.where(
            distance > CONTEXT_RADIUS, torch.ones_like(padding), padding
        )
        padding[:, 0] = 0.0
        return padding
