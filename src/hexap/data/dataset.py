"""Multi-agent path-and-pose windows of JTA, JRDB and VRU.

A split is stored as `<root>/<dataset>/<split>/part_*.pt`. Each part is a list of
windows; a window is a list of agents, the target agent first, and each agent is
a pair `(tokens, mask)` with `tokens` of shape (frames, 25, 4) and `mask` of
shape (frames, 25). Token 0 holds the path (x, y on the ground plane), token 1
the pelvis and tokens 2-24 the other 23 SMPL joints in world coordinates.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

Window = List[Tuple[torch.Tensor, torch.Tensor]]


class TrajPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, dataset: str, split: str, max_windows: int = 0
    ) -> None:
        directory = Path(root) / dataset / split
        files = sorted(
            directory.glob("part_*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        assert files, f"no part_*.pt in {directory}"
        self.windows: List[Window] = []
        for f in files:
            self.windows += torch.load(f)
            if max_windows and len(self.windows) >= max_windows:
                self.windows = self.windows[:max_windows]
                break

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        tokens = torch.stack([agent[0] for agent in window]).float()  # (N, F, 25, 4)
        mask = torch.stack([agent[1] for agent in window]).float()  # (N, F, 25)
        return tokens, mask


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads the agents; padding_mask is True for padded agents."""
    tokens = pad_sequence([b[0] for b in batch], batch_first=True)
    masks = pad_sequence([b[1] for b in batch], batch_first=True)
    padding_mask = pad_sequence(
        [torch.zeros(b[0].shape[0]) for b in batch], batch_first=True, padding_value=1
    ).bool()
    return tokens, masks, padding_mask


def random_rotation(tokens: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotates (B, N, F, J, 3) points about the vertical axis by per-window angles (B,)."""
    B = tokens.shape[0]
    rotation = torch.zeros(B, 3, 3, device=tokens.device)
    rotation[:, 0, 0] = torch.cos(angles)
    rotation[:, 0, 1] = -torch.sin(angles)
    rotation[:, 1, 0] = torch.sin(angles)
    rotation[:, 1, 1] = torch.cos(angles)
    rotation[:, 2, 2] = 1
    return torch.bmm(tokens.reshape(B, -1, 3).float(), rotation).reshape(tokens.shape)


def batch_process_coords(
    tokens: torch.Tensor,
    masks: torch.Tensor,
    padding_mask: torch.Tensor,
    config: DictConfig,
    training: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits windows into observed and future tokens.

    Returns in_tokens (B, in_F, N * 25, 4), in_masks, out_tokens (B, out_F, N * 25, 4)
    relative to the last observed frame, out_masks, and padding_mask (B, N) as float.
    """
    tokens = tokens.to(device)
    masks = masks.to(device)
    padding_mask = padding_mask.to(device)
    B, N, F, J, K = tokens.shape
    assert J == 25 and K == 4, tokens.shape
    in_F, out_F = config.data.input_track_size, config.data.output_track_size

    if training:
        angles = torch.deg2rad(torch.rand(len(tokens)) * 360)
        tokens[:, :, :, 0:1, :3] = random_rotation(tokens[:, :, :, 0:1, :3], angles)
        tokens[:, :, :, 1:, :3] = random_rotation(tokens[:, :, :, 1:, :3], angles)

    tokens = tokens.transpose(1, 2).reshape(B, F, N * J, K)
    masks = (
        torch.cat((masks[:, :, :, :1], masks[:, :, :, 2:]), dim=3)
        .transpose(1, 2)
        .reshape(B, F, N * (J - 1))
    )
    in_tokens = tokens[:, :in_F].float()
    out_tokens = tokens[:, in_F : in_F + out_F].float() - tokens[
        :, in_F - 1 : in_F
    ].reshape(B, 1, N * J, K)
    return (
        in_tokens,
        masks[:, :in_F].float(),
        out_tokens,
        masks[:, in_F : in_F + out_F].float(),
        padding_mask.float(),
    )
