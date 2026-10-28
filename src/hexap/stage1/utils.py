"""Losses, normalizers and schedulers of the physics pre-training."""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "third_party" / "pcgrad")
)
from pcgrad import PCGrad  # noqa: E402


class PCGradOptimizer(torch.optim.Optimizer):
    """PCGrad (Yu et al., NeurIPS 2020) exposed as a torch Optimizer so that Fabric and schedulers accept it."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(optimizer.param_groups, defaults=optimizer.defaults)
        self._pcgrad = PCGrad(optimizer)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._pcgrad.zero_grad()

    def step(self, closure=None) -> None:
        self._pcgrad.step()

    def pc_backward(self, objectives: List[torch.Tensor]) -> None:
        self._pcgrad.pc_backward(objectives)


class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up followed by cosine annealing, stepped once per epoch."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        span = self.max_epochs - self.warmup_epochs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * span) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / span)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        t = self.last_epoch - self.warmup_epochs
        return [
            (1 + math.cos(math.pi * t / span))
            / (1 + math.cos(math.pi * (t - 1) / span))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class QuantileNormalizer(nn.Module):
    """Maps returns to [0, 1] with running estimates of their 5% and 95% quantiles."""

    def __init__(
        self,
        low_quantile: float = 0.05,
        high_quantile: float = 0.95,
        momentum: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("low_quantile", torch.tensor(low_quantile))
        self.register_buffer("high_quantile", torch.tensor(high_quantile))
        self.register_buffer("momentum", torch.tensor(momentum))
        self.register_buffer("eps", torch.tensor(1e-10))
        self.register_buffer("running_low", torch.tensor(float("nan")))
        self.register_buffer("running_high", torch.tensor(float("nan")))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            low, high = (
                torch.quantile(x, self.low_quantile),
                torch.quantile(x, self.high_quantile),
            )
            if self.num_batches_tracked == 0:
                self.running_low.copy_(low)
                self.running_high.copy_(high)
            else:
                self.running_low.mul_(1 - self.momentum).add_(low, alpha=self.momentum)
                self.running_high.mul_(1 - self.momentum).add_(
                    high, alpha=self.momentum
                )
            self.num_batches_tracked.add_(1)
        clipped = torch.clamp(x, self.running_low, self.running_high)
        return (clipped - self.running_low) / (
            self.running_high - self.running_low + self.eps
        )


def variety_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best-of-K ADE per sample. pred: (B, T, K, 2+); target: (B, T, 1, 2+). Returns (B,) and the best mode (B,)."""
    gt = target[..., :2].repeat(1, 1, pred.size(2), 1)
    ade = torch.norm(pred[..., :2] - gt, p=2, dim=-1).mean(dim=1)  # (B, K)
    return torch.min(ade, dim=1)


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting of the path and LocoVal losses (Kendall et al., CVPR 2018)."""

    def __init__(self) -> None:
        super().__init__()
        self.log_var_traj = nn.Parameter(torch.zeros(1))
        self.log_var_locoval = nn.Parameter(torch.zeros(1))

    def compute_loss(
        self, traj_loss: torch.Tensor, locoval_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.exp(-self.log_var_traj) * traj_loss + self.log_var_traj,
            torch.exp(-self.log_var_locoval) * locoval_loss + self.log_var_locoval,
        )
