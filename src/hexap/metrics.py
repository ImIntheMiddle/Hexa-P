"""Displacement errors, motion-primitive distributions and LocoVal filtering statistics."""

from typing import Dict, List

import numpy as np
import torch

PRIMITIVES = ("velocity", "acceleration", "ang_velocity", "ang_acceleration")


def min_ade_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Best-of-M ADE in centimetres. pred: (B, T, M, 2); gt: (B, T, N * 25, 4) with the target path first."""
    gt_xy = gt[:, :, 0, :2].unsqueeze(2).repeat(1, 1, pred.size(2), 1)
    ade = torch.norm(pred[..., :2] - gt_xy, p=2, dim=-1).mean(dim=1)  # (B, M)
    return ade.min(dim=1)[0].mean() * 100


def displacement_errors(gt: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    """gt: (T, 2); pred: (T, M, 2). Returns the per-mode ADE and FDE, each (M,)."""
    dist = np.linalg.norm(
        gt[np.newaxis] - np.transpose(pred, (1, 0, 2)), axis=2
    )  # (M, T)
    return {"ade": dist.mean(axis=1), "fde": dist[:, -1]}


def motion_primitives(path: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Speed, absolute acceleration, absolute heading rate and its absolute change of a (T, 2) path."""
    step = path[1:] - path[:-1]
    vel = np.linalg.norm(step / dt, axis=1)
    ang_vel = np.abs(np.arctan2(step[:, 1], step[:, 0]) / dt)
    return {
        "velocity": vel,
        "acceleration": np.abs((vel[1:] - vel[:-1]) / dt),
        "ang_velocity": ang_vel,
        "ang_acceleration": np.abs((ang_vel[1:] - ang_vel[:-1]) / dt),
    }


def chi_square_distance(gt: np.ndarray, pred: np.ndarray, num_bins: int = 20) -> float:
    """Chi-square distance between the histograms of two samples over their joint range."""
    bins = np.linspace(
        min(gt.min(), pred.min()), max(gt.max(), pred.max()), num_bins + 1
    )
    g = np.histogram(gt, bins=bins, density=True)[0] * np.diff(bins)
    p = np.histogram(pred, bins=bins, density=True)[0] * np.diff(bins)
    nonzero = (g != 0) | (p != 0)
    return float(((g[nonzero] - p[nonzero]) ** 2 / (g[nonzero] + p[nonzero])).sum())


def filtering_statistics(
    scores: np.ndarray, ade: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Kept and rejected hypotheses under LocoVal filtering.

    scores, ade: (S, M). A hypothesis is kept when its score is at least the
    threshold; when every hypothesis of a sample is below it, the highest-scored
    one is kept.
    """
    kept = scores >= threshold
    none = ~kept.any(axis=1)
    kept[np.where(none)[0], scores[none].argmax(axis=1)] = True
    rejected = ~kept
    out = {
        "kept_ade": float(ade[kept].mean()),
        "kept_min_ade": float(
            np.mean([ade[i, kept[i]].min() for i in range(len(ade))])
        ),
        "kept_ratio": float(kept.mean()),
    }
    if rejected.any():
        out["rejected_ade"] = float(ade[rejected].mean())
        out["rejected_min_ade"] = float(
            np.mean(
                [
                    ade[i, rejected[i]].min()
                    for i in range(len(ade))
                    if rejected[i].any()
                ]
            )
        )
    return out


class Accumulator:
    """Collects per-sample errors, scores and motion primitives over a split."""

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.ade: List[np.ndarray] = []
        self.fde: List[np.ndarray] = []
        self.scores: List[np.ndarray] = []
        self.gt_prim: Dict[str, List[np.ndarray]] = {k: [] for k in PRIMITIVES}
        self.pred_prim: Dict[str, List[np.ndarray]] = {k: [] for k in PRIMITIVES}

    def add(self, gt: np.ndarray, pred: np.ndarray) -> None:
        """gt: (T, 2); pred: (T, M, 2), both relative to the last observed position."""
        err = displacement_errors(gt, pred)
        self.ade.append(err["ade"])
        self.fde.append(err["fde"])
        for k, v in motion_primitives(gt, self.dt).items():
            self.gt_prim[k].append(v.astype(np.float32))
        for m in range(pred.shape[1]):
            for k, v in motion_primitives(pred[:, m], self.dt).items():
                self.pred_prim[k].append(v.astype(np.float32))

    def summary(self) -> Dict[str, float]:
        ade, fde = np.stack(self.ade), np.stack(self.fde)  # (S, M)
        out = {
            "samples": len(ade),
            "ade": float(ade.mean(axis=1).mean()),
            "fde": float(fde.mean(axis=1).mean()),
            "min_ade": float(ade.min(axis=1).mean()),
            "min_fde": float(fde.min(axis=1).mean()),
        }
        for k in PRIMITIVES:
            out[f"chi2_{k}"] = chi_square_distance(
                np.concatenate(self.gt_prim[k]), np.concatenate(self.pred_prim[k])
            )
        return out
