"""3D plots and videos of observed motion, ground truth and predicted paths.

The style follows the qualitative figures of the paper: the target agent's skeleton
in magenta (centre and right) and lime (left) drawn above its path, past paths in
black, the ground truth in red, other agents in grey, and one colour per method for
the predicted paths. Paths lie on a plane below the skeletons.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

# Bones of the 24-joint SMPL layout (pelvis = 0).
SKELETON = (
    [[0, 9], [9, 10], [10, 11], [11, 12], [12, 13]],
    [[0, 1], [1, 2], [2, 3], [3, 4], [11, 14], [14, 15], [15, 16], [16, 17], [17, 18]],
    [[0, 5], [5, 6], [6, 7], [7, 8], [11, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
)
PARENT = {b: a for part in SKELETON for a, b in part}
CHEST, NECK = 11, 12
POSE_COLORS = ("magenta", "lime", "magenta")
Z_OFFSET = -1.5
POSE_SCALE = 2.0
VIEW = dict(elev=40, azim=-50)


@dataclass
class Sample:
    """One window of a dataset with the predictions of several methods.

    observed: (T_in, N, 25, 3) tokens of all agents (token 0 = path, 1 = pelvis, 2-24 = joints);
    future: (T_out, N, 25, 3) tokens of the same agents in the same world frame; predictions: method -> (T_out, M, 2)
    paths relative to the last observed position of the target.
    """

    observed: np.ndarray
    future: np.ndarray
    predictions: Dict[str, np.ndarray]
    colors: Dict[str, str] = field(default_factory=dict)
    limit_obs: int = 0
    shown: Optional[List[str]] = (
        None  # methods to draw (all by default); the view and the legend always cover all of them
    )
    zoom: float = 1.25  # magnification of the 3D box in the frame


def bones(present: np.ndarray) -> List[Tuple[int, int, int]]:
    """(part, a, b) bones between the joints a dataset provides (`present`, (24,) bool).

    A missing joint passes its children to its nearest provided ancestor; a missing chest
    passes the arms to the neck (JTA has no toes, chest, thoraxes or hands).
    """
    out = []
    for k, part in enumerate(SKELETON):
        for a, b in part:
            if not present[b]:
                continue
            while not present[a]:
                a = NECK if a == CHEST and b != NECK and present[NECK] else PARENT[a]
            out.append((k, a, b))
    return out


def skeleton_points(tokens: np.ndarray) -> np.ndarray:
    """(25, 3) tokens -> (24, 3) joints drawn above the path point."""
    path = np.array([tokens[0, 0], tokens[0, 1], Z_OFFSET])
    joints = (tokens[2:] - tokens[1:2]) * POSE_SCALE + path
    return np.concatenate([path[None], joints], axis=0)


def draw_skeleton(
    ax, tokens: np.ndarray, colors: Sequence[str], alpha: float = 1.0, size: float = 5
) -> None:
    if np.isnan(tokens).any() or np.abs(tokens[1:]).max() < 1e-6:
        return
    points = skeleton_points(tokens)
    # Joints that the dataset does not provide sit on the pelvis.
    present = np.abs(tokens[1:] - tokens[1]).sum(axis=-1) > 0
    present[0] = True
    drawn = bones(present)
    for k, a, b in drawn:
        ax.plot(*points[[a, b]].T, c=colors[k], linewidth=1.5, alpha=alpha)
    for k, color in enumerate(colors):
        ids = sorted({b for part, _, b in drawn if part == k})
        ax.scatter(*points[ids].T, c=color, s=size, alpha=alpha)


def ground(path: np.ndarray) -> np.ndarray:
    return np.concatenate([path, np.full((len(path), 1), Z_OFFSET)], axis=1)


def axis_limits(
    sample: Sample, margin: float = 0.75, min_size: float = 4.0
) -> Dict[str, tuple]:
    """A square box around the target's paths and the predictions, at least `min_size` metres wide."""
    obs = sample.observed
    last = obs[-1, 0, 0, :2]
    xy = [obs[:, 0, 0, :2], sample.future[:, 0, 0, :2]] + [
        (p + last).reshape(-1, 2) for p in sample.predictions.values()
    ]
    xy = np.concatenate(xy)
    xy = xy[~np.isnan(xy).any(axis=1) & (np.abs(xy).sum(axis=1) > 1e-6)]
    centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
    half = max((xy.max(axis=0) - xy.min(axis=0)).max() / 2 + margin, min_size / 2)
    z = (obs[:, 0, 2:, 2] - obs[:, 0, 1:2, 2]) * POSE_SCALE + Z_OFFSET
    top = np.nanmax(z) if np.isfinite(z).any() else Z_OFFSET + 2 * POSE_SCALE
    return {
        "x": (centre[0] - half, centre[0] + half),
        "y": (centre[1] - half, centre[1] + half),
        "z": (Z_OFFSET - 0.5, top + 0.5),
    }


def inside(xy: np.ndarray, lim: Dict[str, tuple]) -> bool:
    return lim["x"][0] <= xy[0] <= lim["x"][1] and lim["y"][0] <= xy[1] <= lim["y"][1]


def setup_axes(fig, sample: Sample):
    ax = fig.add_subplot(111, projection="3d")
    lim = axis_limits(sample)
    ax.set_xlim(*lim["x"])
    ax.set_ylim(*lim["y"])
    ax.set_zlim(*lim["z"])
    ax.set_box_aspect(
        [
            lim["x"][1] - lim["x"][0],
            lim["y"][1] - lim["y"][0],
            lim["z"][1] - lim["z"][0],
        ],
        zoom=sample.zoom,
    )
    ax.view_init(**VIEW)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return ax


def agent_tokens(seq: np.ndarray, t: float) -> Optional[np.ndarray]:
    """(25, 3) tokens of one agent at time t in frames of seq (T, 25, 3), or None while it is absent.

    The path point is interpolated; the pose is interpolated between complete frames, else the
    latest complete pose is moved onto the path point (JRDB poses are missing in some frames).
    """
    i0 = int(np.floor(t))
    w, i1 = t - i0, min(i0 + 1, len(seq) - 1)
    a, b = seq[i0, 0, :2], seq[i1, 0, :2]
    if np.isnan(a).any() or np.abs(a).sum() < 1e-6:
        return None
    complete = [k for k in range(i0 + 1) if not np.isnan(seq[k]).any()]
    if not complete:
        return None
    if complete[-1] == i0 and not np.isnan(seq[i1]).any():
        return (1 - w) * seq[i0] + w * seq[i1]
    tokens = seq[complete[-1]].copy()
    tokens[0, :2] = a if np.isnan(b).any() else (1 - w) * a + w * b
    return tokens


def draw_state(ax, sample: Sample, t: float) -> None:
    """Draws the scene at time t in frames: t < T_in replays the observation, then the future unfolds."""
    obs, fut = sample.observed, sample.future
    T_in, N = obs.shape[0], obs.shape[1]
    first = T_in - sample.limit_obs if sample.limit_obs else 0
    seq = np.concatenate([obs, fut], axis=0)  # (T_in + T_out, N, 25, 3)
    lim = axis_limits(sample)
    for n in range(1, N):  # other agents inside the box: path so far and pose
        tokens = agent_tokens(seq[:, n], t)
        if tokens is None or not inside(tokens[0, :2], lim):
            continue
        path = seq[first : int(np.floor(t)) + 1, n, 0, :2]
        path = path[~np.isnan(path).any(axis=1) & (np.abs(path).sum(axis=1) > 1e-6)]
        path = np.concatenate([path, tokens[None, 0, :2]])
        ax.plot(*ground(path).T, c="gray", alpha=0.7, linewidth=0.8, linestyle=":")
        draw_skeleton(ax, tokens, ("gray",) * 3, alpha=0.25)
    i0 = int(np.floor(min(t, T_in - 1)))
    past = obs[first : i0 + 1, 0, 0, :2]
    ax.plot(*ground(past).T, c="black", linewidth=1.5, marker="o", markersize=3)
    last = obs[-1, 0, 0, :2]
    target = agent_tokens(seq[:, 0], t)
    if t <= T_in - 1:
        if target is not None:
            draw_skeleton(ax, target, POSE_COLORS)
        return
    ax.scatter(*ground(last[None]).T, c="black", marker="x", s=80)
    s = t - (T_in - 1)  # future frames shown
    j0, v = int(np.floor(s)), s - np.floor(s)
    gt = np.concatenate([last[None], fut[:, 0, 0, :2]], axis=0)
    shown = np.concatenate(
        [gt[: j0 + 1], [(1 - v) * gt[j0] + v * gt[min(j0 + 1, len(gt) - 1)]]]
    )
    ax.plot(*ground(shown).T, c="red", linewidth=1.5, marker="o", markersize=3)
    if j0 >= len(gt) - 1:
        ax.scatter(*ground(gt[-1:]).T, c="red", marker="*", s=100)
    if target is not None:
        draw_skeleton(ax, target, POSE_COLORS)
    for name, paths in sample.predictions.items():
        if sample.shown is not None and name not in sample.shown:
            continue
        full = np.concatenate([np.zeros((1, paths.shape[1], 2)), paths], axis=0) + last
        color = sample.colors.get(name, "green")
        for m in range(paths.shape[1]):
            p = full[:, m]
            seg = np.concatenate(
                [p[: j0 + 1], [(1 - v) * p[j0] + v * p[min(j0 + 1, len(p) - 1)]]]
            )
            ax.plot(
                *ground(seg).T,
                c=color,
                linestyle="--",
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                alpha=0.8,
            )
            if j0 >= len(p) - 1:
                ax.scatter(*ground(p[-1:]).T, c=color, marker="^", s=50)


def legend(fig, sample: Sample) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], c="black", marker="o", markersize=3, label="Observed path"),
        Line2D([], [], c="red", marker="o", markersize=3, label="Ground truth"),
    ] + [
        Line2D(
            [],
            [],
            c=sample.colors.get(n, "green"),
            linestyle="--",
            marker="o",
            markersize=3,
            label=n,
        )
        for n in sample.predictions
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=9,
        frameon=False,
    )


def render_video(
    sample: Sample,
    out: str,
    data_fps: float = 2.5,
    substeps: int = 4,
    hold_seconds: float = 1.0,
    size=(6.4, 4.0),
) -> None:
    """Writes an mp4 of the observation followed by the predicted and ground-truth futures, in real time."""
    T_in, T_out = sample.observed.shape[0], sample.future.shape[0]
    first = T_in - sample.limit_obs if sample.limit_obs else 0
    fps = data_fps * substeps
    times = list(np.arange(first, T_in - 1 + T_out + 1e-6, 1.0 / substeps))
    times += [T_in - 1 + T_out] * int(hold_seconds * fps)
    fig = plt.figure(figsize=size, dpi=200)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "23"],
    )
    with writer.saving(fig, out, dpi=200):
        for t in times:
            fig.clf()
            ax = setup_axes(fig, sample)
            draw_state(ax, sample, float(t))
            legend(fig, sample)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
            writer.grab_frame()
    plt.close(fig)


def render_image(sample: Sample, out: str, size=(6.4, 4.0)) -> None:
    """Writes the final frame (observation, ground truth and all predictions) as an image."""
    T_in, T_out = sample.observed.shape[0], sample.future.shape[0]
    fig = plt.figure(figsize=size, dpi=200)
    ax = setup_axes(fig, sample)
    draw_state(ax, sample, float(T_in - 1 + T_out))
    legend(fig, sample)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
