"""Videos of the simulated humanoid walking a commanded path (saved by `hexap.executability`).

python -m hexap.viz.rollout executability_rollouts.pt --condition hexap_best --num 4 --out media/executability
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from hexap.viz.scene3d import POSE_COLORS, SKELETON, VIEW  # noqa: E402


def render_rollout(
    bodies: np.ndarray,
    path: np.ndarray,
    start: np.ndarray,
    out: str,
    fps: float = 30.0,
    color: str = "green",
    label: str = "Commanded path",
) -> None:
    """bodies: (T, 24, 3) world positions; path: (F, 2) waypoints after `start` (2,)."""
    path = np.concatenate([start[None], path], axis=0)
    xy = np.concatenate([bodies[:, 0, :2], path])
    centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
    half = max((xy.max(axis=0) - xy.min(axis=0)).max() / 2 + 0.75, 1.5)
    fig = plt.figure(figsize=(6.4, 4.0), dpi=200)
    writer = FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "23"]
    )
    frames = list(range(len(bodies))) + [len(bodies) - 1] * int(fps)
    with writer.saving(fig, out, dpi=200):
        for t in frames:
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_zlim(0, 2)
            ax.set_box_aspect([2 * half, 2 * half, 2])
            ax.view_init(**VIEW)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.plot(
                path[:, 0],
                path[:, 1],
                np.zeros(len(path)),
                c=color,
                linestyle="--",
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                label=label,
            )
            ax.plot(
                bodies[: t + 1, 0, 0],
                bodies[: t + 1, 0, 1],
                np.zeros(t + 1),
                c="black",
                linewidth=1.5,
                label="Pelvis",
            )
            for part, c in zip(SKELETON, POSE_COLORS):
                for a, b in part:
                    ax.plot(*bodies[t, [a, b]].T, c=c, linewidth=2)
            fig.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("records")
    parser.add_argument("--condition", default="hexap_best")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = torch.load(args.records)[args.condition]
    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    for episode, rec in enumerate(records):
        for i in range(rec["bodies"].shape[0]):
            if count >= args.num:
                return
            stem = args.out / f"{args.condition}_{episode}_{i}"
            render_rollout(
                rec["bodies"][i].numpy(),
                rec["path"][i].numpy(),
                rec["start"][i].numpy(),
                str(stem.with_suffix(".mp4")),
                fps=args.fps,
            )
            print(stem, "fell" if rec["fall"][i] > 0 else "walked")
            count += 1


if __name__ == "__main__":
    main()
