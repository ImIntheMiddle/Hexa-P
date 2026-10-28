"""Render videos that compare methods on test windows.

    python -m hexap.viz.render --data-root data --split test --limit-obs 2 \
        --method "Hexa-P=weights/stage2/jrdb/hexap.pt" --method "Social-Trans=outputs/stage2/jrdb_social_trans/best.pt" \
        --select improvement --num 6 --out media/jrdb_obs2

All methods must be trained on the same dataset. For each selected window the script
writes `<id>.mp4` (H.264), `<id>.webp` (animated, small), and `<id>.jpg` (last frame), and
`index.json` with the per-method ADE and minADE of each window. `--select improvement`
picks the windows where the first method improves most on the minADE of the second;
`--select random` picks windows uniformly at random (seeded).
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from hexap.data.dataset import TrajPoseDataset, batch_process_coords, collate_batch
from hexap.models import load_checkpoint
from hexap.viz.scene3d import Sample, render_image, render_video

COLORS = ["green", "blue", "darkorange", "gold"]


@torch.no_grad()
def predict(
    checkpoint: str, root: str, split: str, limit_obs: int, device: str
) -> Tuple[np.ndarray, list, list]:
    """Returns predictions (S, T_out, M, 2), and per window the observed and future tokens (T, N, 25, 3) of all agents."""
    model, config = load_checkpoint(checkpoint, device)
    dataset = TrajPoseDataset(root, config.data.dataset, split)
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        collate_fn=collate_batch,
        shuffle=False,
    )
    out_F, nmode = config.data.output_track_size, config.model.nmode
    preds, observed, future = [], [], []
    for tokens, masks, padding in loader:
        in_tokens, _, _, _, padding = batch_process_coords(
            tokens, masks, padding, config, device=device
        )
        pred = model(
            torch.nan_to_num(in_tokens), padding, nmode=nmode, limit_obs=limit_obs
        )[:, -out_F:]
        preds.append(pred.cpu().numpy())
        in_F = in_tokens.shape[1]
        for b in range(tokens.shape[0]):
            n = int((~padding[b].bool()).sum())
            observed.append(tokens[b, :n, :in_F, :, :3].permute(1, 0, 2, 3).numpy())
            future.append(tokens[b, :n, in_F : in_F + out_F, :, :3].permute(1, 0, 2, 3).numpy())
    return np.concatenate(preds), observed, future


def errors(
    pred: np.ndarray, observed: List[np.ndarray], future: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    gt = np.stack(
        [f[:, 0, 0, :2] - o[-1, 0, 0, :2] for f, o in zip(future, observed)]
    )  # (S, T, 2)
    ade = np.linalg.norm(pred - gt[:, :, None], axis=-1).mean(axis=1)  # (S, M)
    return ade.mean(axis=1), ade.min(axis=1)


def to_webp(mp4: Path, width: int = 640) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            f"fps=15,scale={width}:-1",
            "-loop",
            "0",
            "-q:v",
            "60",
            str(mp4.with_suffix(".webp")),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="NAME=CHECKPOINT; the first is the reference",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit-obs", type=int, default=0)
    parser.add_argument(
        "--select", choices=["improvement", "random"], default="improvement"
    )
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--min-motion", type=float, default=1.0, help="minimum future displacement of the target (m)")
    parser.add_argument(
        "--windows", default="", help="comma-separated window ids; overrides --select"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods: Dict[str, np.ndarray] = {}
    observed = future = None
    for spec in args.method:
        name, ckpt = spec.split("=", 1)
        methods[name], observed, future = predict(
            ckpt, args.data_root, args.split, args.limit_obs, device
        )
    names = list(methods)
    stats = {n: errors(methods[n], observed, future) for n in names}
    agents = np.array([o.shape[1] for o in observed])
    # Windows whose target has a complete pose in the last observed frame.
    posed = np.array([not np.isnan(o[-1, 0, 1:]).any() and np.abs(o[-1, 0, 2:]).sum() > 0 for o in observed])
    motion = np.array([np.linalg.norm(f[-1, 0, 0, :2] - o[-1, 0, 0, :2]) for f, o in zip(future, observed)])
    valid = np.where((agents >= args.min_agents) & posed & (motion >= args.min_motion))[0]
    if args.windows:
        chosen = np.array([int(w) for w in args.windows.split(",")])
    elif args.select == "improvement" and len(names) > 1:
        gain = stats[names[1]][1] - stats[names[0]][1]
        chosen = valid[np.argsort(-gain[valid])[: args.num]]
    else:
        chosen = np.random.default_rng(args.seed).choice(
            valid, size=args.num, replace=False
        )

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    fps = float(load_checkpoint(args.method[0].split("=", 1)[1], "cpu")[1].data.fps)
    for i in chosen:
        sample = Sample(
            observed=observed[i],
            future=future[i],
            predictions={n: methods[n][i] for n in names},
            colors={n: COLORS[k] for k, n in enumerate(names)},
            limit_obs=args.limit_obs,
        )
        stem = args.out / f"{args.split}_{i:05d}"
        render_video(sample, str(stem.with_suffix(".mp4")), data_fps=fps)
        to_webp(stem.with_suffix(".mp4"))
        render_image(sample, str(stem.with_suffix(".jpg")))
        index.append(
            {
                "id": int(i),
                "agents": int(agents[i]),
                **{f"{n}/ade": float(stats[n][0][i]) for n in names},
                **{f"{n}/min_ade": float(stats[n][1][i]) for n in names},
            }
        )
        print(stem, index[-1])
    with open(args.out / "index.json", "w") as f:
        json.dump(
            {"methods": names, "limit_obs": args.limit_obs, "windows": index},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
