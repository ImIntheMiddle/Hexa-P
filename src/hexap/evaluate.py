"""Evaluate a Stage-2 checkpoint on a split.

    python -m hexap.evaluate CHECKPOINT [--split test] [--limit-obs 2] [--filter-threshold 0.6]

`--limit-obs K` keeps only the last K observed frames (0 keeps all of them).
`--filter-threshold` applies LocoVal filtering with the Evaluation Mode of the
same network and reports the kept and rejected hypotheses.
"""

import argparse
import json
import random
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hexap.data.dataset import TrajPoseDataset, batch_process_coords, collate_batch
from hexap.metrics import Accumulator, filtering_statistics
from hexap.models import load_checkpoint


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    config: DictConfig,
    loader: DataLoader,
    device: str,
    limit_obs: int,
    filter_threshold: Optional[float] = None,
) -> Dict[str, float]:
    out_F = config.data.output_track_size
    nmode = config.model.nmode
    acc = Accumulator(dt=1.0 / config.data.fps)
    scores = []
    for tokens, masks, padding_mask in loader:
        in_tokens, _, out_tokens, _, padding_mask = batch_process_coords(
            tokens, masks, padding_mask, config, device=device
        )
        in_tokens = torch.nan_to_num(in_tokens)
        pred = model(in_tokens, padding_mask, nmode=nmode, limit_obs=limit_obs)[
            :, -out_F:
        ]
        B = pred.shape[0]
        assert pred.shape == (B, out_F, nmode, 2)
        if filter_threshold is not None:
            scores.append(
                model.plausibility(in_tokens, padding_mask, pred, limit_obs=limit_obs)
                .cpu()
                .numpy()
            )
        gt = out_tokens[:, :, 0, :2].cpu().numpy()
        pred = pred.cpu().numpy()
        for b in range(B):
            acc.add(gt[b], pred[b])
    result = acc.summary()
    if filter_threshold is not None:
        result.update(
            filtering_statistics(
                np.concatenate(scores), np.stack(acc.ade), filter_threshold
            )
        )
    return result


def make_loader(config: DictConfig, root: str, split: str) -> DataLoader:
    dataset = TrajPoseDataset(root, config.data.dataset, split)
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        collate_fn=collate_batch,
        shuffle=False,
        num_workers=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit-obs", type=int, default=0)
    parser.add_argument("--filter-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default=None, help="write the metrics to this JSON file"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_checkpoint(args.checkpoint, device)
    result = evaluate(
        model,
        config,
        make_loader(config, args.data_root, args.split),
        device,
        args.limit_obs,
        args.filter_threshold,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "limit_obs": args.limit_obs,
                    **result,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
