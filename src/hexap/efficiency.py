"""Parameter count and per-sample inference latency of Hexa-P, built up component by component.

    python -m hexap.efficiency CHECKPOINT [--data-root data] [--batch-size 1]

Rows are cumulative: the base predictor (Physical Core, social module and Path-Head
without TaPE and adapters), + TaPE, + adapters (= Hexa-P prediction), and the extra
cost of scoring the M hypotheses with the LocoVal-Head, either from the features of
the prediction pass or with a separate Evaluation-Mode pass.
"""

import argparse
import copy
from typing import Callable, Dict

import torch
from omegaconf import DictConfig

from hexap.data.dataset import TrajPoseDataset, batch_process_coords, collate_batch
from hexap.models import build_model, load_checkpoint


def components(model: torch.nn.Module) -> Dict[str, int]:
    out = {"adapters": 0, "tape": 0, "locoval_head": 0, "other": 0}
    for name, p in model.named_parameters():
        if "adapter" in name:
            out["adapters"] += p.numel()
        elif name.startswith("relative_encoder"):
            out["tape"] += p.numel()
        elif name.startswith("reward_head"):
            out["locoval_head"] += p.numel()
        else:
            out["other"] += p.numel()
    return out


@torch.no_grad()
def latency(
    fn: Callable[[], torch.Tensor], warmup: int = 15, iters: int = 60
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return {"mean_ms": t.mean().item(), "std_ms": t.std().item()}


def variant(
    model: torch.nn.Module, config: DictConfig, **changes: str
) -> torch.nn.Module:
    cfg = copy.deepcopy(config.model)
    for k, v in changes.items():
        cfg[k] = v
    m = build_model("LocoValTransformer", cfg, "cuda")
    m.load_state_dict(model.state_dict(), strict=False)
    return m.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    model, config = load_checkpoint(args.checkpoint, "cuda")
    dataset = TrajPoseDataset(args.data_root, config.data.dataset, "test")
    tokens, masks, padding = collate_batch([dataset[i] for i in range(args.batch_size)])
    tgt, _, _, _, pad = batch_process_coords(
        tokens, masks, padding, config, device="cuda"
    )
    tgt = torch.nan_to_num(tgt)
    nmode, out_F = config.model.nmode, config.data.output_track_size

    base = variant(model, config, peft_type="none", position_encoding="none")
    tape = variant(model, config, peft_type="none")
    full = model
    pred = full(tgt, pad, nmode=nmode)[:, -out_F:]

    def score_from_prediction_pass() -> torch.Tensor:
        feature, rel, N = full.encode(tgt, pad, 0, False, with_adapter=True)
        paths = full.predict_traj(
            feature, pad, full.mask_relative(rel, tgt, N, 0, False), nmode
        )[:, -out_F:]
        return full.estimate_reward(feature, paths)

    c = components(full)
    rows = {
        "base predictor": (c["other"], latency(lambda: base(tgt, pad, nmode=nmode))),
        "+ TaPE": (c["other"] + c["tape"], latency(lambda: tape(tgt, pad, nmode=nmode))),
        "+ adapters (Hexa-P prediction)": (
            c["other"] + c["tape"] + c["adapters"],
            latency(lambda: full(tgt, pad, nmode=nmode)),
        ),
        "+ LocoVal scoring from the prediction pass": (sum(c.values()), latency(score_from_prediction_pass)),
        "+ LocoVal scoring in Evaluation Mode": (
            sum(c.values()),
            latency(lambda: (full(tgt, pad, nmode=nmode), full.plausibility(tgt, pad, pred))),
        ),
    }
    print(
        f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}, batch size {args.batch_size}"
    )
    print(f"parameters by component: {c}")
    for name, (params, t) in rows.items():
        print(
            f"{name:45s} {params / 1e6:6.3f} M  {t['mean_ms']:6.2f} ± {t['std_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
