"""Robot navigation through replayed JTA crowds (Table 8).

    python -m hexap.nav.run [--hexap weights/stage2/jta/hexap.pt] [--social-trans outputs/stage2/jta_social_trans/best.pt]

Each episode replays a random scene of the split, with the robot starting 5 m below
the crowd and walking 5 m above it. A Social Force planner steers the robot. The
conditions differ in what the planner avoids besides the current positions of the humans:

- `none`: nothing.
- `social_trans`: the paths predicted by Social-Trans.
- `hexap`: the paths predicted by Hexa-P.
- `hexap_locoval`: the paths predicted by Hexa-P that LocoVal scores at least
  `--locoval-threshold`.

Every condition draws the same scenes. The metrics are written to `--out` as JSON.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from hexap.nav.crowdnav import SOCIAL_FORCE, Config
from hexap.nav.env import ReplayEnv, load_scenes
from hexap.nav.episode import run_condition
from hexap.nav.planner import PredictiveSocialForce, make_config
from hexap.nav.predictor import PathPredictor

CONDITIONS = ("none", "social_trans", "hexap", "hexap_locoval")


def build(
    condition: str,
    args: argparse.Namespace,
    config: Config,
    env: ReplayEnv,
    device: str,
) -> Tuple[SOCIAL_FORCE, Optional[PathPredictor]]:
    if condition == "none":
        return SOCIAL_FORCE(config), None
    checkpoint = args.social_trans if condition == "social_trans" else args.hexap
    threshold = args.locoval_threshold if condition == "hexap_locoval" else None
    predictor = PathPredictor(
        checkpoint, device, env.max_human_num, with_scores=threshold is not None
    )
    return PredictiveSocialForce(config, threshold), predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hexap", default="weights/stage2/jta/hexap.pt")
    parser.add_argument("--social-trans", default="outputs/stage2/jta_social_trans/best.pt")
    parser.add_argument("--replay-root", default="data/jta_replay")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS
    )
    parser.add_argument("--locoval-threshold", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default=None, help="write the metrics to this JSON file"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = make_config()
    scenes = load_scenes(args.replay_root, args.split)
    env = ReplayEnv(config, scenes)
    print(
        f"{args.split}: {len(scenes)} scenes, {args.episodes} episodes, seed {args.seed}"
    )

    results: Dict[str, Dict[str, float]] = {}
    for condition in args.conditions:
        random.seed(args.seed)
        planner, predictor = build(condition, args, config, env, device)
        results[condition] = run_condition(
            env,
            planner,
            predictor,
            args.episodes,
            config.env.time_step,
            config.reward.discomfort_dist,
        )
        r = results[condition]
        print(
            f"{condition:<14} collision {100 * r['collision_rate']:5.1f}%  "
            f"success {100 * r['success_rate']:5.1f}%  "
            f"intrusion {100 * r['intrusion_ratio']:5.2f}%  "
            f"near-miss {r['near_miss_count']:4d} (gap {r['near_miss_min_dist']:.3f} m)  "
            f"path {r['mean_path_length']:.2f} m  time {r['mean_time_to_goal']:.2f} s",
            flush=True,
        )
    if args.out:
        meta = {
            "split": args.split,
            "scenes": len(scenes),
            "episodes": args.episodes,
            "seed": args.seed,
            "locoval_threshold": args.locoval_threshold,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
