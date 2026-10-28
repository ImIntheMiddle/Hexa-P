"""Stage 1: run ProtoMotions training with the Hexa-P configs on the Hydra search path.

    python -m hexap.stage1.train +exp=hexap/pretrain +robot=smpl +simulator=isaacgym \
        motion_file=data/motions/stage1_motions.pt checkpoint=weights/stage1/controller.ckpt \
        experiment_name=hexap_pretrain

Run it from the repository root: ProtoMotions loads its assets from the relative path
`protomotions/data/assets`, which the `protomotions` link at the root (to
`third_party/protomotions/protomotions`) provides.

`+exp=hexap/controller` trains the path-following controller, `+exp=hexap/pretrain` the
Hexa-P network on its rollouts, and `+exp=hexap/emloco` the EmLoco plausibility estimator.
"""

import importlib.util
import runpy
import sys
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"


def main() -> None:
    # Locate ProtoMotions without importing it: Isaac Gym must be imported before torch.
    train_agent = (
        Path(importlib.util.find_spec("protomotions").origin).parent / "train_agent.py"
    )
    sys.argv = (
        [str(train_agent)] + sys.argv[1:] + [f"hydra.searchpath=[file://{CONFIGS}]"]
    )
    sys.path.insert(0, str(train_agent.parent))
    runpy.run_path(str(train_agent), run_name="__main__")


if __name__ == "__main__":
    main()
