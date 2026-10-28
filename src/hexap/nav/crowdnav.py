"""Loads the unmodified CrowdNav++ in `third_party/crowdnav`.

CrowdNav++ was written for NumPy < 1.24 and a command line of its own. The three
adjustments below let it import with NumPy 1.24, gym 0.25 and the command line of
this package:

- `numpy.bool`, which CrowdNav++ uses and NumPy 1.24 removed, is restored.
- `arguments.get_args` parses an empty command line instead of `sys.argv`.
- `crowd_sim.envs` is imported before `crowd_nav.policy`, which breaks the import
  cycle between `ORCA` and `CrowdSim`.

Importing this module is what puts `crowd_sim` and `crowd_nav` on `sys.path`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "third_party" / "crowdnav"
sys.path.insert(0, str(ROOT))

if "bool" not in vars(np):
    np.bool = bool

import arguments  # noqa: E402

_get_args = arguments.get_args


def _get_args_from_defaults() -> argparse.Namespace:
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        return _get_args()
    finally:
        sys.argv = argv


arguments.get_args = _get_args_from_defaults

import crowd_sim.envs  # noqa: E402,F401
from crowd_nav.configs.config import Config  # noqa: E402
from crowd_nav.policy.social_force import SOCIAL_FORCE  # noqa: E402
from crowd_sim.envs.crowd_sim_pred_real_gst import CrowdSimPredRealGST  # noqa: E402
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402
from crowd_sim.envs.utils.human import Human  # noqa: E402
from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout  # noqa: E402
from crowd_sim.envs.utils.state import JointState  # noqa: E402

__all__ = [
    "ActionXY",
    "Collision",
    "Config",
    "CrowdSimPredRealGST",
    "Human",
    "JointState",
    "ReachGoal",
    "SOCIAL_FORCE",
    "Timeout",
]
