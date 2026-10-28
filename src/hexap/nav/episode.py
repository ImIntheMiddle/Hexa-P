"""Episodes of the robot in the replayed scenes and their metrics."""

from typing import Dict, List, Optional

import numpy as np

from hexap.nav.crowdnav import SOCIAL_FORCE, Collision, ReachGoal, Timeout
from hexap.nav.env import ReplayEnv, build_joint_state
from hexap.nav.predictor import PathPredictor

# A step is a near miss when the gap between the bodies of robot and human is below this (m).
NEAR_MISS_DIST = 0.5


class SafetyMetrics:
    """Counts the steps in which the robot is within the discomfort distance of a human and the near misses."""

    def __init__(self, discomfort_dist: float) -> None:
        self.discomfort_dist = discomfort_dist
        self.intrusion_steps = 0
        self.total_steps = 0
        self.near_miss_gaps: List[float] = []

    def update(self, env: ReplayEnv) -> None:
        robot = env.robot
        gap = np.inf
        for human in env.humans[: env.human_num]:
            distance = float(np.hypot(robot.px - human.px, robot.py - human.py))
            gap = min(gap, distance - robot.radius - human.radius)
        self.total_steps += 1
        if gap < self.discomfort_dist:
            self.intrusion_steps += 1
        if gap < NEAR_MISS_DIST:
            self.near_miss_gaps.append(gap)

    def summary(self) -> Dict[str, float]:
        return {
            "intrusion_ratio": self.intrusion_steps / max(1, self.total_steps),
            "near_miss_count": len(self.near_miss_gaps),
            "near_miss_min_dist": float(np.mean(self.near_miss_gaps))
            if self.near_miss_gaps
            else float("nan"),
            "total_steps": self.total_steps,
        }


def run_condition(
    env: ReplayEnv,
    planner: SOCIAL_FORCE,
    predictor: Optional[PathPredictor],
    n_episodes: int,
    time_step: float,
    discomfort_dist: float,
) -> Dict[str, float]:
    """Runs `n_episodes` episodes with `planner` (fed by `predictor` if given) and returns the rates and safety metrics."""
    outcomes = {"success": 0, "collision": 0, "timeout": 0, "scene_end": 0}
    path_lengths: List[float] = []
    times_to_goal: List[float] = []
    safety = SafetyMetrics(discomfort_dist)

    for _ in range(n_episodes):
        env.reset()
        if predictor is not None:
            predictor.reset()
            planner.predicted_positions = None
            planner.scores = None
        done = False
        steps = 0
        path_length = 0.0
        previous = np.array([env.robot.px, env.robot.py])
        prediction = None

        while not done:
            if prediction is not None:
                planner.predicted_positions = prediction.positions
                planner.scores = prediction.scores
            action = planner.predict(build_joint_state(env))
            done, outcome = env.step(action)
            steps += 1

            current = np.array([env.robot.px, env.robot.py])
            path_length += float(np.linalg.norm(current - previous))
            previous = current
            safety.update(env)

            if predictor is not None:
                predictor.update(env)
                prediction = predictor.predict()

        path_lengths.append(path_length)
        if isinstance(outcome, ReachGoal):
            outcomes["success"] += 1
            times_to_goal.append(steps * time_step)
        elif isinstance(outcome, Collision):
            outcomes["collision"] += 1
        elif isinstance(outcome, Timeout):
            outcomes["timeout"] += 1
        else:
            outcomes["scene_end"] += 1

    return {
        "success_rate": outcomes["success"] / n_episodes,
        "collision_rate": outcomes["collision"] / n_episodes,
        "timeout_rate": outcomes["timeout"] / n_episodes,
        "scene_end_rate": outcomes["scene_end"] / n_episodes,
        "mean_path_length": float(np.mean(path_lengths)),
        "mean_time_to_goal": float(np.mean(times_to_goal))
        if times_to_goal
        else float("nan"),
        **safety.summary(),
    }
