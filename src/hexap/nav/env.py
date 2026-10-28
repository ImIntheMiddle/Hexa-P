"""CrowdNav++ environment in which the humans replay recorded JTA scenes.

A scene is a list of agents, the first one being the person the scene was built
around, and each agent is a pair `(joints, mask)` with `joints` of shape (frames, 25, 4)
and `mask` of shape (frames, 25), as in `hexap.data.dataset`. The humans never react
to the robot.
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from hexap.nav.crowdnav import (
    ActionXY,
    Config,
    CrowdSimPredRealGST,
    Human,
    JointState,
)

Scene = List[Tuple[torch.Tensor, torch.Tensor]]


def load_scenes(root: str, split: str) -> List[Scene]:
    """Reads `<root>/<split>/part_*.pt`."""
    directory = Path(root) / split
    files = sorted(directory.glob("part_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    assert files, f"no part_*.pt in {directory}"
    scenes: List[Scene] = []
    for f in files:
        scenes += torch.load(f)
    return scenes


class ReplayEnv(CrowdSimPredRealGST):
    """The robot starts 5 m below the centroid of the crowd and walks to the point 5 m above it.

    Every step moves the robot for `config.env.time_step` seconds and replays the next
    frame of the scene.
    """

    def __init__(self, config: Config, scenes: List[Scene]) -> None:
        super().__init__()
        self.scenes = scenes
        self.current_scene: Scene = []
        self.current_frame = 0
        self.scene_len = 0
        self.configure(config)

    def set_robot(self, robot) -> None:
        self.robot = robot

    def reset(self) -> None:
        """Draws a random scene and places the robot."""
        self.global_time = 0
        self.current_scene = random.choice(self.scenes)
        self.current_frame = 0
        self.scene_len = int(self.current_scene[0][0].shape[0])
        self.human_num = min(len(self.current_scene), self.max_human_num)

        self.humans = []
        for i in range(self.human_num):
            human = Human(self.config, "humans")
            joints = self.current_scene[i][0]
            px0 = float(joints[0, 0, 0])
            py0 = float(joints[0, 0, 1])
            gx = float(joints[-1, 0, 0])
            gy = float(joints[-1, 0, 1])
            vx0 = float(joints[1, 0, 0] - joints[0, 0, 0]) / self.time_step
            vy0 = float(joints[1, 0, 1] - joints[0, 0, 1]) / self.time_step
            human.set(px0, py0, gx, gy, vx0, vy0, theta=0.0)
            human.id = i
            human.time_step = self.time_step
            self.humans.append(human)

        xs = [float(self.current_scene[i][0][0, 0, 0]) for i in range(self.human_num)]
        ys = [float(self.current_scene[i][0][0, 0, 1]) for i in range(self.human_num)]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        self.robot.set(cx, cy - 5.0, cx, cy + 5.0, 0.0, 0.0, np.pi / 2)

        goal_vector = np.array([self.robot.gx, self.robot.gy]) - np.array(
            [self.robot.px, self.robot.py]
        )
        self.potential = -abs(np.linalg.norm(goal_vector))
        self.last_human_states = np.zeros((self.human_num, 5))
        self.generate_ob(reset=True)

    def step(self, action: ActionXY) -> Tuple[bool, object]:
        """Moves the robot, replays the next frame and returns (episode ended, outcome)."""
        vx, vy = action
        norm = np.linalg.norm([vx, vy])
        if norm > self.robot.v_pref:
            vx = vx / norm * self.robot.v_pref
            vy = vy / norm * self.robot.v_pref
        action = ActionXY(vx, vy)

        _, done, outcome = self.calc_reward(action, danger_zone="circle")
        self.robot.step(action)
        self.global_time += self.time_step

        self.current_frame = min(self.current_frame + 1, self.scene_len - 1)
        self._sync_human_positions(self.current_frame)
        if self.current_frame >= self.scene_len - 1:
            done = True
        self.generate_ob(reset=False)
        return done, outcome

    def _sync_human_positions(self, t: int) -> None:
        """Sets the position and velocity of every human from frame `t`."""
        for i, human in enumerate(self.humans):
            joints = self.current_scene[i][0]
            human.px = float(joints[t, 0, 0])
            human.py = float(joints[t, 0, 1])
            if t + 1 < self.scene_len:
                human.vx = float(joints[t + 1, 0, 0] - joints[t, 0, 0]) / self.time_step
                human.vy = float(joints[t + 1, 0, 1] - joints[t, 0, 1]) / self.time_step
            else:
                human.vx = 0.0
                human.vy = 0.0


def build_joint_state(env: ReplayEnv) -> JointState:
    """The robot and the last observed state of every human."""
    return JointState(
        env.robot.get_full_state_list(), env.last_human_states[: env.human_num].tolist()
    )
