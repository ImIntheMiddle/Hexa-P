"""Social Force planners.

The Social Force planner of CrowdNav++ (`SOCIAL_FORCE`) pulls the robot toward its goal
and pushes it away from the current position of every human. `PredictiveSocialForce`
additionally pushes it away from every predicted position of every human, where the
forces of the M hypotheses of a human add up. With a LocoVal threshold, the
hypotheses scored below it are dropped first and the remaining ones are scaled up to
the weight of the M hypotheses.
"""

from typing import Optional

import numpy as np

from hexap.nav.crowdnav import SOCIAL_FORCE, ActionXY, Config, JointState

# Interaction strength, interaction range and goal gain of the Social Force model.
FORCE_STRENGTH = 0.4
FORCE_RANGE = 3.0
GOAL_GAIN = 4.0


def make_config() -> Config:
    """CrowdNav++ configuration with the Social Force parameters of the experiment."""
    config = Config()
    # Humans and robot keep the radius and preferred speed of the configuration.
    config.env.randomize_attributes = False
    config.sf.A = FORCE_STRENGTH
    config.sf.B = FORCE_RANGE
    config.sf.KI = GOAL_GAIN
    return config


class PredictiveSocialForce(SOCIAL_FORCE):
    """Social Force that also avoids the predicted positions of the humans.

    Before each `predict`, set `predicted_positions` (humans, M, 2) and, for the
    LocoVal gate, `scores` (humans, M).
    """

    def __init__(self, config: Config, locoval_threshold: Optional[float] = None):
        super().__init__(config)
        self.locoval_threshold = locoval_threshold
        self.predicted_positions: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None

    def predict(self, state: JointState) -> ActionXY:
        robot = state.self_state
        delta_x = robot.gx - robot.px
        delta_y = robot.gy - robot.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        desired_vx = (delta_x / dist_to_goal) * robot.v_pref
        desired_vy = (delta_y / dist_to_goal) * robot.v_pref
        KI = self.config.sf.KI
        curr_delta_vx = KI * (desired_vx - robot.vx)
        curr_delta_vy = KI * (desired_vy - robot.vy)

        A = self.config.sf.A
        B = self.config.sf.B
        interaction_vx = 0.0
        interaction_vy = 0.0
        for i, human in enumerate(state.human_states):
            radii = robot.radius + human.radius
            dx = robot.px - human.px
            dy = robot.py - human.py
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-6:
                scale = A * np.exp((radii - dist) / B) / dist
                interaction_vx += scale * dx
                interaction_vy += scale * dy
            if (
                self.predicted_positions is None
                or i >= self.predicted_positions.shape[0]
            ):
                continue
            modes = self.predicted_positions.shape[1]
            kept, forces, dxs, dys = [], [], [], []
            for k in range(modes):
                dxk = robot.px - float(self.predicted_positions[i, k, 0])
                dyk = robot.py - float(self.predicted_positions[i, k, 1])
                dist = np.sqrt(dxk**2 + dyk**2)
                if dist < 1e-6:
                    continue
                kept.append(k)
                forces.append(A * np.exp((radii - dist) / B) / dist)
                dxs.append(dxk)
                dys.append(dyk)
            if not forces:
                continue
            forces = np.asarray(forces)
            dxs = np.asarray(dxs)
            dys = np.asarray(dys)
            count = modes
            if self.locoval_threshold is not None:
                assert self.scores is not None
                scores = np.asarray([float(self.scores[i, k]) for k in kept])
                keep = scores >= self.locoval_threshold
                if not keep.any():
                    keep = np.zeros_like(scores, dtype=bool)
                    keep[int(np.argmax(scores))] = True
                forces, dxs, dys = forces[keep], dxs[keep], dys[keep]
                count = len(forces)
            renorm = float(modes) / count
            interaction_vx += float(np.sum(forces * dxs)) * renorm
            interaction_vy += float(np.sum(forces * dys)) * renorm

        total_delta_vx = (curr_delta_vx + interaction_vx) * self.config.env.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.config.env.time_step
        new_vx = robot.vx + total_delta_vx
        new_vy = robot.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])
        if act_norm > robot.v_pref:
            return ActionXY(
                new_vx / act_norm * robot.v_pref, new_vy / act_norm * robot.v_pref
            )
        return ActionXY(new_vx, new_vy)
