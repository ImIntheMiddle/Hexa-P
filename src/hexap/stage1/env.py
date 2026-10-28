"""Path-following environment of the physics pre-training.

The humanoid follows a random target path with its pelvis. Besides the upstream
path observation, every step records the target path in world coordinates
(`path_non_rotated`) and the pelvis position at each path sample time
(`actual_waypoints`, with a third channel that marks the samples already
reached). The reward adds a heading term and an optional power penalty to the
upstream position reward.

This file is a derivative work of `protomotions/envs/path_follower/env.py` of ProtoMotions
(Copyright (c) NVIDIA Corporation) and is distributed under the NVIDIA License
(third_party/protomotions/LICENSE), not under the Apache License 2.0 of the rest of
Hexa-P. It may be used for research or evaluation purposes only (non-commercially).
"""

import torch
from torch import Tensor

from protomotions.envs.path_follower.env import (
    PathFollowing,
    compute_humanoid_reset,
    compute_path_observations,
    compute_path_reward,
)
from protomotions.envs.path_follower.path_generator import PathGenerator


class InitialStatePathGenerator(PathGenerator):
    """Random paths whose first segment can follow the initial velocity of the humanoid.

    With `adjust_root_vel` the segment speeds are scaled so that the path starts at
    the speed of the humanoid; with `init_heading` the path is rotated about its
    start so that it starts in the walking direction of the humanoid.
    """

    def reset(self, env_ids, init_pos, root_vel=None):
        super().reset(env_ids, init_pos)
        if len(env_ids) == 0 or root_vel is None:
            return
        verts = self.verts[env_ids].clone()
        if self.config.adjust_root_vel:
            step = verts[:, 1:, :2] - verts[:, :-1, :2]
            length = torch.norm(step, dim=-1)
            direction = step / length.unsqueeze(-1)
            speed = length / self.dt
            ratio = torch.norm(root_vel[:, :2], dim=-1) / speed[:, 0]
            speed = torch.clip(ratio.unsqueeze(-1) * speed, self.config.speed_min, self.config.speed_max)
            verts[:, 1:, :2] = verts[:, :1, :2] + torch.cumsum(direction * (speed * self.dt).unsqueeze(-1), dim=1)
        if self.config.init_heading:
            eps, threshold = 1e-8, 1e-6
            first = verts[:, 1, :2] - verts[:, 0, :2]
            root_heading = torch.where(
                torch.sqrt(torch.sum(root_vel**2, dim=1) + eps) > threshold,
                torch.atan2(root_vel[..., 1], root_vel[..., 0]),
                torch.zeros_like(root_vel[..., 0]),
            )
            path_heading = torch.where(
                torch.sqrt(torch.sum(first**2, dim=1) + eps) > threshold,
                torch.atan2(first[..., 1], first[..., 0]),
                torch.zeros_like(first[..., 0]),
            )
            diff = path_heading - root_heading
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            c, s = torch.cos(diff), torch.sin(diff)
            rot = torch.stack([c, -s, s, c], dim=-1).view(-1, 2, 2)
            origin = verts[:, :1, :2].clone()
            verts[:, :, :2] = torch.bmm(verts[:, :, :2] - origin, rot) + origin
        self.verts[env_ids] = verts


class PelvisPathFollowing(PathFollowing):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config, device, *args, **kwargs)
        # Upstream code tracks `head_body_id`; Hexa-P tracks the pelvis everywhere.
        self.head_body_id = self.simulator.build_body_ids_tensor(["Pelvis"]).item()
        n = self._num_traj_samples
        self._waypoint_times = (
            (
                torch.arange(n, device=device, dtype=torch.float)
                * self._traj_sample_timestep
            )
            .unsqueeze(0)
            .repeat_interleave(self.num_envs, dim=0)
        )
        self.path_non_rotated = torch.zeros((self.num_envs, n, 2), device=device)
        self.actual_waypoints = torch.zeros((self.num_envs, n, 3), device=device)
        self._fail_dist = self.config.path_follower_params.fail_dist
        self._fail_height_dist = self.config.path_follower_params.fail_height_dist
        self.tar_dist = torch.zeros(self.num_envs, device=device)
        self.pos_dist = torch.zeros(self.num_envs, device=device)
        self._prev_root_pos = torch.zeros(self.num_envs, 3, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super(PathFollowing, self).reset(env_ids)
        if len(env_ids) > 0:
            pelvis = (
                self.simulator.get_bodies_state(env_ids)
                .rigid_body_pos[:, self.head_body_id]
                .view(-1, 3)
            )
            pelvis[..., 2] -= self.terrain.get_ground_heights(pelvis[..., :2]).view(-1)
            root_vel = self.simulator.get_root_state(env_ids).root_vel
            self.path_generator.reset(env_ids, pelvis, root_vel)
            self.actual_waypoints[env_ids] = 0.0
        self.compute_observations()
        return self.get_obs()

    def compute_observations(self, env_ids=None):
        super(PathFollowing, self).compute_observations(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        bodies = self.simulator.get_bodies_state(env_ids).rigid_body_pos
        pelvis = bodies[:, self.head_body_id, :]
        flat_pelvis = pelvis.view(-1, 3)
        flat_pelvis[..., 2] -= torch.min(bodies, dim=1).values[..., 2].view(-1)
        samples = self.fetch_path_samples(env_ids)
        self.path_non_rotated[env_ids] = samples[..., :2]
        self.path_obs[env_ids] = compute_path_observations(
            self.simulator.get_root_state(env_ids).root_rot,
            flat_pelvis,
            samples,
            self.config.path_follower_params.height_conditioned,
        )
        # Record where the pelvis is at each path sample time of the episode.
        time = (self.progress_buf[env_ids] * self.dt).unsqueeze(-1)
        hit = torch.where(
            torch.isclose(
                self._waypoint_times[env_ids], time.expand(-1, self._num_traj_samples)
            )
        )
        self.actual_waypoints[env_ids[hit[0]], hit[1], :2] = pelvis[hit[0], :2]
        self.actual_waypoints[env_ids[hit[0]], hit[1], 2] = 1

    def get_obs(self):
        obs = super().get_obs()
        obs.update(
            {
                "path_non_rotated": self.path_non_rotated,
                "actual_waypoints": self.actual_waypoints,
            }
        )
        return obs

    def build_path_generator(self):
        self.path_generator = InitialStatePathGenerator(
            self.config.path_follower_params.path_generator,
            self.device,
            self.num_envs,
            self.config.max_episode_length * self.dt,
            self.config.path_follower_params.height_conditioned,
        )

    def compute_reward(self):
        bodies = self.simulator.get_bodies_state().rigid_body_pos
        pelvis = bodies[:, self.head_body_id, :]
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)
        prev_tar_pos = self.path_generator.calc_pos(env_ids, time - self.dt)
        pelvis[..., 2] -= torch.min(bodies, dim=1).values[..., 2].view(-1)

        reward = compute_path_reward(
            pelvis, tar_pos, self.config.path_follower_params.height_conditioned
        )
        self.pos_dist = torch.norm(pelvis - tar_pos, dim=-1)
        heading = compute_heading_reward(
            pelvis, self._prev_root_pos, tar_pos, prev_tar_pos, self.dt
        )
        self._prev_root_pos = pelvis.clone()
        reward = reward + heading * self.config.heading_reward_w
        if self.config.power_reward_w > 0:
            power = torch.abs(
                self.simulator.get_dof_forces() * self.simulator.get_dof_state().dof_vel
            ).sum(dim=-1)
            reward = reward - power * self.config.power_reward_w
        self.rew_buf[:] = reward

    def compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)
        bodies = self.simulator.get_bodies_state().rigid_body_pos
        bodies[..., 2] -= torch.min(bodies, dim=1).values[:, 2].view(-1, 1)
        # The contact buffer is boolean; the upstream TorchScript function needs floats.
        contacts = self.self_obs_cb.body_contacts.clone().float()
        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            contacts,
            self.non_termination_contact_body_ids,
            bodies,
            tar_pos,
            self.config.max_episode_length,
            self._fail_dist,
            self._fail_height_dist,
            self.config.enable_height_termination,
            self.config.path_follower_params.enable_path_termination,
            self.config.path_follower_params.height_conditioned,
            self.termination_heights + self.terrain.get_ground_heights(bodies[:, self.head_body_id, :2]),
            self.head_body_id,
        )
        self.tar_dist[:] = torch.norm(tar_pos - bodies[:, self.head_body_id, :], dim=-1)


@torch.jit.script
def compute_heading_reward(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    tar_pos: Tensor,
    prev_tar_pos: Tensor,
    dt: float,
) -> Tensor:
    """Rewards moving along the target direction at the target speed."""
    root_vel = (root_pos - prev_root_pos) / dt
    tar_vel = (tar_pos - prev_tar_pos)[..., :2] / dt
    tar_speed = torch.norm(tar_vel, dim=-1)
    tar_dir = tar_vel / (tar_speed.unsqueeze(-1) + 1e-8)
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tangent_speed = torch.sum(
        root_vel[..., :2] - tar_dir_speed.unsqueeze(-1) * tar_dir, dim=-1
    )
    tar_vel_err = tar_speed - tar_dir_speed
    reward = torch.exp(
        -0.25 * (tar_vel_err * tar_vel_err + 0.1 * tangent_speed * tangent_speed)
    )
    reward[tar_dir_speed < -0.5] = 0
    return reward
