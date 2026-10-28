"""Physical executability of predicted paths in the simulation (Table 6).

    python -m hexap.executability +exp=hexap/executability +robot=smpl +simulator=isaacgym \
        motion_file=data/motions/stage1_motions.pt checkpoint=weights/stage1/controller.ckpt \
        agent.config.locoval_checkpoint=weights/stage1/hexap_pretrain.ckpt

Run it from the repository root. For every episode the controller starts from a reference
clip, the Stage-1 network predicts M paths from the observed motion, and the controller
then walks, from the same start state, (a) the sampled target path, (b) the predicted path
with the highest LocoVal score and (c) the first predicted path. A rollout fails when the
humanoid falls (a non-foot contact with the body below the termination height), and
succeeds when it does not fall and ends within `exec_success_radius` of the target.
"""

import isaacgym  # noqa: F401  (Isaac Gym must be imported before torch)

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import protomotions.utils.config_utils  # noqa: F401  (registers the OmegaConf resolvers)
from hexap.stage1.agent import HexaPPretrainer
from protomotions.simulator.base_simulator.robot_state import RobotState

PROTOMOTIONS_CONFIG = (
    Path(importlib.util.find_spec("protomotions").origin).parent / "config"
)
HEXAP_CONFIGS = Path(__file__).resolve().parent / "stage1" / "configs"


class Executability(HexaPPretrainer):
    def snapshot(self) -> None:
        sim = self.env.simulator
        root, dof, bodies = (
            sim.get_root_state(),
            sim.get_dof_state(),
            sim.get_bodies_state(),
        )
        self._start_fields = {
            "root_pos": root.root_pos.clone(),
            "root_rot": root.root_rot.clone(),
            "root_vel": root.root_vel.clone(),
            "root_ang_vel": root.root_ang_vel.clone(),
            "dof_pos": dof.dof_pos.clone(),
            "dof_vel": dof.dof_vel.clone(),
            "rigid_body_pos": bodies.rigid_body_pos.clone(),
            "rigid_body_rot": bodies.rigid_body_rot.clone(),
            "rigid_body_vel": bodies.rigid_body_vel.clone(),
            "rigid_body_ang_vel": bodies.rigid_body_ang_vel.clone(),
        }

    def restore(self) -> None:
        state = RobotState(**{k: v.clone() for k, v in self._start_fields.items()})
        self.env.simulator.reset_envs(state, torch.arange(self.num_envs, device=self.device))

    @torch.no_grad()
    def predict_paths(self, start_xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """World-coordinate paths (n, out_F, 2) of each condition."""
        cfg = self.config.locoval.config
        n, in_F, out_F = self.num_envs, cfg.input_track_size, cfg.output_track_size
        tgt = torch.zeros(
            (n, in_F, self.motion_lib.robot_config.num_bodies + 1, 4),
            device=self.device,
        )
        past_traj, past_poses = self.get_expert_historical_locoval_obs(
            {"env_ids": torch.arange(n, device=self.device)}
        )
        tgt[:, :, 0, :2] = past_traj
        tgt[:, :, 1:, :3] = past_poses
        padding = torch.zeros((n, 1), dtype=torch.bool, device=self.device)
        target = (
            self.env.path_non_rotated[:, 1:, :2] - self.env.path_non_rotated[:, :1, :2]
        )  # (n, out_F, 2)
        pred, pred_rew, _, _ = self.locoval.pretraining_outputs(
            tgt, padding, target.unsqueeze(2), False, cfg.nmode
        )
        pred = pred[:, in_F:]
        best = pred[torch.arange(n, device=self.device), :, pred_rew.argmax(dim=1)]
        shift = start_xy.unsqueeze(1)
        return {
            "ground_truth": target + shift,
            "hexap_best": best + shift,
            "hexap_first": pred[:, :, 0] + shift,
        }

    def set_path(self, path: torch.Tensor) -> None:
        """Writes waypoints at multiples of the path sample time onto the vertices of the path generator."""
        pg = self.env.path_generator
        out_F = path.shape[1]
        dt = self.env.config.path_follower_params.traj_sample_timestep
        wp_times = torch.arange(0, out_F + 1, device=self.device).float() * dt
        vert_times = (
            torch.arange(pg.get_num_verts(), device=self.device).float() * pg.dt
        )
        wp = torch.cat([pg.verts[:, :1, :2], path], dim=1)
        idx = torch.searchsorted(
            wp_times, vert_times.clamp(max=wp_times[-1]), right=True
        )
        i0, i1 = (idx - 1).clamp(0, out_F), idx.clamp(0, out_F)
        t0, t1 = wp_times[i0], wp_times[i1]
        w = (
            ((vert_times - t0) / (t1 - t0).clamp(min=1e-6))
            .clamp(0.0, 1.0)
            .view(1, -1, 1)
        )
        pg.verts[:, :, :2] = (1.0 - w) * wp[:, i0] + w * wp[:, i1]

    @torch.no_grad()
    def rollout(
        self, path: torch.Tensor, steps: int, radius: float
    ) -> Dict[str, torch.Tensor]:
        env, n = self.env, self.num_envs
        self.set_path(path)
        env.progress_buf[:] = 0
        env.reset_buf[:] = 0
        env.terminate_buf[:] = 0
        env.actual_waypoints[:] = 0.0
        env._prev_root_pos[:] = 0.0
        env.compute_observations()
        obs = env.get_obs()
        fell = torch.zeros(n, dtype=torch.bool, device=self.device)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        dist_sum = torch.zeros(n, device=self.device)
        dist_count = torch.zeros(n, device=self.device)
        last_dist = torch.full((n,), float("nan"), device=self.device)
        k = self.config.exec_record
        bodies = []
        for _ in range(steps):
            obs, _, dones, _ = env.step(self.model.act(obs))
            if k > 0:
                bodies.append(env.simulator.get_bodies_state().rigid_body_pos[:k].clone())
            fell |= env.terminate_buf.bool() & alive
            dist_sum += torch.where(alive, env.pos_dist, torch.zeros_like(env.pos_dist))
            dist_count += alive.float()
            last_dist = torch.where(alive, env.tar_dist, last_dist)
            alive &= ~dones.bool()
        out = {
            "fall": fell.float(),
            "tracking_error": dist_sum / dist_count.clamp(min=1.0),
            "final_distance": last_dist,
            "success": (~fell & (last_dist < radius)).float(),
        }
        if k > 0:
            out["bodies"] = torch.stack(bodies, dim=1).cpu()  # (k, steps, bodies, 3)
        return out

    @torch.no_grad()
    def run(self) -> Dict[str, Dict[str, float]]:
        self.eval()
        cfg = self.config
        results: Dict[str, Dict[str, list]] = {}
        records: Dict[str, list] = {}
        for episode in range(cfg.exec_num_episodes):
            self.env.reset()
            self.locoval_motion_ids = self.env.motion_manager.motion_ids.clone()
            self.locoval_motion_times = self.env.motion_manager.motion_times.clone()
            self.snapshot()
            start_xy = (
                self.env.simulator.get_bodies_state()
                .rigid_body_pos[:, self.env.head_body_id, :2]
                .clone()
            )
            for name, path in self.predict_paths(start_xy).items():
                self.restore()
                self.env.path_generator.verts[:, 0, :2] = start_xy
                metrics = self.rollout(
                    path.clone(), self.num_steps, cfg.exec_success_radius
                )
                if "bodies" in metrics:
                    k = cfg.exec_record
                    records.setdefault(name, []).append(
                        {"bodies": metrics.pop("bodies"), "path": path[:k].cpu(), "start": start_xy[:k].cpu(),
                         "fall": metrics["fall"][:k].cpu()}
                    )
                for k, v in metrics.items():
                    results.setdefault(name, {}).setdefault(k, []).append(v)
            print(f"episode {episode + 1}/{cfg.exec_num_episodes} done")
        summary = {
            name: {**{k: torch.cat(v).nanmean().item() for k, v in m.items()}, "n": len(torch.cat(m["fall"]))}
            for name, m in results.items()
        }
        for name, s in summary.items():
            print(
                f"{name:14s} fall {s['fall']:.3%}  tracking error {s['tracking_error']:.3f} m  "
                f"final distance {s['final_distance']:.3f} m  success {s['success']:.1%}  (n={s['n']})"
            )
        if records:
            torch.save(records, cfg.exec_record_path)
        if cfg.exec_out_csv:
            with open(cfg.exec_out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "condition",
                        "fall",
                        "tracking_error",
                        "final_distance",
                        "success",
                        "n",
                    ]
                )
                for name, s in summary.items():
                    writer.writerow(
                        [
                            name,
                            s["fall"],
                            s["tracking_error"],
                            s["final_distance"],
                            s["success"],
                            s["n"],
                        ]
                    )
        return summary


@hydra.main(config_path=str(PROTOMOTIONS_CONFIG), config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    fabric = instantiate(config.fabric)
    fabric.launch()
    fabric.seed_everything(config.seed)
    env = instantiate(config.env, device=fabric.device)
    agent: Executability = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.load(config.checkpoint)
    agent.run()


if __name__ == "__main__":
    sys.argv.append(f"hydra.searchpath=[file://{HEXAP_CONFIGS}]")
    main()
