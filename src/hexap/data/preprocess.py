"""Build the Stage-2 windows from the public datasets.

    python -m hexap.data.preprocess jta --source SOCIAL_TRANSMOTION_JTA_DIR --out data
    python -m hexap.data.preprocess vru --source VRU_PEDESTRIANS_DIR --out data
    python -m hexap.data.preprocess jrdb --windows JRDB_2DBOX_PKL_DIR --smpl SMPL_JOINTS_DIR --out data

`jta` reads the JTA windows released with Social-Transmotion (`train/` and `test/` with
`.ndjson` files). `vru` reads `training_validation.zarr` and `test.zarr` of the VRU
pedestrian dataset (Kress et al., Zenodo 4898838). `jrdb` replaces the pose of the
Social-Transmotion JRDB 2D-box windows with SMPL joints regressed from the JRDB images
(see the README for the pose regression). Every split is written as
`<out>/<dataset>/<split>/part_*.pt` in the layout read by `hexap.data.dataset`.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tqdm

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "third_party" / "social_transmotion")
)
from utils.data import drop_distant_far, drop_ped_with_missing_frame, prepare_data  # noqa: E402
from utils.trajnetplusplustools import Reader_jta_all_visual_cues  # noqa: E402

# Joint indices (source, target) into the 24-joint SMPL layout of token 1-24.
JTA_TO_SMPL = np.array(
    [
        [15, 0],
        [19, 5],
        [20, 6],
        [21, 7],
        [16, 1],
        [17, 2],
        [18, 3],
        [12, 9],
        [11, 10],
        [2, 12],
        [1, 13],
        [8, 20],
        [9, 21],
        [10, 22],
        [4, 15],
        [5, 16],
        [6, 17],
    ]
)
VRU_TO_SMPL = np.array(
    [
        [8, 1],
        [10, 2],
        [12, 3],
        [7, 5],
        [9, 6],
        [11, 7],
        [0, 13],
        [4, 15],
        [5, 16],
        [6, 17],
        [3, 20],
        [2, 21],
        [1, 22],
    ]
)
SMPL_JOINTS_TO_TOKENS = torch.tensor(
    [
        [0, 0],
        [1, 1],
        [2, 4],
        [3, 7],
        [4, 10],
        [5, 2],
        [6, 5],
        [7, 8],
        [8, 11],
        [9, 3],
        [10, 6],
        [11, 9],
        [12, 12],
        [13, 15],
        [14, 13],
        [15, 16],
        [16, 18],
        [17, 20],
        [18, 22],
        [19, 14],
        [20, 17],
        [21, 19],
        [22, 21],
        [23, 23],
    ]
)
WINDOW = 21  # 9 observed + 12 future frames at 2.5 fps


def save_parts(windows: List, out: Path, part_size: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, max(len(windows), 1), part_size):
        torch.save(windows[i : i + part_size], out / f"part_{i // part_size}.pt")
    print(out, len(windows), "windows")


def jta(source: Path, out: Path) -> None:
    """JTA: map the JTA joints to SMPL slots and flip x and z to the right-handed layout of the other datasets."""
    for split in [s for s in ("train", "val", "test") if (source / s).is_dir()]:
        scenes, _, _ = prepare_data(
            str(source) + "/",
            subset=split,
            sample=1.0,
            goals=False,
            dataset_name="jta_all_visual_cues",
        )
        windows = []
        for _, _, paths in tqdm.tqdm(scenes, desc=f"jta {split}"):
            xy = drop_ped_with_missing_frame(
                Reader_jta_all_visual_cues.paths_to_xy(paths)
            )
            xy, _ = drop_distant_far(xy)
            people = np.transpose(
                xy.reshape(xy.shape[0], xy.shape[1], -1, 4), (1, 0, 2, 3)
            )  # (N, F, J, 4)
            tokens = np.zeros((people.shape[0], people.shape[1], 25, 4))
            tokens[:, :, 0] = people[:, :, 0]
            tokens[:, :, 1 + JTA_TO_SMPL[:, 1]] = people[:, :, 3 + JTA_TO_SMPL[:, 0]]
            tokens[..., 0] *= -1
            tokens[..., 2] *= -1
            for start in range(0, tokens.shape[1] - WINDOW + 1, WINDOW):
                window = torch.from_numpy(tokens[:, start : start + WINDOW]).to(
                    torch.float16
                )
                windows.append(
                    [
                        (agent, torch.ones(agent.shape[:-1], dtype=torch.float16))
                        for agent in window
                    ]
                )
        save_parts(windows, out / "jta" / split, 5000)


def vru(source: Path, out: Path) -> None:
    """VRU: 4 observed frames and 12 future frames at 5 fps from the 25 Hz recordings."""
    import zarr

    past_frames = [9, 14, 19, 24]
    future_frames = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    folds = {"train": 2, "val": 1, "test": 0}
    for split, fold in folds.items():
        data = zarr.open(
            str(
                source
                / ("test.zarr" if split == "test" else "training_validation.zarr")
            )
        )
        keep = np.where(np.array(data["fold"][:]) == fold)[0]
        pose = (
            np.array(data["pre_trajectories_and_poses"][:])
            .reshape(-1, 25, 3, 13)
            .transpose(0, 1, 3, 2)[keep]
        )
        past = ((pose[:, :, 7] + pose[:, :, 8]) / 2.0)[
            :, past_frames
        ]  # pelvis between the hips
        future = (
            np.array(data["pos_trajectories"][:])
            .reshape(-1, 2, 63)
            .transpose(0, 2, 1)[keep][:, future_frames]
        )
        smpl = np.zeros((len(keep), 4, 24, 3))
        smpl[:, :, VRU_TO_SMPL[:, 1]] = pose[:, past_frames][:, :, VRU_TO_SMPL[:, 0]]
        windows = []
        for i in range(len(keep)):
            tokens = np.zeros((16, 25, 4))
            tokens[:4, 0, :3] = past[i]
            tokens[4:, 0, :2] = future[i]
            tokens[:4, 1:, :3] = smpl[i]
            tokens = torch.tensor(tokens).to(torch.float16)
            windows.append(
                [(tokens, torch.ones(tokens.shape[:-1], dtype=torch.float16))]
            )
        save_parts(windows, out / "vru" / split, 10000)


def jrdb(windows_dir: Path, smpl_dir: Path, out: Path) -> None:
    """JRDB: insert the SMPL joints of each person, matched by scene, frame and person id."""
    for split in ("train", "val", "test"):
        windows, meta = [], []
        for part in sorted(
            (windows_dir / split).glob("*.pkl"),
            key=lambda p: int(p.stem.split("_")[-1]),
        ):
            with open(part, "rb") as f:
                data = pickle.load(f)
            scenes: Dict[str, Dict] = {}
            for window in tqdm.tqdm(data, desc=f"jrdb {split} {part.name}"):
                agents, info = [], []
                for traj, _, (scene, ids) in window:
                    if scene not in scenes:
                        scenes[scene] = torch.load(
                            next(smpl_dir.glob(scene.split("_shift")[0] + "*.pt"))
                        )
                    joints, mask = scene_joints(scenes[scene], ids)
                    tokens = torch.zeros((traj.shape[0], 25, traj.shape[2]))
                    tokens[:, :1] = traj[:, :1]
                    tokens[:, 1:, :3] = joints
                    agents.append(
                        (
                            tokens,
                            torch.cat(
                                [torch.ones((tokens.shape[0], 1)), mask.float()], dim=1
                            ),
                        )
                    )
                    info.append(
                        {
                            "scene": scene,
                            "frames": [
                                None if np.isnan(x) else int(x) for x in ids[:, 0]
                            ],
                            "ids": [None if np.isnan(x) else int(x) for x in ids[:, 1]],
                        }
                    )
                windows.append(agents)
                meta.append(info)
        save_parts(windows, out / "jrdb" / split, 5000)
        torch.save(meta, out / "jrdb" / split / "meta.pt")


def scene_joints(scene: Dict, ids: np.ndarray):
    """SMPL joints (frames, 24, 3) in token order and their validity mask for one person of a window."""
    frames = ids[:, 0]
    valid = ~np.isnan(frames)
    joints = torch.full((len(frames), 24, 3), float("nan"))
    mask = torch.zeros((len(frames), 24), dtype=torch.bool)
    match = np.where(scene["agent_ids"].numpy() == int(ids[0, 1]))[0]
    if valid.any() and len(match) > 0:
        a = match[0]
        rows = np.where(valid)[0]
        f = frames[rows].astype(int)
        ok = (~scene["x_pose_mask"][a][f]).bool()
        mask[rows] = ok.unsqueeze(1).expand(-1, 24)
        sel = mask[rows].unsqueeze(2).expand(-1, -1, 3)
        joints[rows] = joints[rows].masked_scatter_(sel, scene["Jtr"][a][f][ok])
    tokens = torch.zeros_like(joints)
    tokens[:, SMPL_JOINTS_TO_TOKENS[:, 0]] = joints[:, SMPL_JOINTS_TO_TOKENS[:, 1]]
    return tokens, mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=["jta", "vru", "jrdb"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--smpl", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.dataset == "jta":
        jta(args.source, args.out)
    elif args.dataset == "vru":
        vru(args.source, args.out)
    else:
        jrdb(args.windows, args.smpl, args.out)


if __name__ == "__main__":
    main()
