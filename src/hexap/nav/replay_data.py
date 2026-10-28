"""Build the replay scenes of the navigation experiment from the JTA data of Social-Transmotion.

    python -m hexap.nav.replay_data --source SOCIAL_TRANSMOTION_JTA_DIR --out data/jta_replay [--splits test]

Every `.ndjson` file of `<source>/<split>/` is one recorded sequence. It yields one
scene per pedestrian, with that pedestrian first and all pedestrians of the sequence in
it, over the whole sequence. Frames in which a pedestrian is absent repeat the nearest
frame in which it is present. The scenes are written as
`<out>/<split>/part_*.pt`, with the same joints as `hexap.data.preprocess jta`: each
agent is a pair `(joints, mask)` of shape (frames, 25, 4) and (frames, 25).
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import tqdm

from hexap.data.dataset import Window
from hexap.data.preprocess import JTA_TO_SMPL, save_parts

# Importing hexap.data.preprocess puts third_party/social_transmotion on sys.path.
from utils.trajnetplusplustools import Reader_jta_all_visual_cues

SCENES_PER_PART = 5000


def freeze_fill(tokens: np.ndarray) -> np.ndarray:
    """Fills the NaN frames of every pedestrian of (frames, N, 25, 4) with the nearest present frame."""
    filled = tokens.copy()
    frames, num_agents = filled.shape[:2]
    for n in range(num_agents):
        last = None
        for t in range(frames):
            if not np.isnan(filled[t, n, 0, 0]):
                last = t
            elif last is not None:
                filled[t, n] = filled[last, n]
        first = next(
            (t for t in range(frames) if not np.isnan(filled[t, n, 0, 0])), None
        )
        if first is None:
            filled[:, n] = 0.0
        elif first > 0:
            filled[:first, n] = filled[first, n]
    return filled


def sequence_tokens(reader: Reader_jta_all_visual_cues) -> Tuple[np.ndarray, int]:
    """(frames, N, 25, 4) tokens of all pedestrians of a sequence, NaN where absent.

    Token 0 is the path (x, y, 0, 0), tokens 1-24 the SMPL joints with the JTA joints
    mapped to their slots and the other slots 0.
    """
    frames = sorted(reader.tracks_by_frame.keys())
    pedestrians = sorted(
        {row.pedestrian for rows in reader.tracks_by_frame.values() for row in rows}
    )
    frame_index = {f: i for i, f in enumerate(frames)}
    pedestrian_index = {p: i for i, p in enumerate(pedestrians)}
    tokens = np.full((len(frames), len(pedestrians), 25, 4), np.nan, dtype=np.float32)
    for frame, rows in reader.tracks_by_frame.items():
        for row in rows:
            entry = tokens[frame_index[frame], pedestrian_index[row.pedestrian]]
            entry[:] = 0.0
            entry[0, 0] = row[2]
            entry[0, 1] = row[3]
            for source, target in JTA_TO_SMPL:
                entry[1 + target, 0] = row[12 + source]
                entry[1 + target, 1] = row[34 + source]
                entry[1 + target, 2] = row[56 + source]
    return tokens, len(pedestrians)


def split_scenes(directory: Path) -> List[Window]:
    scenes: List[Window] = []
    for path in tqdm.tqdm(sorted(directory.glob("*.ndjson")), desc=directory.name):
        reader = Reader_jta_all_visual_cues(str(path))
        if not reader.tracks_by_frame:
            continue
        tokens, num_agents = sequence_tokens(reader)
        tokens = freeze_fill(tokens)
        # Flip x and z to the right-handed layout of the other datasets.
        tokens[..., 0] = -tokens[..., 0]
        tokens[..., 2] = -tokens[..., 2]
        agents = [
            (
                torch.from_numpy(np.ascontiguousarray(tokens[:, n])).to(torch.float16),
                torch.ones(tokens.shape[0], 25, dtype=torch.float16),
            )
            for n in range(num_agents)
        ]
        for primary in range(num_agents):
            scenes.append(
                [agents[primary]] + [a for n, a in enumerate(agents) if n != primary]
            )
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/jta_replay"), type=Path)
    parser.add_argument("--splits", nargs="+", default=["test"])
    args = parser.parse_args()
    for split in args.splits:
        save_parts(split_scenes(args.source / split), args.out / split, SCENES_PER_PART)


if __name__ == "__main__":
    main()
