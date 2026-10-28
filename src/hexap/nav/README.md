# Robot navigation with predicted paths (Table 8)

A robot crosses replayed JTA crowds with a Social Force planner. The humans replay
recorded scenes and do not react to the robot. Each episode draws a random scene of the
test split. The robot starts 5 m below the centroid of the crowd and walks to the point
5 m above it, and the episode counts as a collision when the bodies of the robot and a
human overlap. Besides the current positions of the humans, the planner avoids the
predicted positions 2.0 s ahead of every predicted path, and the forces of the five
hypotheses of a human add up. The four rows of Table 8 differ in the predictor:

| Condition | Planner avoids |
| --- | --- |
| `none` | the current positions only |
| `social_trans` | the paths predicted by Social-Trans |
| `hexap` | the paths predicted by Hexa-P |
| `hexap_locoval` | the paths predicted by Hexa-P that LocoVal scores at least 0.5 |

LocoVal scores each hypothesis with the features of the Prediction Mode. A human with no
hypothesis above the threshold keeps its best-scored one. The surviving hypotheses are
scaled up so that the force of a human keeps the weight of five hypotheses.

## Install

The experiment uses the environment of the repository plus gym, Cython and
[Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2), which CrowdNav++ needs even
for the Social Force planner. From the repository root:

```bash
uv sync --extra nav
git clone https://github.com/sybrenstuvel/Python-RVO2.git
git -C Python-RVO2 checkout c2c46ba8d59556aa10faf03479293236efea154d
uv pip install --no-build-isolation ./Python-RVO2
```

Python-RVO2 is built from source with CMake and a C++ compiler. If the build stops with
`clang++: No such file or directory`, set `CC=gcc CXX=g++ LDSHARED="g++ -shared"` for the
last command. `uv sync` removes packages that are not in the lock file, so run it with
`--inexact` (or use `uv run`) after Python-RVO2 is installed.

CrowdNav++ is used unchanged from `third_party/crowdnav`.

## Replay data

The Hugging Face release contains the test scenes (`data/jta_replay/test/part_0.pt`).
To build them yourself, use the JTA data of the
[Social-Transmotion release](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data),
the source of `hexap.data.preprocess jta`:

```bash
uv run python -m hexap.nav.replay_data --source /path/to/social-transmotion/jta/data --out data/jta_replay
```

This writes `data/jta_replay/test/part_0.pt` with the 716 scenes of the test split. Every
pedestrian of a recorded sequence is the first agent of one scene, and a scene lasts the
whole sequence. `--splits train val` builds the other splits.

## Run

The `hexap` rows use Hexa-P fine-tuned on JTA from the Hugging Face release
(`weights/stage2/jta/hexap.pt`). The `social_trans` row uses the Social-Transmotion baseline
trained on JTA with `uv run python -m hexap.train dataset=jta model=social_trans`, which writes
`outputs/stage2/jta_social_trans/best.pt`. The four rows are then one command:

```bash
uv run --extra nav python -m hexap.nav.run --out output/nav.json
```

It runs 100 episodes per condition on the same scenes and prints one line per condition
with the collision and success rate, the fraction of steps with a gap below 0.25 m to a human, the
near misses (steps with a gap below 0.5 m) and their mean gap, and the mean path length
and time to the goal. `--out` writes all metrics to a JSON file. The collision rates of
Table 8 are 35% (`none`), 14% (`social_trans`), 10% (`hexap`) and 8% (`hexap_locoval`).

| Option | Default | |
| --- | --- | --- |
| `--hexap`, `--social-trans` | `weights/stage2/jta/hexap.pt`, `outputs/stage2/jta_social_trans/best.pt` | checkpoints |
| `--replay-root`, `--split` | `data/jta_replay`, `test` | replay scenes |
| `--conditions` | all four | any subset of the rows |
| `--locoval-threshold` | 0.5 | threshold of `hexap_locoval` |
| `--episodes`, `--seed` | 100, 0 | the seed sets the drawn scenes |

The predictor runs on the GPU if one is visible. Every episode takes the predictions of
all 20 humans of the scene at each of its steps, so use a GPU for the full run.
