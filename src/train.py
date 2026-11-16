"""Train the external frame model on the DFL open matches (other games, another provider).

Every match is used twice -- lightly and heavily corrupted (degrade.py) -- with the home/away
roles alternated, and sub-sampled in time because neighbouring frames are near copies.
No competition data and no competition label is read here.

    python src/download_external.py
    python src/train.py                          # final model, all 7 matches -> models/
    python src/train.py --holdout J03WR9 --out models_dev   # development model for validation
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import model as frame_model  # noqa: E402
from src.common import D, H, MODEL_DIR, swap_teams  # noqa: E402
from src.degrade import degrade  # noqa: E402
from src.dfl_io import MATCH_IDS, load_dfl_match  # noqa: E402
from src.features import ball_free_columns  # noqa: E402

# The two corrupted settings bracket the unlabelled statistics of the competition tracking
# (calibrate_degradation.py); the uncorrupted copy is left out, it resembles nothing we will see.
SETTINGS = ("light", "heavy")
STRIDE = 8
ROWS_PER_COPY = 26000     # upper bound used to pre-allocate the training tables


def collect(match_ids, stride=STRIDE, settings=SETTINGS, verbose=True):
    """Two pre-allocated tables, one per head (see model.train).

    Kept separate and filled in place because the machine this was developed on has 8 GB: the
    dead head needs few columns on every row, the team head needs every column on live rows only.
    """
    capacity = ROWS_PER_COPY * len(match_ids) * len(settings)
    dead_x = team_x = None
    dead_y = np.empty(capacity, np.int32)
    team_y = np.empty(capacity, np.int32)
    n_dead = n_team = 0
    names = free = None
    for gi, match_id in enumerate(match_ids):
        started = time.time()
        match = load_dfl_match(match_id)
        for si, setting in enumerate(settings):
            copy = degrade(match, setting, seed=1000 * gi + si)
            if (gi + si) % 2:
                copy = swap_teams(copy)
            rows, y, names = frame_model.training_rows(copy, stride, phase=(gi + 2 * si) % stride)
            if dead_x is None:
                free = ball_free_columns(names)
                dead_x = np.empty((capacity, len(free)), np.float32)
                team_x = np.empty((capacity, rows.shape[1]), np.float32)
            dead_x[n_dead:n_dead + len(y)] = rows[:, free]
            dead_y[n_dead:n_dead + len(y)] = y == D
            n_dead += len(y)
            live = y != D
            k = int(live.sum())
            team_x[n_team:n_team + k] = rows[live]
            team_y[n_team:n_team + k] = y[live] == H
            n_team += k
            del rows, copy
        if verbose:
            print("  {}: {} frames  ({:.0f}s)".format(match_id, match["n_frames"], time.time() - started))
            sys.stdout.flush()
    return dead_x[:n_dead], dead_y[:n_dead], team_x[:n_team], team_y[:n_team], names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", nargs="*", default=[], help="DFL match ids to leave out")
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--out", default=MODEL_DIR)
    args = parser.parse_args()

    train_ids = [m for m in MATCH_IDS if m not in args.holdout]
    print("training on {} DFL matches: {}".format(len(train_ids), " ".join(train_ids)))
    dead_x, dead_y, team_x, team_y, names = collect(train_ids, stride=args.stride)
    print("dead head: {} rows x {} ball-free features (dead share {:.3f})".format(
        dead_x.shape[0], dead_x.shape[1], dead_y.mean()))
    print("team head: {} live rows x {} features (home share {:.3f})".format(
        team_x.shape[0], team_x.shape[1], team_y.mean()))
    sys.stdout.flush()
    started = time.time()
    dead, team = frame_model.train(dead_x, dead_y, team_x, team_y, names)
    print("trained both heads in {:.0f}s".format(time.time() - started))
    frame_model.save(dead, team, names, {
        "training_matches": ["DFL-MAT-" + m for m in train_ids],
        "held_out": ["DFL-MAT-" + m for m in args.holdout],
        "settings": list(SETTINGS), "stride": args.stride,
        "dead_rows": int(dead_x.shape[0]), "team_rows": int(team_x.shape[0]),
        "dead_rounds": frame_model.DEAD_ROUNDS, "team_rounds": frame_model.TEAM_ROUNDS,
        "lgb_params": frame_model.LGB_PARAMS,
    }, model_dir=args.out)
    for label, booster in (("dead", dead), ("team", team)):
        gain = booster.feature_importance("gain")
        feature_names = booster.feature_name()
        order = np.argsort(gain)[::-1][:10]
        print("{} head, top features by gain: ".format(label) + ", ".join(
            "{} {:.1f}%".format(feature_names[i], 100 * gain[i] / gain.sum()) for i in order))
    print("saved ->", args.out)
