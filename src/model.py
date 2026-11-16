"""The external frame model: two gradient-boosted heads over the features in features.py.

  * dead head  -- P(dead ball), from BALL-FREE features only (the players' collective motion and
                  restart shapes). How a tracking provider's ball behaves during a stoppage is the
                  least transferable thing in the data: the DFL provider freezes it, the Metrica
                  sample drops it, a camera tracker follows it in a ball boy's hands. A head that
                  never sees the ball cannot learn any of those conventions.
  * team head  -- P(home has it | ball alive), from every feature, trained on live frames only,
                  so the dead-ball conventions cannot reach it either.

    P(D) = dead,   P(H) = (1 - dead) * team,   P(A) = (1 - dead) * (1 - team)

At inference both heads are run on the match and on its team-swapped copy and averaged, which
makes the H/A decision exactly symmetric.
"""

import json
import os

import numpy as np

from .common import A, D, H, MODEL_DIR, swap_teams
from .features import ball_free_columns, build_features

DEAD_FILE = "dead_head.txt"
TEAM_FILE = "team_head.txt"
META_FILE = "model_meta.json"

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.06, num_leaves=63, min_data_in_leaf=300,
    feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=127,
    num_threads=6, seed=7, deterministic=True, force_row_wise=True, verbose=-1,
)
DEAD_ROUNDS = 400
TEAM_ROUNDS = 500


def play_mask(match):
    mask = np.zeros(match["n_frames"], bool)
    for _, first, last in match["half_blocks"]:
        mask[first:last + 1] = True
    return mask


def training_rows(match, stride, phase, boundary_radius=8, boundary_stride=2):
    """(features, labels) for training: a sparse time grid plus a dense band around transitions.

    Neighbouring frames are near copies, so the interior of a possession is sampled every
    `stride` frames. The frames within `boundary_radius` of a label change are the ones the
    ranking metric is decided on, and they are rare, so they are sampled much more densely.
    """
    features, names = build_features(match)
    labels = match["labels"]
    keep = play_mask(match) & (labels <= D)
    grid = np.zeros(len(labels), bool)
    grid[phase::stride] = True
    change = np.zeros(len(labels), bool)
    change[1:] = labels[1:] != labels[:-1]
    band = np.convolve(change, np.ones(2 * boundary_radius + 1), mode="same") > 0
    dense = np.zeros(len(labels), bool)
    dense[phase % boundary_stride::boundary_stride] = True
    rows = np.flatnonzero(keep & (grid | (band & dense)))
    return features[rows], labels[rows].astype(np.int32), names


def train(dead_rows, dead_labels, team_rows, team_labels, names):
    """Fit both heads from their own tables. Returns (dead_booster, team_booster).

    dead_rows holds the ball-free columns for every sampled frame (label: 1 = dead);
    team_rows holds every column for the live frames only (label: 1 = home has the ball).
    """
    import lightgbm as lgb
    free = ball_free_columns(names)
    dead_data = lgb.Dataset(dead_rows, label=dead_labels, feature_name=[names[i] for i in free],
                            free_raw_data=True)
    dead = lgb.train(LGB_PARAMS, dead_data, num_boost_round=DEAD_ROUNDS)
    del dead_data
    team_data = lgb.Dataset(team_rows, label=team_labels, feature_name=list(names), free_raw_data=True)
    team = lgb.train(LGB_PARAMS, team_data, num_boost_round=TEAM_ROUNDS)
    return dead, team


def save(dead, team, names, extra, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    # newline="\n": LightGBM stores byte offsets, so the files must never be re-wrapped to CRLF.
    for booster, filename in ((dead, DEAD_FILE), (team, TEAM_FILE)):
        with open(os.path.join(model_dir, filename), "w", newline="\n") as handle:
            handle.write(booster.model_to_string())
    meta = dict(extra, feature_names=list(names))
    with open(os.path.join(model_dir, META_FILE), "w", newline="\n") as handle:
        json.dump(meta, handle, indent=1)


def _read_booster(path):
    import lightgbm as lgb
    with open(path, "rb") as handle:
        text = handle.read().replace(b"\r\n", b"\n").decode("utf-8")
    return lgb.Booster(model_str=text)


def load(model_dir=MODEL_DIR):
    """({"dead": booster, "team": booster, "free": column indices}, meta)."""
    with open(os.path.join(model_dir, META_FILE)) as handle:
        meta = json.load(handle)
    bundle = {"dead": _read_booster(os.path.join(model_dir, DEAD_FILE)),
              "team": _read_booster(os.path.join(model_dir, TEAM_FILE)),
              "free": ball_free_columns(meta["feature_names"])}
    return bundle, meta


def predict_proba(bundle, match, names=None):
    """[n_frames, 3] probabilities for H, A, D -- averaged over the two team orientations."""
    dead = np.zeros(match["n_frames"], np.float64)
    team = np.zeros(match["n_frames"], np.float64)
    for orientation, source in enumerate((match, swap_teams(match))):
        features, built = build_features(source)
        if names is not None and list(names) != list(built):
            raise ValueError("feature set differs from the one the model was trained on")
        dead += 0.5 * bundle["dead"].predict(np.ascontiguousarray(features[:, bundle["free"]]))
        home = bundle["team"].predict(features)
        team += 0.5 * (home if orientation == 0 else 1.0 - home)
        del features
    prob = np.empty((match["n_frames"], 3), np.float64)
    prob[:, D] = dead
    prob[:, H] = (1.0 - dead) * team
    prob[:, A] = (1.0 - dead) * (1.0 - team)
    return prob
