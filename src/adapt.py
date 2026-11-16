"""Label-free adaptation to the match being labelled (cross-fitted co-training).

EXPERIMENTAL -- NOT PART OF THE SUBMISSION. Run on six held-out external games with exactly the
procedure below, it lowered spellF1 on five of them (mean -0.008) while raising macroF1 slightly:
the fused probabilities are smoother, which merges short possessions. It is kept behind
`run_pipeline.py --adapt` and `validate_external.py --adapt` as a record of what was tried.

The external model (model.py) decides dead/alive without ever looking at the ball, because the
ball's behaviour during stoppages is provider-specific. On the target recording that behaviour
is informative -- a ball over the touchline, a ball the tracker has lost for twenty seconds, a
ball being walked back to a corner flag -- but what it looks like here has to be learned here.
This step learns it from the target match itself, using no human label of any kind:

  1. Teacher. The external model labels the match (probabilities -> Viterbi decode).
  2. Student. A second model is trained on the target's own features -- all of them, ball
     included, plus any target-only side information such as the Smart Tagging -- against those
     pseudo-labels. Frames next to a pseudo-boundary are left out (the teacher is least reliable
     there). It is cross-fitted in time: the match is cut into one-minute chunks dealt into
     folds, and each fold is predicted by a student that never saw it, so the student cannot
     memorise the teacher's mistakes; it can only reproduce what generalises across the match.
     Both team orientations are trained on, so it cannot learn "this team usually has the ball".
  3. Fusion. Teacher and student log-probabilities are averaged and decoded again.

The teacher sees players, the student also sees the ball and the tags: two views whose errors
are largely independent, which is the condition under which co-training helps. RULES.md allows
this explicitly ("pseudo-labels produced by an algorithm, provided no person decided any of
them"). Whether it helps is measured on external games in validate_external.py, where the same
procedure is run on held-out matches and scored against their real labels.
"""

import numpy as np

from .common import A, D, H, X, swap_teams
from .decoder import DEFAULT_PENALTIES, decode
from .features import build_features

STUDENT_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.08, num_leaves=31,
    min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
    lambda_l2=1.0, max_bin=63, num_threads=6, seed=11, deterministic=True,
    force_row_wise=True, verbose=-1,
)
STUDENT_ROUNDS = 150
N_FOLDS = 4
CHUNK_FRAMES = 1500        # one minute
TRAIN_STRIDE = 4
BOUNDARY_MARGIN = 12       # frames either side of a pseudo-boundary left out of student training
STUDENT_WEIGHT = 0.5       # weight of the student in the log-linear fusion
PREDICT_CHUNK = 20000


def _swap_labels(codes):
    out = codes.copy()
    out[codes == H] = A
    out[codes == A] = H
    return out


def _with_extra(features, extra, rows):
    block = features[rows]
    return block if extra is None else np.hstack([block, extra[rows]])


def _student(match, pseudo, extra, verbose):
    """Cross-fitted student probabilities [n_frames, 3] for the given pseudo-labels.

    Only one full feature matrix is held in memory at a time (the two team orientations are
    built, used and released in turn).
    """
    import lightgbm as lgb

    n = match["n_frames"]
    play = pseudo != X
    change = np.zeros(n, bool)
    change[1:] = pseudo[1:] != pseudo[:-1]
    near_boundary = np.convolve(change, np.ones(2 * BOUNDARY_MARGIN + 1), mode="same") > 0
    fold = (np.arange(n) // CHUNK_FRAMES) % N_FOLDS
    train_rows = np.flatnonzero(play & ~near_boundary)[::TRAIN_STRIDE]
    train_fold = fold[train_rows]
    sources = ((match, None if extra is None else extra[0], pseudo),
               (swap_teams(match), None if extra is None else extra[1], _swap_labels(pseudo)))

    tables, targets = [], []
    for source, side, labels in sources:
        features, _ = build_features(source)
        tables.append(_with_extra(features, side, train_rows))
        targets.append(labels[train_rows].astype(np.int32))
        del features

    boosters = []
    for k in range(N_FOLDS):
        keep = train_fold != k
        data = np.concatenate([tables[0][keep], tables[1][keep]])
        target = np.concatenate([targets[0][keep], targets[1][keep]])
        boosters.append(lgb.train(STUDENT_PARAMS, lgb.Dataset(data, label=target, free_raw_data=True),
                                  num_boost_round=STUDENT_ROUNDS))
        del data
    del tables

    student = np.zeros((n, 3), np.float64)
    for orientation, (source, side, _) in enumerate(sources):
        features, _ = build_features(source)
        for k in range(N_FOLDS):
            held = np.flatnonzero(play & (fold == k))
            for start in range(0, len(held), PREDICT_CHUNK):
                rows = held[start:start + PREDICT_CHUNK]
                p = boosters[k].predict(_with_extra(features, side, rows))
                if orientation == 0:
                    student[rows] += 0.5 * p
                else:
                    student[rows, H] += 0.5 * p[:, A]
                    student[rows, A] += 0.5 * p[:, H]
                    student[rows, D] += 0.5 * p[:, D]
        del features
    if verbose:
        agree = float((student[play].argmax(1) == pseudo[play]).mean())
        print("  student trained on {} rows x 2 orientations; agrees with the pseudo-labels on "
              "{:.1f}% of held-out frames".format(len(train_rows), 100 * agree))
    return student, play


def fuse(teacher, student, play, weight=STUDENT_WEIGHT):
    fused = teacher.astype(np.float64).copy()
    log_fused = ((1.0 - weight) * np.log(np.clip(teacher[play], 1e-4, 1.0))
                 + weight * np.log(np.clip(student[play], 1e-4, 1.0)))
    log_fused -= log_fused.max(axis=1, keepdims=True)
    expd = np.exp(log_fused)
    fused[play] = expd / expd.sum(axis=1, keepdims=True)
    return fused


def adapt(match, teacher_prob, penalties=None, forced_dead=(), extra=None, rounds=1, verbose=False):
    """Fused [n_frames, 3] probabilities after `rounds` of cross-fitted co-training.

    extra: optional (forward, mirrored) arrays [n_frames, k] of target-only features for the
    student, given for the match as it is and with the two teams exchanged.
    """
    penalties = penalties or DEFAULT_PENALTIES
    prob = teacher_prob
    for _ in range(rounds):
        pseudo = decode(prob, match["half_blocks"], penalties, forced_dead=forced_dead)
        student, play = _student(match, pseudo, extra, verbose)
        prob = fuse(teacher_prob, student, play)
    return prob
