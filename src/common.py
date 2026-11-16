"""Shared constants and the match container used by every loader.

Every data source (the competition ATD, the DFL open matches, the Metrica sample games) is
converted to the same dictionary, so one feature builder and one decoder serve all of them:

    n_frames     int
    pitch        (length_m, width_m)
    team_map     int8 [n_slots]            0 = home, 1 = away
    px, py       float32 [n_frames, n_slots]   normalised 0..1, NaN where a slot is not tracked
    ball_x/y     float32 [n_frames]            normalised 0..1, NaN where the ball is not tracked
    half_blocks  [(half, first_frame, last_frame)]   the stretches that are play time
    labels       int8 [n_frames] or None   H/A/D/X codes (external games only)
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTERNAL_DIR = os.path.join(ROOT, "data", "external")
MODEL_DIR = os.path.join(ROOT, "models")

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = ("H", "A", "D", "X")
FPS = 25


def swap_teams(match):
    """The same match with home and away exchanged (labels included).

    Used twice: as a training augmentation, and at inference, where the prediction for a frame is
    the average of the model on the match and on its swapped copy. That makes the H/A call exactly
    symmetric -- the model cannot prefer a side.
    """
    out = dict(match)
    out["team_map"] = (1 - match["team_map"]).astype(match["team_map"].dtype)
    labels = match.get("labels")
    if labels is not None:
        swapped = labels.copy()
        swapped[labels == H] = A
        swapped[labels == A] = H
        out["labels"] = swapped
    return out
