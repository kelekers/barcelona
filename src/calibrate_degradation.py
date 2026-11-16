"""Check that the corruption settings in degrade.py bracket the real automatic tracking.

Prints the same UNLABELLED summary statistics for the competition ATD and for an external match
under each corruption setting. No label of any kind is used: these are properties of the tracking
signal itself (how often the ball is missing and in what block lengths, how far the nearest
player is from the ball, how smooth the ball track is, how many players are tracked).

    python src/calibrate_degradation.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.atd_io import load_atd  # noqa: E402
from src.degrade import SETTINGS, degrade  # noqa: E402
from src.dfl_io import MATCH_IDS, load_dfl_match  # noqa: E402
from src.features import build_features  # noqa: E402
from src.model import play_mask  # noqa: E402

QUANTILES = (10, 25, 50, 75, 90)


def describe(name, match):
    features, names = build_features(match)
    col = {k: i for i, k in enumerate(names)}
    play = play_mask(match)
    visible = np.isfinite(match["ball_x"])
    seen = play & visible

    def q(key, mask):
        return " ".join("{:5.2f}".format(v) for v in np.nanpercentile(features[mask, col[key]], QUANTILES))

    gaps = np.diff(np.concatenate(([0], (play & ~visible).astype(np.int8), [0])))
    lengths = np.flatnonzero(gaps == -1) - np.flatnonzero(gaps == 1)
    print("{:<14} ball missing {:4.1f}% in {:3d} gaps (median {:4.0f} fr) | tracked {:4.1f}".format(
        name, 100 * (play & ~visible).sum() / play.sum(), len(lengths),
        np.median(lengths) if len(lengths) else 0, np.isfinite(match["px"][play]).sum(1).mean()))
    print("{:<14} dmin {} | b_acc {} | rel(min) {} | mps {}".format(
        "", q("dmin", seen), q("b_acc", seen),
        " ".join("{:5.2f}".format(v) for v in np.nanpercentile(
            np.fmin(features[seen, col["rel_H"]], features[seen, col["rel_A"]]), QUANTILES)),
        q("mps", play)))


if __name__ == "__main__":
    print("quantiles shown: {}".format(QUANTILES))
    describe("ATD (target)", load_atd(verbose=False))
    match = load_dfl_match(MATCH_IDS[0])
    for setting in ("clean", "light", "heavy"):
        describe("DFL " + setting, degrade(match, setting, seed=1))
