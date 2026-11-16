"""Turn per-frame class probabilities into possessions.

The ranking metric matches whole possessions, so the output has to be a segmentation, not a
sequence of independent frame calls. A three-state Viterbi decode does that with three numbers:
the cost of handing the ball to the other team, of stopping play, and of restarting it. Each is
paid once per boundary, so a boundary is only placed where the accumulated frame evidence on the
far side outweighs it -- which is also where it gets placed, with no smoothing lag.

The three penalties are the only decoder hyper-parameters. They are chosen on external games
(validate_external.py) and never on the competition match.
"""

import numpy as np

from .common import A, D, H, X

# Chosen by validate_external.py on held-out games from other providers; see TECHNICAL_REPORT.md.
DEFAULT_PENALTIES = {"switch": 6.0, "stop": 8.0, "restart": 8.0}
MIN_PROB = 1e-4

# Every decoded label is moved this many frames earlier. The model is trained on DFL labels and
# places transitions a few frames later than the Metrica labels do (median +2 to +5 frames by
# transition type on the held-out Metrica games); the competition's ground truth comes from the
# Metrica side of that convention. Measured on those held-out games only: spellF1 rises on all
# six runs for an advance of 1-5 frames and peaks at 3. See TECHNICAL_REPORT.md, section 5.2.
LABEL_ADVANCE_FRAMES = 3


def penalty_matrix(switch, stop, restart):
    """cost[i][j] of moving from state i to state j, in log-probability units."""
    cost = [[0.0] * 3 for _ in range(3)]
    cost[H][A] = cost[A][H] = switch
    cost[H][D] = cost[A][D] = stop
    cost[D][H] = cost[D][A] = restart
    return cost


def viterbi3(logp, cost):
    """Most likely 3-state path for log-probabilities [T, 3] under additive transition costs.

    Written with Python scalars on purpose: with three states it is several times faster than
    the vectorised version, and bit-for-bit reproducible across numpy builds.
    """
    T = len(logp)
    rows = logp.tolist()
    back = np.zeros((T, 3), np.int8)
    s0, s1, s2 = rows[0]
    c01, c02 = cost[0][1], cost[0][2]
    c10, c12 = cost[1][0], cost[1][2]
    c20, c21 = cost[2][0], cost[2][1]
    back_rows = []
    for t in range(1, T):
        e0, e1, e2 = rows[t]
        # into state 0
        b0, v0 = 0, s0
        if s1 - c10 > v0:
            b0, v0 = 1, s1 - c10
        if s2 - c20 > v0:
            b0, v0 = 2, s2 - c20
        # into state 1
        b1, v1 = 1, s1
        if s0 - c01 > v1:
            b1, v1 = 0, s0 - c01
        if s2 - c21 > v1:
            b1, v1 = 2, s2 - c21
        # into state 2
        b2, v2 = 2, s2
        if s0 - c02 > v2:
            b2, v2 = 0, s0 - c02
        if s1 - c12 > v2:
            b2, v2 = 1, s1 - c12
        back_rows.append((b0, b1, b2))
        s0, s1, s2 = v0 + e0, v1 + e1, v2 + e2
    back[1:] = np.array(back_rows, np.int8).reshape(-1, 3)

    path = np.empty(T, np.int8)
    state = 0 if (s0 >= s1 and s0 >= s2) else (1 if s1 >= s2 else 2)
    path[-1] = state
    back_list = back.tolist()
    out = [0] * T
    out[-1] = state
    for t in range(T - 1, 0, -1):
        state = back_list[t][state]
        out[t - 1] = state
    return np.array(out, np.int8)


def advance(codes, half_blocks, frames):
    """Move every label `frames` frames earlier inside each half (the last label is repeated)."""
    if frames <= 0:
        return codes
    out = codes.copy()
    for _, first, last in half_blocks:
        segment = codes[first:last + 1]
        out[first:last + 1] = np.concatenate([segment[frames:], np.repeat(segment[-1], frames)])
    return out


def decode(prob, half_blocks, penalties=None, forced_dead=(), advance_frames=LABEL_ADVANCE_FRAMES):
    """Label codes for every frame: Viterbi inside each half, X outside.

    forced_dead is a list of (first, last) frame spans known to be dead for a documented,
    non-label reason (the graded stretch before the second-half kick-off).
    """
    penalties = penalties or DEFAULT_PENALTIES
    cost = penalty_matrix(penalties["switch"], penalties["stop"], penalties["restart"])
    logp = np.log(np.clip(prob, MIN_PROB, 1.0)).astype(np.float64)
    for first, last in forced_dead:
        logp[first:last + 1, H] = logp[first:last + 1, A] = -1e3
        logp[first:last + 1, D] = 0.0
    codes = np.full(len(prob), X, np.int8)
    for _, first, last in half_blocks:
        codes[first:last + 1] = viterbi3(logp[first:last + 1], cost)
    codes = advance(codes, half_blocks, advance_frames)
    for first, last in forced_dead:
        codes[first:last + 1] = D      # the documented span stays exactly where it is
    return codes
