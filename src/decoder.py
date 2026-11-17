"""Turn per-frame class probabilities into possessions.

Supports both the original 3-state constant-penalty Viterbi (`decode`) and the Hybrid
Physical + Smart-Tagging + Touch-Modulated Forward-Backward MBR decoder (`decode_hybrid`).
"""

import numpy as np

from .common import A, D, H, X
from .mbr_decoder import (
    extract_spells, fast_spell_f1, ffbs_sample, forward_log_alpha, viterbi_tv,
)

# Original crossdomain defaults
DEFAULT_PENALTIES = {"switch": 6.0, "stop": 8.0, "restart": 8.0}
MIN_PROB = 1e-4
LABEL_ADVANCE_FRAMES = 3

# Hybrid defaults validated on unseen external Metrica (light + heavy)
HYBRID_PENALTIES = {"switch": 20.0, "stop": 22.0, "restart": 22.0}
HYBRID_ADVANCE_FRAMES = 2


def penalty_matrix(switch, stop, restart):
    """cost[i][j] of moving from state i to state j, in log-probability units."""
    cost = [[0.0] * 3 for _ in range(3)]
    cost[H][A] = cost[A][H] = switch
    cost[H][D] = cost[A][D] = stop
    cost[D][H] = cost[D][A] = restart
    return cost


def viterbi3(logp, cost):
    """Most likely 3-state path for log-probabilities [T, 3] under additive transition costs."""
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
        b0, v0 = 0, s0
        if s1 - c10 > v0:
            b0, v0 = 1, s1 - c10
        if s2 - c20 > v0:
            b0, v0 = 2, s2 - c20
        b1, v1 = 1, s1
        if s0 - c01 > v1:
            b1, v1 = 0, s0 - c01
        if s2 - c21 > v1:
            b1, v1 = 2, s2 - c21
        b2, v2 = 2, s2
        if s0 - c02 > v2:
            b2, v2 = 0, s0 - c02
        if s1 - c12 > v2:
            b2, v2 = 1, s1 - c12
        back_rows.append((b0, b1, b2))
        s0, s1, s2 = v0 + e0, v1 + e1, v2 + e2
    back[1:] = np.array(back_rows, np.int8).reshape(-1, 3)

    state = 0 if (s0 >= s1 and s0 >= s2) else (1 if s1 >= s2 else 2)
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


def _clean_micro(codes, min_poss=8, min_dead=16, oob_slice=None, sp_slice=None):
    """Remove physically implausible micro-flickers while preserving confirmed OOB/SP segments."""
    out = codes.copy()
    n = len(out)
    if n == 0:
        return out
    if min_dead > 0:
        while True:
            br = np.flatnonzero(out[1:] != out[:-1]) + 1
            bounds = np.concatenate(([0], br, [n]))
            if len(bounds) <= 2:
                break
            removed = False
            for i in range(len(bounds) - 1):
                a, b = int(bounds[i]), int(bounds[i + 1])
                if out[a] == D and (b - a) < min_dead:
                    if oob_slice is not None and float(np.mean(oob_slice[a:b])) > 0.3:
                        continue
                    if sp_slice is not None and np.any(sp_slice[a:b]):
                        continue
                    out[a:b] = out[a - 1] if i > 0 else out[b]
                    removed = True
                    break
            if not removed:
                break
    if min_poss > 0:
        while True:
            br = np.flatnonzero(out[1:] != out[:-1]) + 1
            bounds = np.concatenate(([0], br, [n]))
            if len(bounds) <= 2:
                break
            durs = bounds[1:] - bounds[:-1]
            sts = out[bounds[:-1]]
            bad = np.flatnonzero((durs < min_poss) & ((sts == H) | (sts == A)))
            if len(bad) == 0:
                break
            i = int(bad[0])
            a, b = int(bounds[i]), int(bounds[i + 1])
            out[a:b] = out[a - 1] if i > 0 else out[b]
    return out


def decode(prob, half_blocks, penalties=None, forced_dead=(), advance_frames=LABEL_ADVANCE_FRAMES):
    """Original crossdomain 3-state constant-penalty Viterbi decode."""
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
        codes[first:last + 1] = D
    return codes


def decode_hybrid(prob, half_blocks, cues, smart=None, forced_dead=(),
                  penalties=None, touch_mult=0.50, pass_mult=1.50, live_shift=1.6,
                  use_mbr=True, mbr_temp=0.9, mbr_samples=120, seed=42,
                  advance_frames=HYBRID_ADVANCE_FRAMES):
    """Hybrid decoder: DFL probabilities + physical ball cues + Smart Tagging + MBR."""
    penalties = penalties or HYBRID_PENALTIES
    sw = float(penalties["switch"])
    st = float(penalties["stop"])
    rs = float(penalties["restart"])

    n = len(prob)
    logp = np.log(np.clip(prob, MIN_PROB, 1.0)).astype(np.float64)

    # 1. Physical ball-state fusion (same physics on Metrica and BAR-ATH)
    live_in = cues["live_in"]
    logp[:, H] += np.where(live_in, live_shift, 0.0)
    logp[:, A] += np.where(live_in, live_shift, 0.0)
    logp[:, D] += 4.5 * cues["oob_ev"] + 2.2 * cues["near_line_ev"] + 1.6 * cues["s_ign_ev"]

    # 2. Optional Smart Tagging soft log-odds priors & kickoff/restart anchors
    sp_boost = np.zeros(n, dtype=bool)
    sp_mask = np.zeros(n, dtype=bool)
    if smart is not None:
        tb = smart["team_bias"]
        logp[:, H] += 0.5 * tb
        logp[:, A] -= 0.5 * tb
        logp[:, D] += smart["dead_bias"]
        fl = smart["force_live"]
        # Confirmed kick-off / restart frames must exit D promptly even if players lag behind
        logp[fl, D] = np.minimum(logp[fl, D], -12.0)
        sp_boost = smart["sp_boost"]
        sp_mask = smart["sp_mask"]

    for first, last in forced_dead:
        logp[first:last + 1, H] = logp[first:last + 1, A] = -1e3
        logp[first:last + 1, D] = 0.0

    # 3. Touch- and boundary-modulated transition costs
    p_ha = np.where(cues["touch"], sw * touch_mult, np.where(cues["fast_pass"], sw * pass_mult, sw))
    if smart is not None:
        p_ha = np.where(sp_boost, np.minimum(p_ha, 8.0), p_ha)
    p_ad = np.where(cues["oob"], min(st, 4.0), st)
    p_da = np.where(sp_boost, min(rs, 6.0), rs)

    oob = cues["oob"]
    codes = np.full(n, X, dtype=np.int8)
    for h_idx, (_, first, last) in enumerate(half_blocks):
        s, e1 = first, last + 1
        em_h = logp[s:e1]
        pha_h, pad_h, pda_h = p_ha[s:e1], p_ad[s:e1], p_da[s:e1]
        oob_h, sp_h = oob[s:e1], sp_mask[s:e1]

        if not use_mbr:
            codes[s:e1] = viterbi_tv(em_h, pha_h, pad_h, pda_h)
            continue

        alpha = forward_log_alpha(em_h, pha_h, pad_h, pda_h, temp=mbr_temp)
        samp = ffbs_sample(alpha, pha_h, pad_h, pda_h, temp=mbr_temp,
                           n_samples=mbr_samples, seed=seed + h_idx)
        samp_spells = [
            extract_spells(_clean_micro(samp[k], min_poss=8, min_dead=16, oob_slice=oob_h, sp_slice=sp_h))
            for k in range(mbr_samples)
        ]

        candidates = []
        for sc in (0.80, 0.90, 1.0, 1.15, 1.30):
            rv = viterbi_tv(em_h, pha_h * sc, pad_h * sc, pda_h * sc)
            for mp in (0, 10, 14):
                candidates.append(_clean_micro(rv, min_poss=mp, min_dead=16, oob_slice=oob_h, sp_slice=sp_h))
        for k in range(8):
            candidates.append(_clean_micro(samp[k], min_poss=10, min_dead=16, oob_slice=oob_h, sp_slice=sp_h))

        best_cand = candidates[0]
        best_score = -1.0
        seen = set()
        for cand in candidates:
            sp = extract_spells(cand)
            sig = tuple(sp)
            if sig in seen:
                continue
            seen.add(sig)
            avg = sum(fast_spell_f1(ssp, sp) for ssp in samp_spells) / float(mbr_samples)
            if avg > best_score:
                best_score = avg
                best_cand = cand
        codes[s:e1] = best_cand

    codes = advance(codes, half_blocks, advance_frames)
    for first, last in forced_dead:
        codes[first:last + 1] = D
    return codes
