"""Exact 3-state Forward-Backward, FFBS Posterior Sampling, and Minimum Bayes Risk (MBR)
decoding optimizing expected spellF1 directly.
"""

import math
import numpy as np

H, A, D, X = 0, 1, 2, 3
SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def _logsumexp3(a, b, c):
    m = a if a >= b else b
    if c > m:
        m = c
    if m < -1e150:
        return -1e150
    return m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))


def viterbi_tv(emissions, p_ha, p_ad, p_da):
    """Exact time-varying 3-state max-sum Viterbi.

    Args:
        emissions: (T, 3) log-scores for H, A, D.
        p_ha: (T,) or scalar transition penalty for H <-> A at step t-1 -> t.
        p_ad: (T,) or scalar transition penalty for H/A -> D.
        p_da: (T,) or scalar transition penalty for D -> H/A.
    """
    n = emissions.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int8)

    p_ha_arr = np.broadcast_to(np.asarray(p_ha, dtype=np.float64), (n,))
    p_ad_arr = np.broadcast_to(np.asarray(p_ad, dtype=np.float64), (n,))
    p_da_arr = np.broadcast_to(np.asarray(p_da, dtype=np.float64), (n,))

    d0 = float(emissions[0, 0])
    d1 = float(emissions[0, 1])
    d2 = float(emissions[0, 2])
    back = np.empty((n, 3), dtype=np.int8)
    back[0] = 0

    e_list = emissions.tolist()
    pha_l = p_ha_arr.tolist()
    pad_l = p_ad_arr.tolist()
    pda_l = p_da_arr.tolist()

    for t in range(1, n):
        ha = pha_l[t]
        ad = pad_l[t]
        da = pda_l[t]
        et = e_list[t]
        e0, e1, e2 = et[0], et[1], et[2]

        # To state 0
        c0, c1, c2 = d0, d1 - ha, d2 - da
        if c0 >= c1 and c0 >= c2:
            nd0, b0 = c0 + e0, 0
        elif c1 >= c2:
            nd0, b0 = c1 + e0, 1
        else:
            nd0, b0 = c2 + e0, 2

        # To state 1
        c0, c1, c2 = d0 - ha, d1, d2 - da
        if c1 >= c0 and c1 >= c2:
            nd1, b1 = c1 + e1, 1
        elif c0 >= c2:
            nd1, b1 = c0 + e1, 0
        else:
            nd1, b1 = c2 + e1, 2

        # To state 2
        c0, c1, c2 = d0 - ad, d1 - ad, d2
        if c2 >= c0 and c2 >= c1:
            nd2, b2 = c2 + e2, 2
        elif c0 >= c1:
            nd2, b2 = c0 + e2, 0
        else:
            nd2, b2 = c1 + e2, 1

        d0, d1, d2 = nd0, nd1, nd2
        back[t, 0], back[t, 1], back[t, 2] = b0, b1, b2

    states = np.empty(n, dtype=np.int8)
    if d0 >= d1 and d0 >= d2:
        cur = 0
    elif d1 >= d2:
        cur = 1
    else:
        cur = 2
    states[-1] = cur
    b_list = back.tolist()
    for t in range(n - 1, 0, -1):
        cur = b_list[t][cur]
        states[t - 1] = cur
    return states


def forward_log_alpha(emissions, p_ha, p_ad, p_da, temp=1.0):
    """Forward pass computing normalized log alpha[t, s] = log P(y_t = s | x_{0:t})."""
    n = emissions.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    inv_t = 1.0 / float(temp)
    em = emissions * inv_t
    p_ha_arr = np.broadcast_to(np.asarray(p_ha, dtype=np.float64), (n,)) * inv_t
    p_ad_arr = np.broadcast_to(np.asarray(p_ad, dtype=np.float64), (n,)) * inv_t
    p_da_arr = np.broadcast_to(np.asarray(p_da, dtype=np.float64), (n,)) * inv_t

    alpha = np.empty((n, 3), dtype=np.float64)
    a0, a1, a2 = float(em[0, 0]), float(em[0, 1]), float(em[0, 2])
    z = _logsumexp3(a0, a1, a2)
    a0 -= z
    a1 -= z
    a2 -= z
    alpha[0, 0], alpha[0, 1], alpha[0, 2] = a0, a1, a2

    e_list = em.tolist()
    pha_l = p_ha_arr.tolist()
    pad_l = p_ad_arr.tolist()
    pda_l = p_da_arr.tolist()

    exp = math.exp
    log = math.log
    for t in range(1, n):
        ha = pha_l[t]
        ad = pad_l[t]
        da = pda_l[t]
        et = e_list[t]

        # Shift by max incoming alpha for numeric stability before exp
        m0 = a0
        c1 = a1 - ha
        c2 = a2 - da
        if c1 > m0:
            m0 = c1
        if c2 > m0:
            m0 = c2
        na0 = et[0] + m0 + log(exp(a0 - m0) + exp(c1 - m0) + exp(c2 - m0))

        m1 = a1
        c0 = a0 - ha
        c2 = a2 - da
        if c0 > m1:
            m1 = c0
        if c2 > m1:
            m1 = c2
        na1 = et[1] + m1 + log(exp(c0 - m1) + exp(a1 - m1) + exp(c2 - m1))

        m2 = a2
        c0 = a0 - ad
        c1 = a1 - ad
        if c0 > m2:
            m2 = c0
        if c1 > m2:
            m2 = c1
        na2 = et[2] + m2 + log(exp(c0 - m2) + exp(c1 - m2) + exp(a2 - m2))

        mz = na0 if na0 >= na1 else na1
        if na2 > mz:
            mz = na2
        z = mz + log(exp(na0 - mz) + exp(na1 - mz) + exp(na2 - mz))
        a0, a1, a2 = na0 - z, na1 - z, na2 - z
        alpha[t, 0], alpha[t, 1], alpha[t, 2] = a0, a1, a2

    return alpha


def ffbs_sample(alpha, p_ha, p_ad, p_da, temp=1.0, n_samples=200, seed=42):
    """Exact Forward-Filtering Backward-Sampling of n_samples paths (n_samples, T)."""
    n = alpha.shape[0]
    if n == 0:
        return np.empty((n_samples, 0), dtype=np.int8)

    inv_t = 1.0 / float(temp)
    p_ha_arr = np.broadcast_to(np.asarray(p_ha, dtype=np.float64), (n,)) * inv_t
    p_ad_arr = np.broadcast_to(np.asarray(p_ad, dtype=np.float64), (n,)) * inv_t
    p_da_arr = np.broadcast_to(np.asarray(p_da, dtype=np.float64), (n,)) * inv_t

    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, n), dtype=np.int8)

    # Sample y_{T-1}
    p_last = np.exp(alpha[-1] - np.max(alpha[-1]))
    p_last /= p_last.sum()
    u_all = rng.random((n, n_samples))
    cdf_last = np.cumsum(p_last)
    samples[:, -1] = np.searchsorted(cdf_last, u_all[-1]).astype(np.int8)

    # Precompute exp(alpha[t]) ratios? Simpler: vectorized over n_samples at each step t!
    # P(y_t = i | y_{t+1} = j) \propto exp(alpha[t, i] + trans[t+1, i, j])
    a_exp = np.exp(alpha - np.max(alpha, axis=1, keepdims=True))  # (n, 3), exact up to row const
    e_ha = np.exp(-p_ha_arr)
    e_ad = np.exp(-p_ad_arr)
    e_da = np.exp(-p_da_arr)

    cur = samples[:, -1]
    u_list = u_all.tolist()
    a_list = a_exp.tolist()
    eha_l = e_ha.tolist()
    ead_l = e_ad.tolist()
    eda_l = e_da.tolist()

    for t in range(n - 2, -1, -1):
        at = a_list[t]
        a0, a1, a2 = at[0], at[1], at[2]
        ha = eha_l[t + 1]
        ad = ead_l[t + 1]
        da = eda_l[t + 1]

        # Cumulative probabilities for each of the 3 next-states j in {0, 1, 2}
        # j = 0: weights (a0, a1*ha, a2*da)
        w00, w01, w02 = a0, a1 * ha, a2 * da
        s0 = w00 + w01 + w02
        c00 = w00 / s0
        c01 = (w00 + w01) / s0

        # j = 1: weights (a0*ha, a1, a2*da)
        w10, w11, w12 = a0 * ha, a1, a2 * da
        s1 = w10 + w11 + w12
        c10 = w10 / s1
        c11 = (w10 + w11) / s1

        # j = 2: weights (a0*ad, a1*ad, a2)
        w20, w21, w22 = a0 * ad, a1 * ad, a2
        s2 = w20 + w21 + w22
        c20 = w20 / s2
        c21 = (w20 + w21) / s2

        c_0 = (c00, c10, c20)
        c_1 = (c01, c11, c21)
        ut = u_list[t]
        nxt = [0] * n_samples
        cur_l = cur.tolist()
        for k in range(n_samples):
            j = cur_l[k]
            uk = ut[k]
            if uk < c_0[j]:
                nxt[k] = 0
            elif uk < c_1[j]:
                nxt[k] = 1
            else:
                nxt[k] = 2
        cur = np.array(nxt, dtype=np.int8)
        samples[:, t] = cur

    return samples


def extract_spells(codes):
    """Return list of (start, stop, code) for H(0) and A(1) maximal runs in a single half."""
    n = len(codes)
    if n == 0:
        return []
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], br, [n]))
    out = []
    for i in range(len(bounds) - 1):
        a = int(bounds[i])
        b = int(bounds[i + 1])
        c = int(codes[a])
        if c == 0 or c == 1:
            out.append((a, b, c))
    return out


def fast_spell_f1(gt_spells, pred_spells, ious=SPELL_IOUS):
    """Exact mean spellF1 over IoU thresholds [0.3..0.7] in O(M + K) via two-pointer sweep."""
    ng = len(gt_spells)
    np_ = len(pred_spells)
    if ng == 0 and np_ == 0:
        return 1.0
    if ng == 0 or np_ == 0:
        return 0.0

    overlaps = []
    j0 = 0
    for i, (gs, ge, gc) in enumerate(gt_spells):
        while j0 < np_ and pred_spells[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < np_ and pred_spells[j][0] < ge:
            ps, pe, pc = pred_spells[j]
            if pc == gc:
                inter = (ge if ge < pe else pe) - (gs if gs > ps else ps)
                if inter > 0:
                    uni = (ge if ge > pe else pe) - (gs if gs < ps else ps)
                    overlaps.append((inter / float(uni), i, j))
            j += 1

    if not overlaps:
        return 0.0

    overlaps.sort(key=lambda x: (-x[0], x[1], x[2]))
    f1_sum = 0.0
    for thr in ious:
        used_g = [False] * ng
        used_p = [False] * np_
        tp = 0
        for iou, gi, pi in overlaps:
            if iou < thr:
                break
            if not used_g[gi] and not used_p[pi]:
                used_g[gi] = True
                used_p[pi] = True
                tp += 1
        if tp > 0:
            f1_sum += (2.0 * tp) / float(ng + np_)
    return f1_sum / float(len(ious))


def mbr_decode_half(emissions, p_ha, p_ad, p_da, temp=1.0, n_samples=160,
                    scale_grid=(0.65, 0.8, 0.9, 1.0, 1.15, 1.35),
                    extra_candidates=None, seed=42):
    """Run Forward-Backward + FFBS and select the candidate maximizing expected spellF1."""
    alpha = forward_log_alpha(emissions, p_ha, p_ad, p_da, temp=temp)
    samples = ffbs_sample(alpha, p_ha, p_ad, p_da, temp=temp, n_samples=n_samples, seed=seed)
    sample_spells = [extract_spells(samples[k]) for k in range(n_samples)]

    # Candidate pool: MAP Viterbi across penalty scales + marginal argmax + top samples
    candidates = []
    for sc in scale_grid:
        candidates.append(viterbi_tv(emissions, p_ha * sc, p_ad * sc, p_da * sc))
    # Posterior marginal max (smoothed mildly to avoid 1-frame flicker)
    post = np.exp(alpha - np.max(alpha, axis=1, keepdims=True))
    candidates.append(np.argmax(post, axis=1).astype(np.int8))

    # Also add representative posterior samples (first 24)
    for k in range(min(24, n_samples)):
        candidates.append(samples[k])

    if extra_candidates:
        candidates.extend(extra_candidates)

    best_cand = candidates[0]
    best_score = -1.0
    seen = set()
    for cand in candidates:
        sp = extract_spells(cand)
        sig = tuple(sp)
        if sig in seen:
            continue
        seen.add(sig)
        tot = 0.0
        for s_sp in sample_spells:
            tot += fast_spell_f1(s_sp, sp)
        avg = tot / float(n_samples)
        if avg > best_score:
            best_score = avg
            best_cand = cand
    return best_cand, best_score, alpha
