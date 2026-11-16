"""Fast exact PELT change-point detection on collective player kinematics (pure NumPy)."""

import numpy as np


def pelt_l2_blocks(X, penalty=28.0, min_seg=5):
    """Exact PELT for multivariate piecewise-constant L2 cost on (M, D) array.

    Returns sorted interior changepoint indices in [0..M].
    """
    m, d = X.shape
    if m < 2 * min_seg:
        return np.empty(0, dtype=np.int64)

    S1 = np.vstack([np.zeros((1, d), dtype=np.float64), np.cumsum(X, axis=0, dtype=np.float64)])
    S2s = np.concatenate(([0.0], np.cumsum(np.sum(X * X, axis=1, dtype=np.float64))))

    F = np.full(m + 1, np.inf, dtype=np.float64)
    F[0] = -penalty
    prev = np.zeros(m + 1, dtype=np.int64)
    R = np.array([0], dtype=np.int64)

    for t in range(min_seg, m + 1):
        valid_mask = (t - R) >= min_seg
        Rc = R[valid_mask]
        if len(Rc) > 0:
            lens = (t - Rc).astype(np.float64)
            diff1 = S1[t] - S1[Rc]
            seg_cost = (S2s[t] - S2s[Rc]) - np.sum(diff1 * diff1, axis=1) / lens
            cand = F[Rc] + seg_cost + penalty
            best_idx = int(np.argmin(cand))
            F[t] = cand[best_idx]
            prev[t] = Rc[best_idx]
            keep = (cand - penalty) <= F[t] + 1e-12
            R = np.concatenate((Rc[keep], R[~valid_mask]))
        add_pt = t - min_seg + 1
        if add_pt >= min_seg:
            R = np.append(R, add_pt)

    cps = []
    cur = m
    while cur > 0:
        p = int(prev[cur])
        if p > 0:
            cps.append(p)
        cur = p
    cps.reverse()
    return np.array(cps, dtype=np.int64)


def compute_pelt_regime_features(signals, half_blocks, stride=5, penalties=(28.0, 65.0), min_seg_frames=20):
    """Segment collective player signals per half via exact PELT and extract per-frame regime features.

    Args:
        signals: (n, d) float64 array [mps, p90, walk, sprint], NaN-free.
        half_blocks: [(half, start, end), ...]
    Returns:
        dict of per-frame float64 feature arrays of length n.
    """
    n, d = signals.shape
    min_seg_b = max(2, min_seg_frames // stride)

    out = {}
    for tag, pen in zip(("pf", "pm"), penalties):
        seg_means = np.zeros((n, d), dtype=np.float64)
        seg_len = np.zeros(n, dtype=np.float64)
        seg_pos = np.zeros(n, dtype=np.float64)
        seg_rem = np.zeros(n, dtype=np.float64)
        step_prev = np.zeros(n, dtype=np.float64)  # mps jump from previous regime
        step_next = np.zeros(n, dtype=np.float64)  # mps jump to next regime

        for _, s, e in half_blocks:
            e1 = e + 1
            sub = signals[s:e1]
            L = len(sub)
            if L < 2 * stride * min_seg_b:
                seg_means[s:e1] = sub.mean(axis=0)
                seg_len[s:e1] = np.log1p(L)
                continue

            # Standardize within the half so PELT penalty is match-scale invariant
            mu = sub.mean(axis=0)
            sd = np.maximum(sub.std(axis=0), 1e-4)
            Z = (sub - mu) / sd
            M = L // stride
            Xb = Z[:M * stride].reshape(M, stride, d).mean(axis=1)
            cps = pelt_l2_blocks(Xb, penalty=pen, min_seg=min_seg_b)

            # Convert block changepoints back to frame coordinates + local 1-frame refinement
            bounds = [0]
            S1z = np.vstack([np.zeros((1, d)), np.cumsum(Z, axis=0)])
            for c in cps:
                cf = int(c) * stride
                bounds.append(cf)
            bounds.append(L)

            # Fine 1-frame shift (+/- stride) maximizing between-segment L2 gap around each interior cp
            for i in range(1, len(bounds) - 1):
                a_prev, c0, b_next = bounds[i - 1], bounds[i], bounds[i + 1]
                lo = max(a_prev + min_seg_frames // 2, c0 - stride)
                hi = min(b_next - min_seg_frames // 2, c0 + stride)
                if hi > lo:
                    best_c, best_gain = c0, -1.0
                    for cc in range(lo, hi + 1):
                        m1 = (S1z[cc] - S1z[a_prev]) / (cc - a_prev)
                        m2 = (S1z[b_next] - S1z[cc]) / (b_next - cc)
                        diff = m1 - m2
                        g = ((cc - a_prev) * (b_next - cc) / float(b_next - a_prev)) * float(np.dot(diff, diff))
                        if g > best_gain:
                            best_gain = g
                            best_c = cc
                    bounds[i] = best_c

            n_seg = len(bounds) - 1
            s_means = np.empty((n_seg, d), dtype=np.float64)
            for i in range(n_seg):
                a, b = bounds[i], bounds[i + 1]
                s_means[i] = sub[a:b].mean(axis=0)

            for i in range(n_seg):
                a, b = bounds[i], bounds[i + 1]
                ga, gb = s + a, s + b
                seg_means[ga:gb] = s_means[i]
                seg_len[ga:gb] = np.log1p(b - a)
                idx = np.arange(b - a, dtype=np.float64)
                seg_pos[ga:gb] = np.log1p(idx)
                seg_rem[ga:gb] = np.log1p((b - a - 1) - idx)
                if i > 0:
                    step_prev[ga:gb] = s_means[i, 0] - s_means[i - 1, 0]
                if i + 1 < n_seg:
                    step_next[ga:gb] = s_means[i + 1, 0] - s_means[i, 0]

        for k, col_nm in enumerate(("mps", "p90", "walk", "sprint")):
            out[f"{tag}_{col_nm}"] = seg_means[:, k]
        out[f"{tag}_len"] = seg_len
        out[f"{tag}_pos"] = seg_pos
        out[f"{tag}_rem"] = seg_rem
        out[f"{tag}_step_prev"] = step_prev
        out[f"{tag}_step_next"] = step_next
        out[f"{tag}_edge_min"] = np.minimum(seg_pos, seg_rem)

    return out
