"""Sprint 3: Time-varying 3-state Viterbi sequence decoder and post-processing."""

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])


def viterbi_3state_tv(emissions, mult_to_h, mult_to_a, p_ha=54.0, p_ad=34.0, p_da=22.0):
    """Exact 3-state max-sum Viterbi with time-varying outcome-gated H<->A penalties."""
    n_t = emissions.shape[0]
    if n_t == 0:
        return np.empty(0, dtype=np.int8)

    dp = np.empty(3, dtype=np.float64)
    dp[:] = emissions[0]
    back = np.empty((n_t, 3), dtype=np.int8)
    back[0] = 0

    for t in range(1, n_t):
        p_a2h = p_ha * float(mult_to_h[t])
        p_h2a = p_ha * float(mult_to_a[t])
        d0, d1, d2 = float(dp[0]), float(dp[1]), float(dp[2])
        e0, e1, e2 = float(emissions[t, 0]), float(emissions[t, 1]), float(emissions[t, 2])

        c0, c1, c2 = d0, d1 - p_a2h, d2 - p_da
        if c0 >= c1 and c0 >= c2:
            nd0, b0 = c0 + e0, 0
        elif c1 >= c2:
            nd0, b0 = c1 + e0, 1
        else:
            nd0, b0 = c2 + e0, 2

        c0, c1, c2 = d0 - p_h2a, d1, d2 - p_da
        if c1 >= c0 and c1 >= c2:
            nd1, b1 = c1 + e1, 1
        elif c0 >= c2:
            nd1, b1 = c0 + e1, 0
        else:
            nd1, b1 = c2 + e1, 2

        c0, c1, c2 = d0 - p_ad, d1 - p_ad, d2
        if c2 >= c0 and c2 >= c1:
            nd2, b2 = c2 + e2, 2
        elif c0 >= c1:
            nd2, b2 = c0 + e2, 0
        else:
            nd2, b2 = c1 + e2, 1

        dp[0], dp[1], dp[2] = nd0, nd1, nd2
        back[t, 0], back[t, 1], back[t, 2] = b0, b1, b2

    states = np.empty(n_t, dtype=np.int8)
    cur = int(np.argmax(dp))
    states[-1] = cur
    for t in range(n_t - 1, 0, -1):
        cur = int(back[t, cur])
        states[t - 1] = cur
    return states


def cleanup_short_segments(codes, oob_slice, vis_slice, sp_slice,
                           min_dead_len=35, min_inbounds_dead_len=105, min_poss_len=18):
    """Remove physically implausible micro-stoppages and micro-possessions."""
    out = codes.copy()
    n = len(out)
    if n == 0:
        return out

    changed = True
    while changed:
        changed = False
        breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [n]))
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            st = int(out[a])
            dur = b - a
            if st == D and len(bounds) > 2:
                is_fake_inbounds_pause = (
                    dur < min_inbounds_dead_len
                    and float(np.mean(vis_slice[a:b])) > 0.85
                    and float(np.mean(oob_slice[a:b])) < 0.05
                    and not np.any(sp_slice[a:b])
                )
                if dur < min_dead_len or is_fake_inbounds_pause:
                    if i > 0:
                        out[a:b] = out[a - 1]
                    elif i + 1 < len(bounds) - 1:
                        out[a:b] = out[b]
                    changed = True
                    break

    changed = True
    while changed:
        changed = False
        breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [n]))
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            st = int(out[a])
            if st in (H, A) and (b - a) < min_poss_len and len(bounds) > 2:
                if i > 0:
                    out[a:b] = out[a - 1]
                elif i + 1 < len(bounds) - 1:
                    out[a:b] = out[b]
                changed = True
                break

    return out


def snap_transitions(codes, dv, d_h, d_a, oob, start_idx, end_idx, min_seg_len=18):
    """Align Viterbi H<->A and H/A->D(OOB) transition boundaries to physical touch/OOB peaks.

    Preserves segment sequence and enforces a minimum segment length guard so neighboring
    boundaries never collide or create micro-segments.
    """
    out = codes.copy()
    seg = out[start_idx:end_idx]
    if len(seg) == 0:
        return out

    breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(seg)]))

    last_cut = start_idx
    for j in range(1, len(bounds) - 1):
        t_vit = start_idx + int(bounds[j])
        next_start = start_idx + int(bounds[j + 1])
        prev_st = int(out[t_vit - 1])
        next_st = int(out[t_vit])

        w0 = max(last_cut + min_seg_len, t_vit - 8)
        w1 = min(next_start - min_seg_len, t_vit + 16)
        if w1 <= w0 + 2:
            last_cut = t_vit
            continue

        best_t = t_vit
        if prev_st in (H, A) and next_st in (H, A):
            d_next = d_h[w0:w1] if next_st == H else d_a[w0:w1]
            d_prev = d_a[w0:w1] if next_st == H else d_h[w0:w1]
            if np.max(dv[w0:w1]) > 2.5 or np.min(d_next) < 2.2:
                sc = ((dv[w0:w1] / 4.0)
                      + np.exp(-np.nan_to_num(d_next, nan=99.0) / 1.5)
                      - 0.5 * np.exp(-np.nan_to_num(d_prev, nan=99.0) / 1.5))
                best_t = w0 + int(np.argmax(sc))
        elif prev_st in (H, A) and next_st == D:
            cand_oob = np.flatnonzero(oob[w0:w1])
            if len(cand_oob):
                best_t = w0 + int(cand_oob[0])

        if best_t > t_vit:
            out[t_vit:best_t] = prev_st
        elif best_t < t_vit:
            out[best_t:t_vit] = next_st
        last_cut = best_t

    return out


def filter_fast_sandwiches(codes, b_spd, dtheta, start_idx, end_idx,
                           max_len=55, min_med_spd=7.0, max_entry_angle=30.0):
    """Merge brief mid-flight sandwiched turnovers (H-A-H / A-H-A) without ball deflection."""
    out = codes.copy()
    changed = True
    n_total = len(out)
    while changed:
        changed = False
        seg = out[start_idx:end_idx]
        if len(seg) == 0:
            break
        breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(seg)]))
        for j in range(1, len(bounds) - 1):
            a = start_idx + int(bounds[j])
            b = start_idx + int(bounds[j + 1])
            st = int(out[a])
            if st in (H, A) and (b - a) < max_len and out[a - 1] == out[b] and out[a - 1] in (H, A):
                med_bs = float(np.median(b_spd[a:b]))
                entry_ang = float(np.max(dtheta[max(0, a - 3):min(n_total, a + 4)]))
                if med_bs > min_med_spd and entry_ang < max_entry_angle:
                    out[a:b] = out[a - 1]
                    changed = True
                    break
    return out


def decode_match(data, features, p_ha=54.0, p_ad=34.0, p_da=22.0):
    """Decode full 199,319 frames into H/A/D/X label array."""
    n = data["n_frames"]
    team_score = features["team_score"]
    dead_score = features["dead_score"]
    mult_to_h = features["mult_to_H"]
    mult_to_a = features["mult_to_A"]
    dv = features["dv"]
    dtheta = features["dtheta"]
    b_spd = features["b_spd"]
    d_h = features["d_H"]
    d_a = features["d_A"]
    half_blocks = data["half_blocks"]
    ko2 = features["ko2"]

    live_conf = np.clip(1.0 - np.maximum(0.0, dead_score) / 2.2, 0.05, 1.0)
    eff_team = team_score * live_conf

    emissions = np.empty((n, 3), dtype=np.float64)
    emissions[:, H] = eff_team
    emissions[:, A] = -eff_team
    emissions[:, D] = dead_score

    codes = np.full(n, X, dtype=np.int8)

    oob = features["oob"]
    vis = features["vis_clean"]
    sp_mask = features["sp_restart_mask"]

    _, h1_s, h1_e = half_blocks[0]
    h1_seq = viterbi_3state_tv(
        emissions[h1_s:h1_e + 1], mult_to_h[h1_s:h1_e + 1], mult_to_a[h1_s:h1_e + 1],
        p_ha=p_ha, p_ad=p_ad, p_da=p_da,
    )
    codes[h1_s:h1_e + 1] = cleanup_short_segments(
        h1_seq, oob[h1_s:h1_e + 1], vis[h1_s:h1_e + 1], sp_mask[h1_s:h1_e + 1]
    )
    codes = snap_transitions(codes, dv, d_h, d_a, oob, h1_s, h1_e + 1)
    codes = filter_fast_sandwiches(codes, b_spd, dtheta, h1_s, h1_e + 1)

    _, h2_s, h2_e = half_blocks[1]
    codes[h2_s:ko2] = D
    h2_seq = viterbi_3state_tv(
        emissions[ko2:h2_e + 1], mult_to_h[ko2:h2_e + 1], mult_to_a[ko2:h2_e + 1],
        p_ha=p_ha, p_ad=p_ad, p_da=p_da,
    )
    codes[ko2:h2_e + 1] = cleanup_short_segments(
        h2_seq, oob[ko2:h2_e + 1], vis[ko2:h2_e + 1], sp_mask[ko2:h2_e + 1]
    )
    codes = snap_transitions(codes, dv, d_h, d_a, oob, ko2, h2_e + 1)
    codes = filter_fast_sandwiches(codes, b_spd, dtheta, ko2, h2_e + 1)

    return LABEL_CHARS[codes], codes
