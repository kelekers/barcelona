"""Sprint 2/3: Ball reconstruction, kinematics, dead-ball detection, and outcome-gated team features.

All thresholds and weights are derived from football physics (metres, m/s, seconds)
and full-match unlabeled structure -- never fitted against the public 5-minute GT.
"""

import numpy as np

from games import GAMES, anchor_video_frame

HOME, AWAY = 0, 1


def rolling_sum_1d(arr, half_win):
    """Centered rolling sum over [t - half_win, t + half_win] in O(N)."""
    return window_sum_1d(arr, half_win, half_win)


def window_sum_1d(arr, back_win, fwd_win):
    """Asymmetric rolling sum over [t - back_win, t + fwd_win] in O(N)."""
    n = len(arr)
    cs = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - back_win)
    hi = np.minimum(n, idx + fwd_win + 1)
    return cs[hi] - cs[lo]


def rolling_nanmean_1d(arr, half_win):
    """Centered rolling mean ignoring NaNs, returning NaN only if window is empty."""
    valid = ~np.isnan(arr)
    clean = np.where(valid, arr, 0.0)
    num = rolling_sum_1d(clean, half_win)
    den = rolling_sum_1d(valid.astype(np.float64), half_win)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def interpolate_short_gaps(bx, by, max_gap=35):
    """Linearly bridge short ball NaN gaps (<= max_gap frames) inside valid trajectories."""
    bx_out = bx.copy()
    by_out = by.copy()
    n = len(bx)
    vis = ~np.isnan(bx)
    miss = ~vis
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    for a, b in zip(starts, ends):
        if a == 0 or b == n:
            continue
        gap = b - a
        if gap <= max_gap:
            w = np.linspace(0.0, 1.0, gap + 2, dtype=np.float32)[1:-1]
            bx_out[a:b] = bx[a - 1] * (1.0 - w) + bx[b] * w
            by_out[a:b] = by[a - 1] * (1.0 - w) + by[b] * w
    return bx_out, by_out


def compute_ball_speed(bx, by, pitch, fps):
    """Centered finite-difference ball speed (m/s) with 5-frame span fallback to 3-frame."""
    length, width = pitch
    n = len(bx)
    bx_m = bx.astype(np.float64) * length
    by_m = by.astype(np.float64) * width
    spd = np.full(n, np.nan, dtype=np.float64)

    if n > 4:
        ok4 = (~np.isnan(bx_m[4:])) & (~np.isnan(bx_m[:-4]))
        vx4 = (bx_m[4:] - bx_m[:-4]) * (fps / 4.0)
        vy4 = (by_m[4:] - by_m[:-4]) * (fps / 4.0)
        s4 = np.hypot(vx4, vy4)
        spd[2:-2][ok4] = s4[ok4]

    if n > 2:
        ok2 = np.isnan(spd[1:-1]) & (~np.isnan(bx_m[2:])) & (~np.isnan(bx_m[:-2]))
        vx2 = (bx_m[2:] - bx_m[:-2]) * (fps / 2.0)
        vy2 = (by_m[2:] - by_m[:-2]) * (fps / 2.0)
        s2 = np.hypot(vx2, vy2)
        spd[1:-1][ok2] = s2[ok2]

    return np.clip(spd, 0.0, 45.0)


def compute_touch_impulses(bx_i, by_i, d_H, d_A, dead_score, pitch, fps):
    """Detect ball velocity impulses and validate 1-touch clearances / receptions via outcome gating."""
    n = len(bx_i)
    length, width = pitch
    bx_m = bx_i.astype(np.float64) * length
    by_m = by_i.astype(np.float64) * width

    vx_pre = np.full(n, np.nan, dtype=np.float64)
    vy_pre = np.full(n, np.nan, dtype=np.float64)
    vx_post = np.full(n, np.nan, dtype=np.float64)
    vy_post = np.full(n, np.nan, dtype=np.float64)

    if n > 3:
        ok3 = (~np.isnan(bx_m[3:])) & (~np.isnan(bx_m[:-3]))
        vx_pre[3:][ok3] = (bx_m[3:][ok3] - bx_m[:-3][ok3]) * (fps / 3.0)
        vy_pre[3:][ok3] = (by_m[3:][ok3] - by_m[:-3][ok3]) * (fps / 3.0)
        vx_post[:-3][ok3] = (bx_m[3:][ok3] - bx_m[:-3][ok3]) * (fps / 3.0)
        vy_post[:-3][ok3] = (by_m[3:][ok3] - by_m[:-3][ok3]) * (fps / 3.0)

    valid_v = (~np.isnan(vx_pre)) & (~np.isnan(vx_post))
    dv = np.zeros(n, dtype=np.float64)
    dv[valid_v] = np.hypot(vx_post[valid_v] - vx_pre[valid_v],
                           vy_post[valid_v] - vy_pre[valid_v])

    min_d = np.minimum(d_H, d_A)
    touch_any = rolling_sum_1d((valid_v & (dv > 3.5) & (min_d < 2.8)).astype(np.float64), 2) > 0

    def _outcome_gate(d_own, d_opp):
        cand = valid_v & (dv > 3.5) & (d_own < 2.3) & (d_own + 0.2 < d_opp)
        gate = np.zeros(n, dtype=bool)
        for t in np.flatnonzero(cand):
            t_s, t_end = min(n, t + 5), min(n, t + 65)
            if t_s >= t_end:
                continue
            own_later = np.sum((d_own[t_s:t_end] < 2.3) & (d_own[t_s:t_end] + 0.2 < d_opp[t_s:t_end])) >= 6
            disp = 0.0
            if not np.isnan(bx_m[t]):
                vf = ~np.isnan(bx_m[t_s:t_end])
                if np.any(vf):
                    disp = float(np.max(np.hypot(bx_m[t_s:t_end][vf] - bx_m[t],
                                                 by_m[t_s:t_end][vf] - by_m[t])))
            if own_later or disp > 12.0 or float(np.mean(dead_score[t_s:t_end] > 1.0)) > 0.4:
                gate[max(0, t - 2):min(n, t + 3)] = True
        return gate

    gate_H = _outcome_gate(d_H, d_A)
    gate_A = _outcome_gate(d_A, d_H)
    return touch_any, gate_H, gate_A, dv


def compute_player_activity(ps):
    """Per-frame collective player speed summary (mean and p90) without empty-slice warnings."""
    n = ps.shape[0]
    valid = ~np.isnan(ps)
    cnt = valid.sum(axis=1)
    safe = np.where(valid, ps, 0.0)
    mean_ps = np.full(n, np.nan, dtype=np.float64)
    has_any = cnt > 0
    mean_ps[has_any] = safe[has_any].sum(axis=1) / cnt[has_any]

    sorted_ps = np.sort(np.where(valid, ps, np.inf), axis=1)
    p90_idx = np.maximum(0, np.floor(0.9 * (cnt - 1)).astype(np.int64))
    p90_ps = np.full(n, np.nan, dtype=np.float64)
    rows = np.flatnonzero(has_any)
    p90_ps[rows] = sorted_ps[rows, p90_idx[rows]]

    return mean_ps, p90_ps, cnt


def _forward_fill_nan(arr, fill_default=0.0):
    """Linearly interpolate internal NaNs of a 1D signal for smooth rolling summaries."""
    n = len(arr)
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.full(n, fill_default, dtype=np.float64)
    idx = np.arange(n)
    return np.interp(idx, idx[valid], arr[valid])


def detect_goal_celebrations(events, bx_clean, mps_fast, p90_fast, n_frames, fps):
    """Find post-goal dead-ball celebration windows and exact Kick Off (After Goal) frames."""
    goals = [ev for ev in events if ev["code"] == "GOALS"]
    kickoffs = [ev for ev in events
                if ev["code"] == "SET PIECES"
                and "kick off (after goal)" in (ev["tags"].get("Type") or "").lower()]

    dead_windows = []
    used_ko = set()

    for g in goals:
        ga, gb = g["start_frame"], g["end_frame"]
        scoring_team_str = (g["tags"].get("Team") or "").strip().lower()
        if "barcelona" in scoring_team_str:
            conceding_team = AWAY
        elif "atletic" in scoring_team_str or "athletic" in scoring_team_str:
            conceding_team = HOME
        else:
            conceding_team = None

        best_ko = None
        best_dist = 999999
        for i, ko in enumerate(kickoffs):
            if i in used_ko:
                continue
            d = abs(ko["start_frame"] - gb)
            if d < best_dist and d < 10 * fps:
                best_dist = d
                best_ko = (i, ko)
        if best_ko is None:
            continue
        used_ko.add(best_ko[0])
        ko_ev = best_ko[1]
        ko_a = ko_ev["start_frame"]

        t_ko = min(n_frames - 1, ko_a + int(round(3.32 * fps)))

        stop_start = max(ga, t_ko - 50)
        consecutive_active = 0
        for t in range(t_ko - 1, max(0, ga - 50) - 1, -1):
            ball_here = not np.isnan(bx_clean[t])
            active = (mps_fast[t] > 1.65) or (p90_fast[t] > 2.85 and ball_here)
            if active and ball_here:
                consecutive_active += 1
                if consecutive_active >= 12:
                    stop_start = t + 12
                    break
            else:
                consecutive_active = 0
                stop_start = t

        if t_ko - stop_start >= 5 * fps:
            dead_windows.append((stop_start, t_ko, conceding_team))

    return dead_windows


def find_set_piece_restarts(events, bx_clean, by_clean, b_spd, mps_fast, oob, px, py,
                            team_map, pitch, n_frames, fps, goal_dead_windows):
    """Pinpoint the exact kick/throw restart frame and pre-restart dead span for SET PIECES."""
    length, width = pitch
    restarts = []

    in_goal_celeb = np.zeros(n_frames, dtype=bool)
    for a, b, _ in goal_dead_windows:
        in_goal_celeb[max(0, a):min(n_frames, b)] = True

    vis_all = ~np.isnan(bx_clean)
    spd_all = np.nan_to_num(b_spd, nan=0.0)

    sp_events = [ev for ev in events if ev["code"] == "SET PIECES"]
    for ev in sp_events:
        sp_type = (ev["tags"].get("Type") or "").strip()
        if "kick off" in sp_type.lower():
            continue

        sp_a = ev["start_frame"]
        if in_goal_celeb[min(n_frames - 1, sp_a + 25)]:
            continue

        tm_str = (ev["tags"].get("Team") or "").strip().lower()
        if "barcelona" in tm_str:
            sp_team = HOME
        elif "atletic" in tm_str or "athletic" in tm_str:
            sp_team = AWAY
        else:
            sp_team = None

        w0 = max(2, sp_a - 2 * fps)
        w1 = min(n_frames - 30, sp_a + 14 * fps)
        if w1 <= w0 + 10:
            continue

        best_t = None
        best_score = -1e9
        for t in range(w0, w1):
            pre0 = max(0, t - 45)
            post1 = min(n_frames, t + 30)

            pre_dead = (~vis_all[pre0:t]) | oob[pre0:t] | (spd_all[pre0:t] < 1.8)
            pre_dead_frac = float(np.mean(pre_dead)) if t > pre0 else 0.0

            imm1 = min(n_frames, t + 4)
            imm_in = vis_all[t:imm1] & (~oob[t:imm1])
            if np.mean(imm_in) < 0.75:
                continue

            spd_at_t = float(np.mean(spd_all[t:imm1]))
            pre_imm_spd = float(np.mean(spd_all[max(0, t - 6):t])) if t > 0 else 0.0
            if spd_at_t < 3.4:
                t15 = min(n_frames - 1, t + 15)
                if np.isnan(bx_clean[t]) or np.isnan(bx_clean[t15]):
                    continue
                if np.hypot((bx_clean[t15] - bx_clean[t]) * length,
                            (by_clean[t15] - by_clean[t]) * width) < 2.5:
                    continue

            post_live = vis_all[t:post1] & (~oob[t:post1]) & (spd_all[t:post1] > 3.2)
            post_live_frac = float(np.mean(post_live))

            accel_step = np.clip((spd_at_t - pre_imm_spd) / 5.0, 0.0, 1.0)
            if pre_dead_frac >= 0.65 and post_live_frac >= 0.35:
                sc = pre_dead_frac + 1.3 * post_live_frac + 0.5 * accel_step + 0.08 * min(spd_at_t, 12.0)
                if sc > best_score:
                    best_score = sc
                    best_t = t

        if best_t is None:
            for t in range(w0, w1, 2):
                pre0 = max(0, t - 45)
                post1 = min(n_frames, t + 30)
                pre_dead_frac = float(np.mean((~vis_all[pre0:t]) | oob[pre0:t] | (spd_all[pre0:t] < 1.8)))
                post_live_frac = float(np.mean(vis_all[t:post1] & (~oob[t:post1]) & (spd_all[t:post1] > 2.8)))
                sc = pre_dead_frac + post_live_frac
                if sc > best_score:
                    best_score = sc
                    best_t = t

        if best_t is None:
            continue

        if sp_team is None and "throw" in sp_type.lower():
            h_votes = 0.0
            a_votes = 0.0
            r0 = max(0, best_t - 75)
            r1 = min(n_frames, best_t + 30)
            for tt in range(r0, r1, 2):
                side_mask = (py[tt] < 0.035) | (py[tt] > 0.965)
                if np.any(side_mask):
                    if vis_all[tt]:
                        d = np.hypot((px[tt] - bx_clean[tt]) * length, (py[tt] - by_clean[tt]) * width)
                        d = np.where(np.isnan(d), np.inf, d)
                        idx_min = int(np.argmin(d))
                        if d[idx_min] < 6.0:
                            if team_map[idx_min] == HOME:
                                h_votes += 2.0
                            else:
                                a_votes += 2.0
                    h_votes += 0.3 * float(np.sum(side_mask & (team_map == HOME)))
                    a_votes += 0.3 * float(np.sum(side_mask & (team_map == AWAY)))
                elif vis_all[tt]:
                    d = np.hypot((px[tt] - bx_clean[tt]) * length, (py[tt] - by_clean[tt]) * width)
                    d = np.where(np.isnan(d), np.inf, d)
                    idx_min = int(np.argmin(d))
                    if d[idx_min] < 4.0:
                        if team_map[idx_min] == HOME:
                            h_votes += 1.0
                        else:
                            a_votes += 1.0
            if abs(h_votes - a_votes) > 1.0:
                sp_team = HOME if h_votes > a_votes else AWAY

        back_limit = max(0, best_t - 15 * fps)
        dead_hint_start = best_t

        active_streak = 0
        for t in range(best_t - 2, back_limit - 1, -1):
            is_dead_setup = (not vis_all[t]) or oob[t] or (spd_all[t] < 1.5 and mps_fast[t] < 1.45)
            if is_dead_setup:
                dead_hint_start = t
                active_streak = 0
            else:
                active_streak += 1
                if active_streak >= 8:
                    break

        oob_idxs = np.flatnonzero(oob[max(0, best_t - 14 * fps):best_t])
        if len(oob_idxs) >= 4:
            first_oob = max(0, best_t - 14 * fps) + int(oob_idxs[0])
            dead_hint_start = min(dead_hint_start, first_oob)

        restarts.append((dead_hint_start, best_t, sp_team, sp_type))

    return restarts


def build_frame_evidence(data):
    """Compute calibrated per-frame evidence and time-varying transition multipliers."""
    n = data["n_frames"]
    fps = data["fps"]
    pitch = data["pitch"]
    length, width = pitch
    team_map = data["team_map"]
    bx_clean = data["ball_x"]
    by_clean = data["ball_y"]
    px = data["px"]
    py = data["py"]
    ps = data["ps"]
    events = data["events"]
    half_blocks = data["half_blocks"]

    # 1. Interpolate short live-play gaps (<= 1.4 s)
    bx_i, by_i = interpolate_short_gaps(bx_clean, by_clean, max_gap=35)
    b_spd = compute_ball_speed(bx_i, by_i, pitch, fps)
    b_spd_filled = np.nan_to_num(b_spd, nan=0.0)

    # 2. Collective player activity
    mean_ps, p90_ps, _ = compute_player_activity(ps)
    mps_fill = _forward_fill_nan(mean_ps, fill_default=1.5)
    p90_fill = _forward_fill_nan(p90_ps, fill_default=2.3)

    mps_fast = rolling_nanmean_1d(mps_fill, half_win=12)   # +/-0.48s
    mps_slow = rolling_nanmean_1d(mps_fill, half_win=30)   # +/-1.20s
    p90_fast = rolling_nanmean_1d(p90_fill, half_win=12)
    p90_slow = rolling_nanmean_1d(p90_fill, half_win=30)

    mps_ahead = np.empty(n, dtype=np.float64)
    mps_ahead[:-25] = mps_fast[25:]
    mps_ahead[-25:] = mps_fast[-1]

    # 3. Distance from ball to nearest Home and Away player
    vis_i = ~np.isnan(bx_i)
    vis_clean = ~np.isnan(bx_clean)

    dx = (px - bx_i[:, None]) * length
    dy = (py - by_i[:, None]) * width
    dist = np.hypot(dx, dy)
    dist_safe = np.where(np.isnan(dist), np.inf, dist)

    h_mask = team_map == HOME
    a_mask = team_map == AWAY
    d_H = np.min(dist_safe[:, h_mask], axis=1)
    d_A = np.min(dist_safe[:, a_mask], axis=1)
    min_d = np.minimum(d_H, d_A)
    opp_d = np.maximum(d_H, d_A)

    # 4. Calibrated team evidence team_score in roughly [-2.1, +2.1] (>0 = Home, <0 = Away)
    has_both = vis_i & np.isfinite(d_H) & np.isfinite(d_A)
    d_diff = np.zeros(n, dtype=np.float64)
    d_diff[has_both] = np.clip((d_A[has_both] - d_H[has_both]) / 1.4, -2.5, 2.5)

    w_prox = np.where(has_both, np.exp(-min_d / 1.7), 0.0)
    speed_thresh = np.where(min_d <= 1.6, 9.5, 6.0)
    w_speed = 1.0 / (1.0 + np.maximum(0.0, (b_spd_filled - speed_thresh) / 4.5) ** 2)

    opp_d_safe = np.where(has_both, opp_d, 10.0)
    abs_diff_safe = np.where(has_both, np.abs(np.where(has_both, d_A, 0.0) - np.where(has_both, d_H, 0.0)), 10.0)
    w_contested = np.where(
        has_both,
        1.0 - 0.65 * np.exp(-np.maximum(0.0, opp_d_safe - 1.5) / 1.0)
                   * np.exp(-abs_diff_safe / 1.2),
        1.0,
    )
    raw_team = 1.8 * d_diff * w_prox * w_speed * w_contested

    close_H = has_both & (d_H < 2.1) & (d_H + 0.35 < d_A) & (b_spd_filled < 9.5)
    close_A = has_both & (d_A < 2.1) & (d_A + 0.35 < d_H) & (b_spd_filled < 9.5)
    ctrl_H = window_sum_1d(close_H.astype(np.float64), back_win=14, fwd_win=1)
    ctrl_A = window_sum_1d(close_A.astype(np.float64), back_win=14, fwd_win=1)
    ctrl_bonus = np.clip((ctrl_H - ctrl_A) / 5.0, -1.8, 1.8) * w_contested

    team_score = np.tanh((raw_team + 0.85 * ctrl_bonus) / 2.2) * 2.1

    # 5. Smart Tagging Offensive Phase Priors
    off_codes = {
        "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
        "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL",
    }
    h_tag_votes = np.zeros(n, dtype=np.float64)
    a_tag_votes = np.zeros(n, dtype=np.float64)
    in_buildup_or_prog = np.zeros(n, dtype=bool)
    for ev in events:
        if ev["code"] in off_codes:
            tm = (ev["tags"].get("Team") or "").strip().lower()
            s = max(0, ev["start_frame"])
            e = min(n, ev["end_frame"])
            if "barcelona" in tm:
                h_tag_votes[s:e] += 1.0
            elif "atletic" in tm or "athletic" in tm:
                a_tag_votes[s:e] += 1.0
            if ev["code"] in ("BUILD UP", "PROGRESSION"):
                in_buildup_or_prog[s:e] = True

    phase_dir = np.where(
        (h_tag_votes > 0) & (a_tag_votes == 0), 1.0,
        np.where((a_tag_votes > 0) & (h_tag_votes == 0), -1.0, 0.0),
    )
    team_score += np.where(vis_i, 0.12 * phase_dir, 0.42 * phase_dir)

    # 6. Dead-ball evidence dead_score (>0 favors D, <0 favors Alive)
    dead_score = np.full(n, -0.55, dtype=np.float64)

    # 6a. Out-of-bounds ball
    oob = vis_clean & ((bx_clean < -0.004) | (bx_clean > 1.004) |
                       (by_clean < -0.004) | (by_clean > 1.004))
    oob_rate = rolling_sum_1d(oob.astype(np.float64), half_win=3) / 7.0
    dead_score += 4.2 * np.clip(oob_rate * 1.4, 0.0, 1.0)

    # Near-sideline/endline stationary ball
    near_line_slow = vis_clean & (
        (bx_clean < 0.015) | (bx_clean > 0.985) | (by_clean < 0.015) | (by_clean > 0.985)
    ) & (b_spd_filled < 1.5) & (mps_fast < 1.50)
    near_line_rate = rolling_sum_1d(near_line_slow.astype(np.float64), half_win=10) / 21.0
    dead_score += 2.5 * np.clip(near_line_rate * 1.5, 0.0, 1.0)

    # 6b. Asymmetric collective player activity
    in_pitch_feet = vis_clean & (~oob) & (min_d < 1.4) & (
        (bx_clean > 0.04) & (bx_clean < 0.96) & (by_clean > 0.04) & (by_clean < 0.96)
    )
    slow_pos = np.clip((1.30 - mps_slow) / 0.28, 0.0, 2.3) + \
               0.75 * np.clip((2.00 - p90_slow) / 0.45, 0.0, 1.9)
    fast_neg = np.clip((mps_slow - 1.50) / 0.40, 0.0, 0.60) + \
               0.5 * np.clip((p90_slow - 2.40) / 0.60, 0.0, 0.45)
    live_ball_inbounds = vis_clean & (~oob) & (b_spd_filled > 4.0) & (min_d < 3.5)
    act_weight = np.where(live_ball_inbounds, 0.25, np.where(in_pitch_feet & in_buildup_or_prog, 0.35, 0.85))
    dead_score += act_weight * slow_pos - fast_neg

    # 6c. Sustained in-pitch stoppages + Uncontested Slow Ball (S_ignored) + Deceleration Inflection
    slow_stop = vis_clean & (~oob) & (mps_slow < 1.25) & (p90_slow < 1.85) & (
        (b_spd_filled < 3.5) | (min_d > 3.5)
    )
    slow_stop_rate = rolling_sum_1d(slow_stop.astype(np.float64), half_win=25) / 51.0
    dead_score += 2.4 * np.clip((slow_stop_rate - 0.45) / 0.35, 0.0, 1.0)

    # Uncontested slow ball & deceleration onset catch referee whistles promptly before slow_stop saturates
    s_ignored = vis_clean & (~oob) & (b_spd_filled < 2.3) & (min_d > 3.5) & (mps_fast < 1.38)
    s_decel = vis_clean & (~oob) & (b_spd_filled < 4.5) & (min_d > 5.5) & (mps_fast < 1.75) & (mps_ahead < 1.32)
    s_ign_rate = rolling_sum_1d((s_ignored | s_decel).astype(np.float64), half_win=10) / 21.0
    dead_score += 1.8 * np.clip(s_ign_rate * 1.5, 0.0, 1.0)

    # 6d. Long ball-NaN gaps (> 35 frames = 1.4 s) inside play halves
    ko2 = anchor_video_frame(GAMES[data["game"]]["kickoff_2nd"]) + 1
    play_spans = [(half_blocks[0][1], half_blocks[0][2] + 1), (ko2, half_blocks[1][2] + 1)]
    for s_block, e_block in play_spans:
        miss = (~vis_clean[s_block:e_block]).astype(np.int8)
        edges = np.diff(np.concatenate(([0], miss, [0])))
        g_starts = np.flatnonzero(edges == 1) + s_block
        g_ends = np.flatnonzero(edges == -1) + s_block
        for a, b in zip(g_starts, g_ends):
            gap_len = b - a
            if gap_len <= 35:
                dead_score[a:b] -= 0.7
                continue

            gap_mps = float(np.mean(mps_fill[a:b]))
            gap_p90 = float(np.mean(p90_fill[a:b]))
            pre_edge = False
            if a > 0 and not np.isnan(bx_clean[a - 1]):
                pre_edge = (bx_clean[a - 1] < 0.03 or bx_clean[a - 1] > 0.97 or
                            by_clean[a - 1] < 0.03 or by_clean[a - 1] > 0.97)
            post_edge = False
            if b < n and not np.isnan(bx_clean[b]):
                post_edge = (bx_clean[b] < 0.03 or bx_clean[b] > 0.97 or
                             by_clean[b] < 0.03 or by_clean[b] > 0.97)

            if gap_mps >= 1.72 and gap_p90 >= 2.75 and not (pre_edge and gap_len > 150):
                dead_score[a:b] -= 1.4
            elif gap_mps < 1.38 or (pre_edge and gap_mps < 1.60) or (post_edge and gap_len > 150 and gap_mps < 1.55):
                strength = 1.8 + np.clip((1.45 - gap_mps) * 2.5, 0.0, 2.2)
                dead_score[a:b] += strength
            else:
                dead_score[a:b] += 0.3

    # 7. Smart Tagging GOALS celebrations & SET PIECES restarts
    goal_dead_windows = detect_goal_celebrations(events, bx_clean, mps_fast, p90_fast, n, fps)
    for stop_start, t_ko, conceding_team in goal_dead_windows:
        dead_score[stop_start:t_ko] = np.maximum(dead_score[stop_start:t_ko], 5.5)
        if conceding_team is not None:
            bias = 2.8 if conceding_team == HOME else -2.8
            team_score[t_ko:min(n, t_ko + 50)] += bias
            dead_score[t_ko:min(n, t_ko + 50)] = np.minimum(dead_score[t_ko:min(n, t_ko + 50)], -2.2)

    for t_ko_ref in (half_blocks[0][1], ko2):
        dead_score[t_ko_ref:min(n, t_ko_ref + 50)] = np.minimum(
            dead_score[t_ko_ref:min(n, t_ko_ref + 50)], -2.5
        )
        w_ko = slice(t_ko_ref, min(n, t_ko_ref + 25))
        if np.any(has_both[w_ko]):
            k_bias = 2.2 if np.mean(d_H[w_ko][has_both[w_ko]]) < np.mean(d_A[w_ko][has_both[w_ko]]) else -2.2
            team_score[t_ko_ref:min(n, t_ko_ref + 40)] += k_bias

    sp_restarts = find_set_piece_restarts(
        events, bx_clean, by_clean, b_spd_filled, mps_fast, oob, px, py,
        team_map, pitch, n, fps, goal_dead_windows
    )
    sp_restart_mask = np.zeros(n, dtype=bool)
    sp_boost = np.zeros(n, dtype=bool)
    for dead_hint_start, t_restart, sp_team, sp_type in sp_restarts:
        if t_restart - dead_hint_start >= 20:
            dead_score[dead_hint_start:t_restart] = np.maximum(
                dead_score[dead_hint_start:t_restart], 2.5
            )
            sp_restart_mask[max(0, dead_hint_start - 25):t_restart + 10] = True
        sp_boost[max(0, t_restart - 3):min(n, t_restart + 15)] = True
        post_end = min(n, t_restart + 75)
        dead_score[t_restart:post_end] = np.minimum(dead_score[t_restart:post_end], -2.0)
        if sp_team is not None:
            bias = 2.6 if sp_team == HOME else -2.6
            team_score[t_restart:min(n, t_restart + 40)] += bias

    # Deterministic 2nd-half pre-kickoff dead ball: [110808, ko2)
    h2_start = half_blocks[1][1]
    dead_score[h2_start:ko2] = 50.0

    # 8. Time-varying turnover multipliers via Outcome-Gated touch impulses
    touch_any, gate_H, gate_A, dv = compute_touch_impulses(
        bx_i, by_i, d_H, d_A, dead_score, pitch, fps
    )
    touch_or_sp = touch_any | sp_boost
    mult_to_H = np.where(
        gate_H | sp_boost, 0.78,
        np.where(touch_or_sp, 0.90, np.where(has_both & (min_d < 2.0), 1.0, 1.22)),
    )
    mult_to_A = np.where(
        gate_A | sp_boost, 0.78,
        np.where(touch_or_sp, 0.90, np.where(has_both & (min_d < 2.0), 1.0, 1.22)),
    )

    return {
        "team_score": team_score,
        "dead_score": dead_score,
        "mult_to_H": mult_to_H,
        "mult_to_A": mult_to_A,
        "dv": dv,
        "vis_clean": vis_clean,
        "vis_interp": vis_i,
        "oob": oob,
        "mps_slow": mps_slow,
        "p90_slow": p90_slow,
        "d_H": d_H,
        "d_A": d_A,
        "b_spd": b_spd_filled,
        "goal_dead_windows": goal_dead_windows,
        "sp_restarts": sp_restarts,
        "sp_restart_mask": sp_restart_mask,
        "ko2": ko2,
    }
