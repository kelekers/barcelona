"""Physical ball-state cues and Smart Tagging anchors for the Hybrid decoder.

Combines cleanly on top of the DFL-trained external probabilities (`dead_head` + `team_head`):
  1. `compute_physical_cues(match)` uses only tracking geometry (no Smart Tagging, no match
     labels) so it runs identically on Metrica external validation and BAR-ATH ATD:
       - `oob`: visible ball outside pitch boundaries -> dead-ball evidence & crisp H/A->D exit
       - `live_in`: visible in-bounds ball moving >3.0 m/s near a player -> live-ball rescue
       - `near_line_slow` / `s_ignored`: visible stationary ball near touchline or uncontested
       - `touch` / `fast_pass`: ball velocity-impulse modulation of H<->A switch penalties
  2. `compute_smart_tagging_priors(match, cues)` reads tactical Smart Tagging when available
     (BAR-ATH only; no-op on external datasets) to anchor GOALS celebrations, SET PIECES
     restarts (including throw-in touchline spatial votes), and offensive phase tie-breakers.
"""

import os
import sys

import numpy as np

from .common import A, FPS, H, ROOT

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAMES, anchor_video_frame, asset_paths  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402


def _rolling_sum(arr, back, fwd):
    n = len(arr)
    cs = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - back)
    hi = np.minimum(n, idx + fwd + 1)
    return cs[hi] - cs[lo]


def _rolling_mean_centered(arr, half_win):
    return _rolling_sum(arr, half_win, half_win) / (2.0 * half_win + 1.0)


def _interpolate_short_gaps(bx, by, max_gap=35):
    bx_out = bx.copy()
    by_out = by.copy()
    n = len(bx)
    miss = ~np.isfinite(bx)
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if a == 0 or b == n:
            continue
        gap = b - a
        if gap <= max_gap:
            w = np.linspace(0.0, 1.0, gap + 2, dtype=np.float32)[1:-1]
            bx_out[a:b] = bx[a - 1] * (1.0 - w) + bx[b] * w
            by_out[a:b] = by[a - 1] * (1.0 - w) + by[b] * w
    return bx_out, by_out


def _player_activity(px, py, pitch, fps):
    length, width = pitch
    n = px.shape[0]
    vx = np.full_like(px, np.nan, dtype=np.float64)
    vy = np.full_like(py, np.nan, dtype=np.float64)
    if n > 4:
        vx[2:-2] = (px[4:] - px[:-4]) * (length * fps / 4.0)
        vy[2:-2] = (py[4:] - py[:-4]) * (width * fps / 4.0)
    spd = np.hypot(vx, vy)
    spd[spd > 12.0] = np.nan
    valid = np.isfinite(spd)
    cnt = valid.sum(axis=1)
    safe = np.where(valid, spd, 0.0)
    raw_mps = np.where(cnt > 0, safe.sum(axis=1) / np.maximum(cnt, 1), np.nan)
    idx = np.arange(n)
    ok = np.isfinite(raw_mps)
    mps = np.interp(idx, idx[ok], raw_mps[ok]) if ok.any() else np.full(n, 1.5, dtype=np.float64)

    srt = np.sort(np.where(valid, spd, np.inf), axis=1)
    rows = np.flatnonzero(cnt > 0)
    raw_p90 = np.full(n, np.nan, dtype=np.float64)
    raw_p90[rows] = srt[rows, np.maximum(0, np.floor(0.90 * (cnt[rows] - 1)).astype(int))]
    ok9 = np.isfinite(raw_p90)
    p90 = np.interp(idx, idx[ok9], raw_p90[ok9]) if ok9.any() else np.full(n, 2.3, dtype=np.float64)

    mps_fast = _rolling_mean_centered(mps, 12)
    p90_fast = _rolling_mean_centered(p90, 12)
    return mps_fast, p90_fast


def compute_physical_cues(match):
    """Extract pure geometric/kinematic cues without any Smart Tagging or labels."""
    px = match["px"].astype(np.float64)
    py = match["py"].astype(np.float64)
    bx = match["ball_x"].astype(np.float64)
    by = match["ball_y"].astype(np.float64)
    pitch = match["pitch"]
    length, width = pitch
    fps = match.get("fps", FPS)
    n = match["n_frames"]
    tm = np.asarray(match["team_map"])

    vis = np.isfinite(bx) & np.isfinite(by)
    oob = vis & ((bx < -0.004) | (bx > 1.004) | (by < -0.004) | (by > 1.004))

    bx_i, by_i = _interpolate_short_gaps(bx.astype(np.float32), by.astype(np.float32), max_gap=35)
    vis_i = np.isfinite(bx_i) & np.isfinite(by_i)
    bxm = np.where(vis_i, bx_i, 0.5).astype(np.float64) * length
    bym = np.where(vis_i, by_i, 0.5).astype(np.float64) * width

    b_spd = np.zeros(n, dtype=np.float64)
    if n > 4:
        ok4 = vis_i[4:] & vis_i[:-4]
        b_spd[2:-2] = np.where(ok4, np.hypot(bxm[4:] - bxm[:-4], bym[4:] - bym[:-4]) * (fps / 4.0), 0.0)
    b_spd = np.clip(b_spd, 0.0, 45.0)

    dv = np.zeros(n, dtype=np.float64)
    if n > 8:
        okv = vis_i[8:] & vis_i[4:-4] & vis_i[:-8]
        vxp = (bxm[4:-4] - bxm[:-8]) * (fps / 4.0)
        vyp = (bym[4:-4] - bym[:-8]) * (fps / 4.0)
        vxq = (bxm[8:] - bxm[4:-4]) * (fps / 4.0)
        vyq = (bym[8:] - bym[4:-4]) * (fps / 4.0)
        dv[4:-4] = np.where(okv, np.hypot(vxq - vxp, vyq - vyp), 0.0)

    dx = (px - np.where(vis_i, bx_i, 0.5)[:, None]) * length
    dy = (py - np.where(vis_i, by_i, 0.5)[:, None]) * width
    dist = np.where(vis_i[:, None] & np.isfinite(dx) & np.isfinite(dy), np.hypot(dx, dy), np.inf)
    d_h = np.min(dist[:, tm == H], axis=1)
    d_a = np.min(dist[:, tm == A], axis=1)
    min_d = np.minimum(d_h, d_a)

    mps_fast, p90_fast = _player_activity(px, py, pitch, fps)

    oob_rate = _rolling_mean_centered(oob.astype(np.float64), 3)
    oob_ev = np.clip(oob_rate * 1.4, 0.0, 1.0)

    near_line_slow = vis & (
        (bx < 0.015) | (bx > 0.985) | (by < 0.015) | (by > 0.985)
    ) & (b_spd < 1.5) & (mps_fast < 1.50)
    near_line_ev = np.clip(_rolling_mean_centered(near_line_slow.astype(np.float64), 10) * 1.5, 0.0, 1.0)

    s_ignored = vis & (~oob) & (b_spd < 2.3) & (min_d > 3.5) & (mps_fast < 1.38)
    s_ign_ev = np.clip(_rolling_mean_centered(s_ignored.astype(np.float64), 10) * 1.5, 0.0, 1.0)

    live_in = vis & (~oob) & (b_spd > 3.0) & (min_d < 3.5)
    touch_raw = (vis_i & (dv > 2.2) & (min_d < 3.0)).astype(np.float64)
    touch_win = np.convolve(touch_raw, np.ones(7, dtype=np.float64), mode="same") > 0
    fast_pass = vis_i & (b_spd > 7.5) & (min_d > 1.8) & (~touch_win)

    return {
        "vis": vis,
        "vis_i": vis_i,
        "oob": oob,
        "oob_ev": oob_ev,
        "near_line_ev": near_line_ev,
        "s_ign_ev": s_ign_ev,
        "live_in": live_in,
        "touch": touch_win,
        "fast_pass": fast_pass,
        "b_spd": b_spd,
        "dv": dv,
        "d_h": d_h,
        "d_a": d_a,
        "mps_fast": mps_fast,
        "p90_fast": p90_fast,
    }


def _detect_goal_celebrations(events, vis, mps_fast, p90_fast, n_frames, fps):
    goals = [ev for ev in events if ev["code"] == "GOALS"]
    kickoffs = [
        ev for ev in events
        if ev["code"] == "SET PIECES"
        and "kick off (after goal)" in (ev["tags"].get("Type") or "").lower()
    ]
    dead_windows = []
    used_ko = set()
    for g in goals:
        ga, gb = g["start_frame"], g["end_frame"]
        scoring = (g["tags"].get("Team") or "").strip().lower()
        if "barcelona" in scoring:
            conceding = A
        elif "atletic" in scoring or "athletic" in scoring:
            conceding = H
        else:
            conceding = None

        best_ko, best_dist = None, 999999
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
        ko_a = best_ko[1]["start_frame"]
        t_ko = min(n_frames - 1, ko_a + int(round(3.32 * fps)))

        stop_start = max(ga, t_ko - 50)
        consec = 0
        for t in range(t_ko - 1, max(0, ga - 50) - 1, -1):
            ball_here = bool(vis[t])
            active = (mps_fast[t] > 1.65) or (p90_fast[t] > 2.85 and ball_here)
            if active and ball_here:
                consec += 1
                if consec >= 12:
                    stop_start = t + 12
                    break
            else:
                consec = 0
                stop_start = t
        if t_ko - stop_start >= 5 * fps:
            dead_windows.append((stop_start, t_ko, conceding))
    return dead_windows


def _find_set_piece_restarts(events, bx, by, vis, b_spd, mps_fast, oob, px, py,
                             tm, pitch, n_frames, fps, goal_dead_windows):
    length, width = pitch
    restarts = []
    in_goal = np.zeros(n_frames, dtype=bool)
    for a, b, _ in goal_dead_windows:
        in_goal[max(0, a):min(n_frames, b)] = True

    for ev in events:
        if ev["code"] != "SET PIECES":
            continue
        sp_type = (ev["tags"].get("Type") or "").strip()
        if "kick off" in sp_type.lower():
            continue
        sp_a = ev["start_frame"]
        if in_goal[min(n_frames - 1, sp_a + 25)]:
            continue

        tm_str = (ev["tags"].get("Team") or "").strip().lower()
        if "barcelona" in tm_str:
            sp_team = H
        elif "atletic" in tm_str or "athletic" in tm_str:
            sp_team = A
        else:
            sp_team = None

        w0 = max(2, sp_a - 2 * fps)
        w1 = min(n_frames - 30, sp_a + 14 * fps)
        if w1 <= w0 + 10:
            continue

        best_t, best_score = None, -1e9
        for t in range(w0, w1):
            pre0 = max(0, t - 45)
            post1 = min(n_frames, t + 30)
            pre_dead_frac = float(np.mean((~vis[pre0:t]) | oob[pre0:t] | (b_spd[pre0:t] < 1.8))) if t > pre0 else 0.0
            imm1 = min(n_frames, t + 4)
            if np.mean(vis[t:imm1] & (~oob[t:imm1])) < 0.75:
                continue
            spd_at_t = float(np.mean(b_spd[t:imm1]))
            pre_imm = float(np.mean(b_spd[max(0, t - 6):t])) if t > 0 else 0.0
            if spd_at_t < 3.4:
                t15 = min(n_frames - 1, t + 15)
                if (not vis[t]) or (not vis[t15]):
                    continue
                if np.hypot((bx[t15] - bx[t]) * length, (by[t15] - by[t]) * width) < 2.5:
                    continue
            post_live_frac = float(np.mean(vis[t:post1] & (~oob[t:post1]) & (b_spd[t:post1] > 3.2)))
            accel = np.clip((spd_at_t - pre_imm) / 5.0, 0.0, 1.0)
            if pre_dead_frac >= 0.65 and post_live_frac >= 0.35:
                sc = pre_dead_frac + 1.3 * post_live_frac + 0.5 * accel + 0.08 * min(spd_at_t, 12.0)
                if sc > best_score:
                    best_score, best_t = sc, t

        if best_t is None:
            for t in range(w0, w1, 2):
                pre0, post1 = max(0, t - 45), min(n_frames, t + 30)
                pre_df = float(np.mean((~vis[pre0:t]) | oob[pre0:t] | (b_spd[pre0:t] < 1.8)))
                post_lf = float(np.mean(vis[t:post1] & (~oob[t:post1]) & (b_spd[t:post1] > 2.8)))
                if pre_df + post_lf > best_score:
                    best_score, best_t = pre_df + post_lf, t

        if best_t is None:
            continue

        if sp_team is None and "throw" in sp_type.lower():
            h_votes = a_votes = 0.0
            for tt in range(max(0, best_t - 75), min(n_frames, best_t + 30), 2):
                side_mask = (py[tt] < 0.035) | (py[tt] > 0.965)
                if np.any(side_mask):
                    if vis[tt]:
                        d = np.hypot((px[tt] - bx[tt]) * length, (py[tt] - by[tt]) * width)
                        d = np.where(np.isfinite(d), d, np.inf)
                        idx_min = int(np.argmin(d))
                        if d[idx_min] < 6.0:
                            if tm[idx_min] == H:
                                h_votes += 2.0
                            else:
                                a_votes += 2.0
                    h_votes += 0.3 * float(np.sum(side_mask & (tm == H)))
                    a_votes += 0.3 * float(np.sum(side_mask & (tm == A)))
                elif vis[tt]:
                    d = np.hypot((px[tt] - bx[tt]) * length, (py[tt] - by[tt]) * width)
                    d = np.where(np.isfinite(d), d, np.inf)
                    idx_min = int(np.argmin(d))
                    if d[idx_min] < 4.0:
                        if tm[idx_min] == H:
                            h_votes += 1.0
                        else:
                            a_votes += 1.0
            if abs(h_votes - a_votes) > 1.0:
                sp_team = H if h_votes > a_votes else A

        back_limit = max(0, best_t - 15 * fps)
        dead_hint_start = best_t
        active_streak = 0
        for t in range(best_t - 2, back_limit - 1, -1):
            is_dead = (not vis[t]) or oob[t] or (b_spd[t] < 1.5 and mps_fast[t] < 1.45)
            if is_dead:
                dead_hint_start = t
                active_streak = 0
            else:
                active_streak += 1
                if active_streak >= 8:
                    break
        oob_idxs = np.flatnonzero(oob[max(0, best_t - 14 * fps):best_t])
        if len(oob_idxs) >= 4:
            dead_hint_start = min(dead_hint_start, max(0, best_t - 14 * fps) + int(oob_idxs[0]))
        restarts.append((dead_hint_start, best_t, sp_team, sp_type))
    return restarts


def compute_smart_tagging_priors(match, cues, data_dir=None):
    """Build soft log-odds priors and restart anchors from tactical Smart Tagging."""
    game = match["game"]
    if game not in GAMES:
        return None
    try:
        paths = asset_paths(game=game, data_dir=data_dir)
        tag_path = paths.get("smart_tagging")
    except Exception:
        return None
    if not tag_path or not os.path.exists(tag_path):
        return None

    events = load_smart_tagging(game=game, data_dir=data_dir, fps=match.get("fps", FPS))
    n = match["n_frames"]
    fps = match.get("fps", FPS)
    vis = cues["vis"]
    vis_i = cues["vis_i"]
    oob = cues["oob"]
    b_spd = cues["b_spd"]
    d_h = cues["d_h"]
    d_a = cues["d_a"]
    mps_fast = cues["mps_fast"]
    p90_fast = cues["p90_fast"]

    off_codes = {
        "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
        "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL",
    }
    active_tempo_codes = {"ATTACKING TRANSITION", "LONG BALL", "BUILD UP", "PROGRESSION"}
    h_votes = np.zeros(n, dtype=np.float64)
    a_votes = np.zeros(n, dtype=np.float64)
    sp_any = np.zeros(n, dtype=bool)
    goal_any = np.zeros(n, dtype=bool)
    active_tempo = np.zeros(n, dtype=bool)
    for ev in events:
        s, e = max(0, ev["start_frame"]), min(n, ev["end_frame"])
        if ev["code"] == "SET PIECES":
            sp_any[max(0, s - 15):min(n, e + 15)] = True
        elif ev["code"] == "GOALS":
            goal_any[max(0, s - 25):min(n, e + 25)] = True
        if ev["code"] in active_tempo_codes:
            active_tempo[s:e] = True
        if ev["code"] in off_codes:
            tm_str = (ev["tags"].get("Team") or "").strip().lower()
            if "barcelona" in tm_str:
                h_votes[s:e] += 1.0
            elif "atletic" in tm_str or "athletic" in tm_str:
                a_votes[s:e] += 1.0
    phase_dir = np.where(
        (h_votes > 0) & (a_votes == 0), 1.0,
        np.where((a_votes > 0) & (h_votes == 0), -1.0, 0.0),
    )

    team_bias = np.where(vis_i, 0.18 * phase_dir, 0.90 * phase_dir)
    dead_bias = np.zeros(n, dtype=np.float64)
    # Protect active-tempo phases (AT/LB/Build-Up/Progression) during ball-NaN gaps without SP/Goal
    active_nan_live = (~vis) & (~oob) & active_tempo & (~sp_any) & (~goal_any) & (mps_fast > 1.55)
    dead_bias = np.where(active_nan_live, -1.8, dead_bias)
    sp_boost = np.zeros(n, dtype=bool)
    sp_mask = np.zeros(n, dtype=bool)
    force_live = np.zeros(n, dtype=bool)

    goal_windows = _detect_goal_celebrations(events, vis, mps_fast, p90_fast, n, fps)
    for stop_start, t_ko, conceding in goal_windows:
        dead_bias[stop_start:t_ko] = np.maximum(dead_bias[stop_start:t_ko], 8.0)
        force_live[t_ko:min(n, t_ko + 50)] = True
        dead_bias[t_ko:min(n, t_ko + 50)] = np.minimum(dead_bias[t_ko:min(n, t_ko + 50)], -6.0)
        sp_boost[max(0, t_ko - 3):min(n, t_ko + 12)] = True
        if conceding is not None:
            team_bias[t_ko:min(n, t_ko + 50)] += (4.5 if conceding == H else -4.5)

    hb = match["half_blocks"]
    ko2 = anchor_video_frame(GAMES[game]["kickoff_2nd"]) + 1
    has_both = vis_i & np.isfinite(d_h) & np.isfinite(d_a)
    for t_ko_ref in (hb[0][1], ko2):
        force_live[t_ko_ref:min(n, t_ko_ref + 50)] = True
        dead_bias[t_ko_ref:min(n, t_ko_ref + 50)] = np.minimum(dead_bias[t_ko_ref:min(n, t_ko_ref + 50)], -6.0)
        w_ko = slice(t_ko_ref, min(n, t_ko_ref + 25))
        if np.any(has_both[w_ko]):
            k_b = 3.5 if np.mean(d_h[w_ko][has_both[w_ko]]) < np.mean(d_a[w_ko][has_both[w_ko]]) else -3.5
            team_bias[t_ko_ref:min(n, t_ko_ref + 40)] += k_b

    sp_restarts = _find_set_piece_restarts(
        events, match["ball_x"], match["ball_y"], vis, b_spd, mps_fast, oob,
        match["px"], match["py"], match["team_map"], match["pitch"], n, fps, goal_windows,
    )
    for dead_hint_start, t_restart, sp_team, sp_type in sp_restarts:
        if t_restart - dead_hint_start >= 10:
            dead_bias[dead_hint_start:t_restart] = np.maximum(dead_bias[dead_hint_start:t_restart], 4.0)
            sp_mask[max(0, dead_hint_start - 25):t_restart + 10] = True
        sp_boost[max(0, t_restart - 3):min(n, t_restart + 15)] = True
        post_end = min(n, t_restart + 75)
        force_live[t_restart:post_end] = True
        dead_bias[t_restart:post_end] = np.minimum(dead_bias[t_restart:post_end], -5.0)
        if sp_team is not None:
            team_bias[t_restart:min(n, t_restart + 40)] += (4.5 if sp_team == H else -4.5)

    return {
        "team_bias": team_bias,
        "dead_bias": dead_bias,
        "sp_boost": sp_boost,
        "sp_mask": sp_mask,
        "force_live": force_live,
        "goal_windows": goal_windows,
        "sp_restarts": sp_restarts,
        "ko2": ko2,
    }
