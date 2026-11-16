"""Domain-neutral per-frame features for the learned possession models.

Computed IDENTICALLY on Metrica open data (training) and on BAR-ATH (inference).
Features are cleanly partitioned into:
  * PLAYER_COLS: Pure collective player kinematics + PELT regime features (100% blind to ball
    visibility and symmetric under Home<->Away swap). Used by the Dead-Ball Player model so
    Metrica's ball=NaN convention during stoppages never leaks into BAR-ATH.
  * TEAM_COLS:   All ball-player proximity, velocity alignment, impulse, formation, and
    context features for Alive team attribution (H vs A) and Boundary detection.
"""

import numpy as np

from src.changepoints import compute_pelt_regime_features
from src.features import compute_ball_speed, interpolate_short_gaps, window_sum_1d

HOME, AWAY = 0, 1


def _wmean(arr, back, fwd):
    """Rolling mean over [t-back, t+fwd] of a NaN-free array."""
    num = window_sum_1d(arr, back, fwd)
    den = window_sum_1d(np.ones(len(arr), dtype=np.float64), back, fwd)
    return num / den


def _fill(arr, default):
    out = np.asarray(arr, dtype=np.float64).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return np.full(len(out), default, dtype=np.float64)
    idx = np.arange(len(out))
    return np.interp(idx, idx[valid], out[valid])


def _safe_nanmean_rows(arr, default=0.0):
    valid = np.isfinite(arr)
    cnt = valid.sum(axis=1)
    tot = np.where(valid, arr, 0.0).sum(axis=1)
    raw = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return _fill(raw, default), cnt


def _safe_nanstd_rows(arr, default=0.15):
    valid = np.isfinite(arr)
    cnt = valid.sum(axis=1)
    tot = np.where(valid, arr, 0.0).sum(axis=1)
    mu = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    sq = np.where(valid, (arr - mu[:, None]) ** 2, 0.0).sum(axis=1)
    raw = np.where(cnt > 1, np.sqrt(sq / np.maximum(cnt, 1)), np.nan)
    return _fill(raw, default)


def _ffill_bfill(arr):
    """Last known value (forward fill) and next known value (backward fill), NaN where none."""
    n = len(arr)
    valid = np.isfinite(arr)
    idx = np.where(valid, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    idx2 = np.where(valid, np.arange(n), n)
    nxt = np.minimum.accumulate(idx2[::-1])[::-1]
    f = np.where(last >= 0, arr[np.clip(last, 0, n - 1)], np.nan)
    b = np.where(nxt < n, arr[np.clip(nxt, 0, n - 1)], np.nan)
    since = np.where(last >= 0, np.arange(n) - last, 9999)
    until = np.where(nxt < n, nxt - np.arange(n), 9999)
    return f, b, since, until


def player_speeds(px, py, pitch, fps, span=4):
    """Per-slot speed (m/s) from positions, centred difference over `span` frames."""
    length, width = pitch
    n = px.shape[0]
    ps = np.full(px.shape, np.nan, dtype=np.float32)
    h = span // 2
    dx = (px[span:] - px[:-span]) * length
    dy = (py[span:] - py[:-span]) * width
    ps[h:n - h] = np.hypot(dx, dy) * (fps / span)
    return np.clip(ps, 0.0, 12.0)


def _gap_features(vis):
    n = len(vis)
    miss = (~vis).astype(np.int8)
    edges = np.diff(np.concatenate(([0], miss, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    glen = np.zeros(n, dtype=np.float64)
    gpos = np.zeros(n, dtype=np.float64)
    for a, b in zip(starts, ends):
        glen[a:b] = b - a
        gpos[a:b] = np.arange(b - a)
    return glen, gpos


def attack_directions(px, team_map, half_blocks, n):
    """+1 if Home attacks towards x=1 in that half, else -1 (from mean team x per half)."""
    dir_h = np.ones(n, dtype=np.float64)
    hm = team_map == HOME
    am = team_map == AWAY
    for _, s, e in half_blocks:
        xh = px[s:e + 1][:, hm]
        xa = px[s:e + 1][:, am]
        mx_h = float(np.nanmean(xh)) if np.isfinite(xh).any() else 0.4
        mx_a = float(np.nanmean(xa)) if np.isfinite(xa).any() else 0.6
        dir_h[s:e + 1] = 1.0 if mx_h < mx_a else -1.0
    return dir_h


def _ball_vel_impulse(bx_i, by_i, length, width, fps, span):
    """Pre/post velocity vectors and impulse magnitude |dv|, turning angle dtheta over `span`."""
    n = len(bx_i)
    bxm = bx_i.astype(np.float64) * length
    bym = by_i.astype(np.float64) * width
    vxp = np.full(n, np.nan)
    vyp = np.full(n, np.nan)
    vxq = np.full(n, np.nan)
    vyq = np.full(n, np.nan)
    if n > span:
        ok = np.isfinite(bxm[span:]) & np.isfinite(bxm[:-span])
        vx = np.where(ok, (bxm[span:] - bxm[:-span]) * (fps / float(span)), np.nan)
        vy = np.where(ok, (bym[span:] - bym[:-span]) * (fps / float(span)), np.nan)
        vxp[span:] = vx
        vyp[span:] = vy
        vxq[:-span] = vx
        vyq[:-span] = vy
    valid = np.isfinite(vxp) & np.isfinite(vxq)
    dv = np.zeros(n, dtype=np.float64)
    dv[valid] = np.clip(np.hypot(vxq[valid] - vxp[valid], vyq[valid] - vyp[valid]), 0.0, 40.0)

    sp_pre = np.hypot(vxp, vyp)
    sp_post = np.hypot(vxq, vyq)
    ok_ang = valid & (sp_pre > 2.0) & (sp_post > 2.0)
    dtheta = np.zeros(n, dtype=np.float64)
    cos_t = np.clip(
        (vxp[ok_ang] * vxq[ok_ang] + vyp[ok_ang] * vyq[ok_ang]) / (sp_pre[ok_ang] * sp_post[ok_ang]),
        -1.0, 1.0,
    )
    dtheta[ok_ang] = np.degrees(np.arccos(cos_t))

    vxc = np.nan_to_num(0.5 * (vxp + vxq), nan=0.0)
    vyc = np.nan_to_num(0.5 * (vyp + vyq), nan=0.0)
    return dv, dtheta, vxc, vyc


def compute_ml_features(data):
    """Return (F float32 (n, k), names list, groups dict{'player': [...], 'team': [...]})."""
    px = data["px"].astype(np.float32)
    py = data["py"].astype(np.float32)
    team_map = np.asarray(data["team_map"])
    fps = data["fps"]
    pitch = data["pitch"]
    length, width = pitch
    n = px.shape[0]
    half_blocks = data["half_blocks"]
    bx = data["ball_x"].astype(np.float32)
    by = data["ball_y"].astype(np.float32)

    hm = team_map == HOME
    am = team_map == AWAY
    feats = {}
    player_names = []

    def add_player(k, val):
        feats[k] = val
        player_names.append(k)

    # =========================================================================
    # 1. PURE PLAYER KINEMATICS + PELT REGIMES (ball-blind, Home/Away symmetric)
    # =========================================================================
    ps = player_speeds(px, py, pitch, fps)
    valid_p = np.isfinite(ps)
    cnt = valid_p.sum(axis=1)
    mean_ps, _ = _safe_nanmean_rows(ps, default=1.5)

    srt = np.sort(np.where(valid_p, ps, np.inf), axis=1)
    rows = np.flatnonzero(cnt > 0)
    p90 = np.full(n, np.nan, dtype=np.float64)
    p75 = np.full(n, np.nan, dtype=np.float64)
    p25 = np.full(n, np.nan, dtype=np.float64)
    p90[rows] = srt[rows, np.maximum(0, np.floor(0.90 * (cnt[rows] - 1)).astype(int))]
    p75[rows] = srt[rows, np.maximum(0, np.floor(0.75 * (cnt[rows] - 1)).astype(int))]
    p25[rows] = srt[rows, np.maximum(0, np.floor(0.25 * (cnt[rows] - 1)).astype(int))]
    p90 = _fill(p90, 2.5)
    p75 = _fill(p75, 1.9)
    p25 = _fill(p25, 0.8)

    walk = np.where(cnt > 0, (np.where(valid_p, ps, 99.0) < 1.0).sum(1) / np.maximum(cnt, 1), 0.5)
    sprint = np.where(cnt > 0, (np.where(valid_p, ps, 0.0) > 4.0).sum(1) / np.maximum(cnt, 1), 0.0)
    run = np.where(cnt > 0, (np.where(valid_p, ps, 0.0) > 2.8).sum(1) / np.maximum(cnt, 1), 0.1)

    for nm, sig in (("mps", mean_ps), ("p90", p90), ("p75", p75), ("p25", p25),
                    ("walk", walk), ("sprint", sprint), ("run", run)):
        add_player(nm, sig)
        for w in (12, 30, 75):
            add_player(f"{nm}_c{w}", _wmean(sig, w, w))
        for w in (25, 50, 100):
            add_player(f"{nm}_b{w}", _wmean(sig, w, 0))
            add_player(f"{nm}_f{w}", _wmean(sig, 0, w))

    add_player("mps_c125", _wmean(mean_ps, 125, 125))
    add_player("mps_rel", feats["mps_c12"] - feats["mps_c125"])
    add_player("mps_trend25", feats["mps_f25"] - feats["mps_b25"])
    add_player("mps_trend50", feats["mps_f50"] - feats["mps_b50"])
    add_player("mps_trend100", feats["mps_f100"] - feats["mps_b100"])
    add_player("p90_trend50", feats["p90_f50"] - feats["p90_b50"])
    add_player("walk_trend50", feats["walk_f50"] - feats["walk_b50"])

    # Standardized collective step-change GLR scores (sharp whistle / restart onset)
    sig_mat = np.column_stack([mean_ps, p90, walk, sprint])
    for w in (15, 35):
        glr = np.zeros(n, dtype=np.float64)
        for k in range(4):
            s_col = sig_mat[:, k]
            sd = max(float(np.std(s_col)), 1e-4)
            diff = (_wmean(s_col, 0, w) - _wmean(s_col, w, 0)) / sd
            glr += diff * diff
        add_player(f"glr_{w}", np.sqrt(glr))

    # PELT piecewise-constant regime segmentation on [mps, p90, walk, sprint]
    pelt_feats = compute_pelt_regime_features(sig_mat, half_blocks)
    for k, v in pelt_feats.items():
        add_player(k, v)
    add_player("pf_mps_diff", mean_ps - feats["pf_mps"])

    # Team spatial geometry (both team-symmetric for PLAYER_COLS and signed for TEAM_COLS)
    all_cx, _ = _safe_nanmean_rows(px.astype(np.float64), 0.5)
    all_sx = _safe_nanstd_rows(px.astype(np.float64), 0.15) * length
    all_sy = _safe_nanstd_rows(py.astype(np.float64), 0.15) * width
    add_player("all_sx", _wmean(all_sx, 12, 12))
    add_player("all_sy", _wmean(all_sy, 12, 12))
    cmps = np.zeros(n, dtype=np.float64)
    if n > 24:
        cmps[12:-12] = np.abs(all_cx[24:] - all_cx[:-24]) * length * (fps / 24.0)
    add_player("all_cmps", np.clip(cmps, 0.0, 10.0))

    dir_h = attack_directions(px, team_map, half_blocks, n)
    team_geom = {}
    for tag, m, sgn in (("h", hm, 1.0), ("a", am, -1.0)):
        xs = px[:, m].astype(np.float64)
        ys = py[:, m].astype(np.float64)
        cx, _ = _safe_nanmean_rows(xs, 0.5)
        sx = _safe_nanstd_rows(xs, 0.15) * length
        sy = _safe_nanstd_rows(ys, 0.15) * width
        tsp, _ = _safe_nanmean_rows(ps[:, m].astype(np.float64), 1.5)
        team_geom[tag] = (cx, sx, sy, tsp, sgn)
        feats[f"adv_{tag}"] = _wmean((cx - 0.5) * dir_h * sgn * length, 12, 12)
        feats[f"sprx_{tag}"] = _wmean(sx, 12, 12)
        feats[f"spry_{tag}"] = _wmean(sy, 12, 12)
        feats[f"tsp_{tag}"] = _wmean(tsp, 12, 12)

    add_player("team_cx_gap", np.abs(team_geom["h"][0] - team_geom["a"][0]) * length)
    add_player("sym_sprx_min", np.minimum(feats["sprx_h"], feats["sprx_a"]))
    add_player("sym_sprx_max", np.maximum(feats["sprx_h"], feats["sprx_a"]))
    add_player("sym_spry_min", np.minimum(feats["spry_h"], feats["spry_a"]))
    add_player("sym_spry_max", np.maximum(feats["spry_h"], feats["spry_a"]))
    add_player("sym_tsp_diff", np.abs(feats["tsp_h"] - feats["tsp_a"]))

    # =========================================================================
    # 2. BALL + TEAM PROXIMITY + VELOCITY ALIGNMENT + BOUNDARY FEATURES
    # =========================================================================
    vis = np.isfinite(bx)
    bx_i, by_i = interpolate_short_gaps(bx, by, max_gap=35)
    vis_i = np.isfinite(bx_i)
    feats["vis"] = vis.astype(np.float64)
    feats["vis_i"] = vis_i.astype(np.float64)
    for w in (12, 50):
        feats[f"vis_b{w}"] = _wmean(vis.astype(np.float64), w, 0)
        feats[f"vis_f{w}"] = _wmean(vis.astype(np.float64), 0, w)
    glen, gpos = _gap_features(vis)
    feats["gap_len"] = np.log1p(glen)
    feats["gap_pos"] = np.log1p(gpos)
    feats["gap_rem"] = np.log1p(np.maximum(0.0, glen - gpos))

    b_spd = compute_ball_speed(bx_i, by_i, pitch, fps)
    b_spd_f = np.nan_to_num(b_spd, nan=0.0)
    feats["b_spd"] = np.where(vis_i, b_spd_f, -1.0)
    feats["b_spd_m5"] = _wmean(b_spd_f, 5, 5)
    for w in (15, 35):
        feats[f"b_spd_b{w}"] = _wmean(b_spd_f, w, 0)
        feats[f"b_spd_f{w}"] = _wmean(b_spd_f, 0, w)
    feats["b_spd_step"] = feats["b_spd_f15"] - feats["b_spd_b15"]

    dv3, dtheta3, _, _ = _ball_vel_impulse(bx_i, by_i, length, width, fps, span=3)
    dv5, dtheta5, vxc5, vyc5 = _ball_vel_impulse(bx_i, by_i, length, width, fps, span=5)
    dv8, dtheta8, _, _ = _ball_vel_impulse(bx_i, by_i, length, width, fps, span=8)
    feats["dv3"] = dv3
    feats["dv5"] = dv5
    feats["dv8"] = dv8
    feats["dv5_max4"] = np.maximum.reduce([np.roll(dv5, k) for k in range(-4, 5)])
    feats["dv8_max4"] = np.maximum.reduce([np.roll(dv8, k) for k in range(-4, 5)])
    feats["dtheta3"] = dtheta3
    feats["dtheta5"] = dtheta5
    feats["dtheta8"] = dtheta8
    feats["touch_b25"] = window_sum_1d((dv5 > 3.0).astype(np.float64), 25, 0)
    feats["touch_f25"] = window_sum_1d((dv5 > 3.0).astype(np.float64), 0, 25)

    lx, nx, since, until = _ffill_bfill(bx.astype(np.float64))
    ly, ny, _, _ = _ffill_bfill(by.astype(np.float64))
    feats["since_vis"] = np.log1p(np.minimum(since, 3000))
    feats["until_vis"] = np.log1p(np.minimum(until, 3000))

    def _line(x, y, tag):
        x = np.nan_to_num(x, nan=0.5)
        y = np.nan_to_num(y, nan=0.5)
        feats[f"{tag}_dside"] = np.clip(np.minimum(y, 1.0 - y) * width, -5.0, 40.0)
        feats[f"{tag}_dend"] = np.clip(np.minimum(x, 1.0 - x) * length, -5.0, 60.0)
        feats[f"{tag}_oob"] = ((x < -0.004) | (x > 1.004) | (y < -0.004) | (y > 1.004)).astype(np.float64)

    _line(lx, ly, "blast")
    _line(nx, ny, "bnext")
    jump = np.hypot((nx - lx) * length, (ny - ly) * width)
    feats["gap_jump"] = np.where(vis, 0.0, np.clip(np.nan_to_num(jump, nan=0.0), 0.0, 120.0))

    bx_safe = np.where(vis_i, bx_i, 0.5)[:, None]
    by_safe = np.where(vis_i, by_i, 0.5)[:, None]
    dx_m = (px.astype(np.float64) - bx_safe) * length
    dy_m = (py.astype(np.float64) - by_safe) * width
    dist_raw = np.hypot(dx_m, dy_m)
    dist = np.where(vis_i[:, None] & np.isfinite(dist_raw), dist_raw, np.inf)

    dh_sorted = np.sort(dist[:, hm], axis=1)
    da_sorted = np.sort(dist[:, am], axis=1)
    d_h = dh_sorted[:, 0]
    d_a = da_sorted[:, 0]
    d_h2 = dh_sorted[:, 1] if dh_sorted.shape[1] > 1 else d_h
    d_a2 = da_sorted[:, 1] if da_sorted.shape[1] > 1 else d_a

    has = vis_i & np.isfinite(d_h) & np.isfinite(d_a)
    cap = 40.0
    d_h_c = np.where(has, np.minimum(d_h, cap), cap)
    d_a_c = np.where(has, np.minimum(d_a, cap), cap)
    feats["d_h"] = d_h_c
    feats["d_a"] = d_a_c
    feats["d_h2"] = np.where(has, np.minimum(d_h2, cap), cap)
    feats["d_a2"] = np.where(has, np.minimum(d_a2, cap), cap)

    dd = np.where(has, np.clip(d_a_c - d_h_c, -15.0, 15.0), 0.0)
    feats["d_diff"] = dd
    feats["d_ratio"] = np.where(has, (d_a_c - d_h_c) / (d_a_c + d_h_c + 1.0), 0.0)
    feats["min_d"] = np.minimum(d_h_c, d_a_c)
    feats["opp_d"] = np.maximum(d_h_c, d_a_c)
    feats["mind_c25"] = _wmean(feats["min_d"], 25, 25)

    for L in (1.5, 2.5, 4.0):
        feats[f"exp_prox_{L}"] = np.where(has, np.exp(-d_h_c / L) - np.exp(-d_a_c / L), 0.0)
        feats[f"exp_prox_{L}_c10"] = _wmean(feats[f"exp_prox_{L}"], 10, 10)

    for rad in (3.0, 5.0, 8.0):
        feats[f"n_h{int(rad)}"] = np.where(has, (dist[:, hm] < rad).sum(axis=1), 0).astype(np.float64)
        feats[f"n_a{int(rad)}"] = np.where(has, (dist[:, am] < rad).sum(axis=1), 0).astype(np.float64)

    close_h = (has & (d_h_c < 2.2) & (d_h_c + 0.3 < d_a_c)).astype(np.float64)
    close_a = (has & (d_a_c < 2.2) & (d_a_c + 0.3 < d_h_c)).astype(np.float64)
    for w in (6, 12, 25, 50, 100, 200):
        ch_b = window_sum_1d(close_h, w, 0) / (w + 1)
        ca_b = window_sum_1d(close_a, w, 0) / (w + 1)
        ch_f = window_sum_1d(close_h, 0, w) / (w + 1)
        ca_f = window_sum_1d(close_a, 0, w) / (w + 1)
        feats[f"ctrl_h_b{w}"] = ch_b
        feats[f"ctrl_a_b{w}"] = ca_b
        feats[f"ctrl_h_f{w}"] = ch_f
        feats[f"ctrl_a_f{w}"] = ca_f
        feats[f"ctrl_diff_b{w}"] = ch_b - ca_b
        feats[f"ctrl_diff_f{w}"] = ch_f - ca_f
    for w in (12, 25, 50):
        feats[f"ctrl_shift_{w}"] = feats[f"ctrl_diff_f{w}"] - feats[f"ctrl_diff_b{w}"]

    t_dd = np.tanh(dd / 2.5)
    for w in (5, 15, 35):
        feats[f"ddiff_c{w}"] = _wmean(t_dd, w, w)
    for w in (12, 25, 50):
        b_dd = _wmean(t_dd, w, 0)
        f_dd = _wmean(t_dd, 0, w)
        feats[f"ddiff_b{w}"] = b_dd
        feats[f"ddiff_f{w}"] = f_dd
        feats[f"ddiff_shift_{w}"] = f_dd - b_dd

    h_slots = np.flatnonzero(hm)
    a_slots = np.flatnonzero(am)
    ih = h_slots[np.argmin(dist[:, h_slots], axis=1)]
    ia = a_slots[np.argmin(dist[:, a_slots], axis=1)]
    idx_n = np.arange(n)

    sp_h = np.nan_to_num(ps[idx_n, ih], nan=0.0)
    sp_a = np.nan_to_num(ps[idx_n, ia], nan=0.0)
    feats["sp_near_h"] = np.where(has, sp_h, 0.0)
    feats["sp_near_a"] = np.where(has, sp_a, 0.0)
    feats["rel_spd_h"] = np.where(has, np.abs(b_spd_f - sp_h), 0.0)
    feats["rel_spd_a"] = np.where(has, np.abs(b_spd_f - sp_a), 0.0)

    b_sp_c = np.hypot(vxc5, vyc5)
    for tag, islot in (("h", ih), ("a", ia)):
        ux = np.nan_to_num(-dx_m[idx_n, islot], nan=0.0)
        uy = np.nan_to_num(-dy_m[idx_n, islot], nan=0.0)
        un = np.hypot(ux, uy)
        ok_al = has & (un > 0.25) & (b_sp_c > 1.0)
        dot = np.where(ok_al, (ux * vxc5 + uy * vyc5) / np.maximum(un * b_sp_c, 1e-6), 0.0)
        feats[f"v_align_{tag}"] = dot

    dh_sl = np.zeros(n, dtype=np.float64)
    da_sl = np.zeros(n, dtype=np.float64)
    if n > 8:
        ok_sl = has[8:] & has[:-8]
        dh_sl[4:-4] = np.where(ok_sl, np.clip(d_h_c[8:] - d_h_c[:-8], -15.0, 15.0), 0.0)
        da_sl[4:-4] = np.where(ok_sl, np.clip(d_a_c[8:] - d_a_c[:-8], -15.0, 15.0), 0.0)
    feats["d_slope_h"] = dh_sl
    feats["d_slope_a"] = da_sl

    feats["adv_sum"] = feats["adv_h"] + feats["adv_a"]
    feats["ball_adv_h"] = np.where(vis_i, (np.nan_to_num(bx_i, nan=0.5) - 0.5) * dir_h * length, 0.0)
    feats["spr_diff"] = feats["sprx_h"] * feats["spry_h"] - feats["sprx_a"] * feats["spry_a"]

    names = list(feats.keys())
    F = np.empty((n, len(names)), dtype=np.float32)
    for j, k in enumerate(names):
        col = np.asarray(feats[k], dtype=np.float64)
        F[:, j] = np.where(np.isfinite(col), col, 0.0)

    # Symmetric boundary features for H<->A boundary detector (invariant to who had the ball before vs after)
    # Add these right here before returning:
    extra = {}
    extra["abs_ctrl_shift_12"] = np.abs(feats["ctrl_shift_12"])
    extra["abs_ctrl_shift_25"] = np.abs(feats["ctrl_shift_25"])
    extra["abs_ctrl_shift_50"] = np.abs(feats["ctrl_shift_50"])
    extra["abs_ddiff_shift_12"] = np.abs(feats["ddiff_shift_12"])
    extra["abs_ddiff_shift_25"] = np.abs(feats["ddiff_shift_25"])
    extra["abs_ddiff_shift_50"] = np.abs(feats["ddiff_shift_50"])
    extra["abs_d_diff"] = np.abs(dd)
    # Product of pre-H/post-A or pre-A/post-H flip evidence!
    extra["cross_ctrl_12"] = np.maximum(
        feats["ctrl_h_b12"] * feats["ctrl_a_f12"],
        feats["ctrl_a_b12"] * feats["ctrl_h_f12"],
    )
    extra["cross_ctrl_25"] = np.maximum(
        feats["ctrl_h_b25"] * feats["ctrl_a_f25"],
        feats["ctrl_a_b25"] * feats["ctrl_h_f25"],
    )
    extra["sign_flip_ddiff_12"] = np.maximum(0.0, -feats["ddiff_b12"] * feats["ddiff_f12"])
    extra["sign_flip_ddiff_25"] = np.maximum(0.0, -feats["ddiff_b25"] * feats["ddiff_f25"])

    all_names = names + list(extra.keys())
    F_all = np.empty((n, len(all_names)), dtype=np.float32)
    F_all[:, :len(names)] = F
    for idx_e, k in enumerate(extra.keys()):
        col = np.asarray(extra[k], dtype=np.float64)
        F_all[:, len(names) + idx_e] = np.where(np.isfinite(col), col, 0.0)

    # Team/boundary models exclude raw Metrica-biased `vis` gap features (they see `vis_i` & ball-kinematics)
    leak_cols = {"vis", "gap_len", "gap_pos", "gap_rem", "since_vis", "until_vis", "gap_jump", "vis_b12", "vis_f12", "vis_b50", "vis_f50"}
    team_names = [k for k in all_names if k not in leak_cols]
    # Boundary H<->A uses symmetric player+ball+shift features (no directional H/A bias)
    sym_exclude = leak_cols | {
        "d_h", "d_a", "d_h2", "d_a2", "d_diff", "d_ratio",
        "exp_prox_1.5", "exp_prox_2.5", "exp_prox_4.0",
        "exp_prox_1.5_c10", "exp_prox_2.5_c10", "exp_prox_4.0_c10",
        "n_h3", "n_a3", "n_h5", "n_a5", "n_h8", "n_a8",
        "sp_near_h", "sp_near_a", "rel_spd_h", "rel_spd_a",
        "v_align_h", "v_align_a", "d_slope_h", "d_slope_a",
        "adv_h", "adv_a", "adv_sum", "ball_adv_h", "sprx_h", "spry_h", "sprx_a", "spry_a", "tsp_h", "tsp_a", "spr_diff",
    } | {k for k in all_names if k.startswith(("ctrl_h_", "ctrl_a_", "ctrl_diff_", "ctrl_shift_", "ddiff_"))}
    bound_names = [k for k in all_names if k not in sym_exclude]

    groups = {
        "player": player_names,
        "team": team_names,
        "boundary": bound_names,
        "all": all_names,
    }
    return F_all, all_names, groups
