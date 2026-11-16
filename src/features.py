"""Per-frame features, identical for every data source.

Design rules, each one there to keep a model trained on other providers' games valid here:

  * Permutation-invariant. Track slots carry no identity, so nothing is per-player: only
    per-team minima, counts and moments.
  * Metres and seconds, never pitch-normalised units or frame indices.
  * Mirror-invariant. Positions enter as distances to the nearest line or along the home team's
    attack axis (inferred per half from where the two teams stand), so which goal a team defends
    never matters.
  * Missing-aware. Everything derived from the ball is NaN while the ball is not tracked; the
    gradient-boosted trees route NaN natively, and a block of ball-free "collective" features
    (speeds, centroid motion, team shape) carries those frames.
  * Bidirectional context. The task is offline, so each signal is summarised over windows before
    AND after the frame; a possession change shows up as a past/future contrast.
"""

import warnings

import numpy as np

from .common import FPS

# (name, windows in frames) -- past window is [t-w, t-1], future window is [t+1, t+w].
SLOW_WINDOWS = (25, 75, 250)        # collective motion: stoppages build up over seconds
FAST_WINDOWS = (3, 6, 15, 40, 100)  # ball and control: a touch is a fraction of a second
SLOW_SIGNALS = ("mps", "walk_frac", "run_frac", "stand_frac", "top3_spd", "cvu_H", "cvu_A",
                "p_line_min", "p_corner_min", "n_box", "b_vis")
FAST_SIGNALS = ("b_spd", "b_vu", "b_acc", "dmin", "ddiff", "ctrl_H", "ctrl_A", "near_H",
                "b_edge_x", "b_edge_y", "rel_H", "rel_A")

MAX_PLAYER_SPEED = 12.0   # m/s; anything faster is a track-slot identity switch, not a player
CONTROL_RADIUS = 2.5      # m
GAP_CAP_S = 60.0
# A "touch" is a local peak of the ball's velocity change (m/s over four frames) above a
# threshold; two thresholds, because automatic ball tracks are smoothed and blunt the peaks.
TOUCH_THRESHOLDS = (1.5, 3.0)
TOUCH_CAP_S = 20.0


def rolling_nanmean(x, lo, hi):
    """Mean of x[t+lo .. t+hi] ignoring NaN; NaN where the window holds no finite sample."""
    finite = np.isfinite(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0), dtype=np.float64)))
    ccnt = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(x))
    a = np.clip(index + lo, 0, len(x))
    b = np.clip(index + hi + 1, 0, len(x))
    count = ccnt[b] - ccnt[a]
    out = (csum[b] - csum[a]) / np.maximum(count, 1)
    out[count == 0] = np.nan
    return out.astype(np.float32)


def rolling_extreme(x, lo, hi, fn):
    """fn (np.nanmax / np.nanmin) of x[t+lo .. t+hi]; NaN where the window holds no sample."""
    n = len(x)
    left, right = max(0, -lo), max(0, hi)
    padded = np.concatenate([np.full(left, np.nan, np.float32), x.astype(np.float32),
                             np.full(right, np.nan, np.float32)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, hi - lo + 1)
    start = lo + left
    return fn(windows[start:start + n], axis=1)


def touch_features(b_acc, ddiff, dmin):
    """Who was at the ball the last time, and the next time, its velocity changed sharply.

    Possession follows touches: a pass in flight still belongs to the passer, and the label
    flips at the receiving team's touch. These features hand the model that structure directly:
    time since / until the neighbouring touches and which team stood closer at each of them.
    """
    n = len(b_acc)
    acc = np.nan_to_num(b_acc, nan=0.0)
    peak = np.ones(n, bool)
    for k in (1, 2, 3):
        peak[k:] &= acc[k:] >= acc[:-k]
        peak[:-k] &= acc[:-k] >= acc[k:]
    dd = rolling_nanmean(ddiff, -2, 2)
    dm = rolling_nanmean(dmin, -2, 2)
    index = np.arange(n)
    out = {}
    for threshold in TOUCH_THRESHOLDS:
        tag = "t{:02d}".format(int(round(threshold * 10)))
        touch = peak & (acc > threshold) & np.isfinite(dd)
        last = np.maximum.accumulate(np.where(touch, index, -1))
        nxt = np.minimum.accumulate(np.where(touch, index, n)[::-1])[::-1]
        has_last, has_next = last >= 0, nxt < n
        safe_last, safe_next = np.maximum(last, 0), np.minimum(nxt, n - 1)
        out[tag + "_since"] = np.where(has_last, np.minimum((index - last) / FPS, TOUCH_CAP_S), np.nan)
        out[tag + "_dd_last"] = np.where(has_last, dd[safe_last], np.nan)
        out[tag + "_dm_last"] = np.where(has_last, dm[safe_last], np.nan)
        out[tag + "_until"] = np.where(has_next, np.minimum((nxt - index) / FPS, TOUCH_CAP_S), np.nan)
        out[tag + "_dd_next"] = np.where(has_next, dd[safe_next], np.nan)
        out[tag + "_dm_next"] = np.where(has_next, dm[safe_next], np.nan)
        # the touch before the last one, and the one after the next: was it the same team?
        before = np.concatenate(([-1], last[:-1]))          # last touch strictly before t
        after = np.concatenate((nxt[1:], [n]))              # next touch strictly after t
        last2 = np.where(has_last, before[safe_last], -1)
        next2 = np.where(has_next, after[safe_next], n)
        out[tag + "_dd_last2"] = np.where(last2 >= 0, dd[np.maximum(last2, 0)], np.nan)
        out[tag + "_dd_next2"] = np.where(next2 < n, dd[np.minimum(next2, n - 1)], np.nan)
    return out


def _run_lengths(mask):
    """For each True frame: frames since the run began (1-based), frames to its end, run length."""
    n = len(mask)
    since = np.zeros(n, np.float32)
    until = np.zeros(n, np.float32)
    length = np.zeros(n, np.float32)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        since[a:b] = np.arange(1, b - a + 1)
        until[a:b] = np.arange(b - a, 0, -1)
        length[a:b] = b - a
    return since, until, length


def _two_smallest(dist, mask):
    """Smallest and second-smallest distance over the slots in mask; NaN where unavailable."""
    sub = np.where(np.isfinite(dist[:, mask]), dist[:, mask], np.inf)
    part = np.partition(sub, 1, axis=1)[:, :2]
    part.sort(axis=1)
    part[~np.isfinite(part)] = np.nan
    return part[:, 0], part[:, 1]


def attack_direction(match):
    """+1 where the home team attacks towards larger x, -1 otherwise; one value per frame."""
    length = match["pitch"][0]
    home = match["team_map"] == 0
    direction = np.ones(match["n_frames"], np.float32)
    blocks = sorted(match["half_blocks"], key=lambda b: b[1])
    for i, (_, first, last) in enumerate(blocks):
        x = match["px"][first:last + 1:25] * length
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sign = 1.0 if np.nanmean(x[:, ~home]) > np.nanmean(x[:, home]) else -1.0
        start = 0 if i == 0 else first
        stop = match["n_frames"] if i == len(blocks) - 1 else blocks[i + 1][1]
        direction[start:stop] = sign
    return direction


def build_features(match):
    """(features float32 [n_frames, n_features], names). See the module docstring."""
    length, width = match["pitch"]
    n = match["n_frames"]
    home = match["team_map"] == 0
    away = ~home
    X = match["px"].astype(np.float32) * np.float32(length)
    Y = match["py"].astype(np.float32) * np.float32(width)
    bx = match["ball_x"].astype(np.float32) * np.float32(length)
    by = match["ball_y"].astype(np.float32) * np.float32(width)
    direction = attack_direction(match)
    base = {}

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)

        # -- players: velocities over a 4-frame baseline ---------------------------------------
        vx = np.full_like(X, np.nan)
        vy = np.full_like(Y, np.nan)
        vx[2:-2] = (X[4:] - X[:-4]) * (FPS / 4.0)
        vy[2:-2] = (Y[4:] - Y[:-4]) * (FPS / 4.0)
        speed = np.hypot(vx, vy)
        glitch = speed > MAX_PLAYER_SPEED
        vx[glitch], vy[glitch], speed[glitch] = np.nan, np.nan, np.nan
        tracked = np.isfinite(speed)
        n_tracked = np.maximum(tracked.sum(1), 1)

        base["mps"] = np.nanmean(speed, axis=1)
        base["walk_frac"] = ((speed < 1.0) & tracked).sum(1) / n_tracked
        base["run_frac"] = ((speed > 4.0) & tracked).sum(1) / n_tracked
        base["stand_frac"] = ((speed < 0.5) & tracked).sum(1) / n_tracked
        base["top3_spd"] = np.partition(np.nan_to_num(speed, nan=0.0), -3, axis=1)[:, -3:].mean(1)
        # Restart shapes that need no ball: a player on the touchline (throw-in), at a corner
        # flag (corner kick), a crowded penalty area (corner or free kick).
        line = np.minimum(Y, width - Y)
        base["p_line_min"] = np.nanmin(line, axis=1)
        base["p_corner_min"] = np.nanmin(np.hypot(np.minimum(X, length - X), line), axis=1)
        in_box = (np.abs(Y - width / 2.0) < 20.16) & (np.minimum(X, length - X) < 16.5)
        left = (in_box & (X < length / 2.0)).sum(1)
        right = (in_box & (X >= length / 2.0)).sum(1)
        base["n_box"] = np.maximum(left, right).astype(np.float32)
        for tag, mask in (("H", home), ("A", away)):
            base["mps_" + tag] = np.nanmean(speed[:, mask], axis=1)
            # centroid velocity and position along the HOME attack axis, for both teams
            base["cvu_" + tag] = np.nanmean(vx[:, mask], axis=1) * direction
            cx = np.nanmean(X[:, mask], axis=1)
            base["cu_" + tag] = np.where(direction > 0, cx, length - cx)
            base["sx_" + tag] = np.nanstd(X[:, mask], axis=1)
            base["sy_" + tag] = np.nanstd(Y[:, mask], axis=1)

        # -- ball: visibility, position, kinematics ---------------------------------------------
        visible = np.isfinite(bx) & np.isfinite(by)
        gap_since, gap_until, gap_len = _run_lengths(~visible)
        vis_since, vis_until, _ = _run_lengths(visible)
        base["b_vis"] = visible.astype(np.float32)
        base["gap_since"] = np.minimum(gap_since / FPS, GAP_CAP_S)
        base["gap_until"] = np.minimum(gap_until / FPS, GAP_CAP_S)
        base["gap_len"] = np.minimum(gap_len / FPS, 2 * GAP_CAP_S)
        base["vis_since"] = np.minimum(vis_since / FPS, GAP_CAP_S)
        base["vis_until"] = np.minimum(vis_until / FPS, GAP_CAP_S)

        base["b_edge_x"] = np.minimum(bx, length - bx)     # negative once over a goal line
        base["b_edge_y"] = np.minimum(by, width - by)      # negative once over a touchline
        base["b_u"] = np.where(direction > 0, bx, length - bx)

        bvx = np.full(n, np.nan, np.float32)
        bvy = np.full(n, np.nan, np.float32)
        bvx[1:-1] = (bx[2:] - bx[:-2]) * (FPS / 2.0)
        bvy[1:-1] = (by[2:] - by[:-2]) * (FPS / 2.0)
        b_spd = np.hypot(bvx, bvy)
        base["b_spd"] = b_spd
        base["b_vu"] = bvx * direction
        b_acc = np.full(n, np.nan, np.float32)
        b_acc[2:-2] = np.hypot(bvx[4:] - bvx[:-4], bvy[4:] - bvy[:-4])
        base["b_acc"] = b_acc
        turn = np.full(n, np.nan, np.float32)
        dot = bvx[6:] * bvx[:-6] + bvy[6:] * bvy[:-6]
        norm = b_spd[6:] * b_spd[:-6]
        moving = (b_spd[6:] > 1.0) & (b_spd[:-6] > 1.0)
        turn[3:-3] = np.where(moving, np.arccos(np.clip(dot / np.maximum(norm, 1e-6), -1, 1)), 0.0)
        turn[3:-3][~np.isfinite(norm)] = np.nan
        base["b_turn"] = turn

        # -- ball vs players ----------------------------------------------------------------------
        dist = np.hypot(X - bx[:, None], Y - by[:, None])
        d1 = {}
        for tag, mask in (("H", home), ("A", away)):
            d1[tag], second = _two_smallest(dist, mask)
            base["d1_" + tag] = d1[tag]
            base["d2_" + tag] = second
            base["n4_" + tag] = np.where(visible, (dist[:, mask] < 4.0).sum(1), np.nan)
            base["n10_" + tag] = np.where(visible, (dist[:, mask] < 10.0).sum(1), np.nan)
            # ball speed relative to that team's nearest player: small while a player carries it
            safe = np.where(np.isfinite(dist), dist, np.inf)
            safe[:, ~mask] = np.inf
            nearest = safe.argmin(axis=1)
            rows = np.arange(n)
            rel = np.hypot(bvx - vx[rows, nearest], bvy - vy[rows, nearest])
            rel[~np.isfinite(d1[tag])] = np.nan
            base["rel_" + tag] = rel
        base["dmin"] = np.fmin(d1["H"], d1["A"])
        base["ddiff"] = d1["A"] - d1["H"]                  # > 0: a home player is closer
        both = np.isfinite(d1["H"]) & np.isfinite(d1["A"])
        base["near_H"] = np.where(both, (d1["H"] < d1["A"]).astype(np.float32), np.nan)
        base["ctrl_H"] = np.where(both, ((d1["H"] < CONTROL_RADIUS) & (d1["H"] < d1["A"])), np.nan)
        base["ctrl_A"] = np.where(both, ((d1["A"] < CONTROL_RADIUS) & (d1["A"] < d1["H"])), np.nan)

        # -- touches and closest approaches ----------------------------------------------------------
        base.update(touch_features(b_acc, base["ddiff"], base["dmin"]))
        for w in (6, 15):
            base["b_acc_maxp{}".format(w)] = rolling_extreme(b_acc, -w, -1, np.nanmax)
            base["b_acc_maxf{}".format(w)] = rolling_extreme(b_acc, 1, w, np.nanmax)
        for tag in ("H", "A"):
            base["d1_{}_minp10".format(tag)] = rolling_extreme(d1[tag], -10, -1, np.nanmin)
            base["d1_{}_minf10".format(tag)] = rolling_extreme(d1[tag], 1, 10, np.nanmin)

        # The per-slot arrays are no longer needed; release them before the output is allocated
        # (the development machine has 8 GB and this is the memory peak of the whole pipeline).
        del X, Y, vx, vy, speed, glitch, tracked, dist, safe, line, in_box, rel

        names = list(base)
        windowed = [(key, w) for signals, windows in ((SLOW_SIGNALS, SLOW_WINDOWS),
                                                      (FAST_SIGNALS, FAST_WINDOWS))
                    for key in signals for w in windows]
        out = np.empty((n, len(names) + 2 * len(windowed)), np.float32)
        for j, key in enumerate(names):
            out[:, j] = base[key]
        j = len(names)
        for key, w in windowed:
            series = out[:, names.index(key)].copy()
            out[:, j] = rolling_nanmean(series, -w, -1)
            out[:, j + 1] = rolling_nanmean(series, 1, w)
            j += 2
        names = names + [name for key, w in windowed
                         for name in ("{}_p{}".format(key, w), "{}_f{}".format(key, w))]

    return out, names


# Features computed from the players alone. The dead/alive head is restricted to these: how a
# provider's ball behaves during a stoppage (frozen, carried, lost) is the least transferable
# thing in the data, while twenty-two players slowing to a walk looks the same everywhere.
BALL_FREE_BASE = ("mps", "walk_frac", "run_frac", "stand_frac", "top3_spd", "mps_H", "mps_A",
                  "cvu_H", "cvu_A", "cu_H", "cu_A", "sx_H", "sy_H", "sx_A", "sy_A",
                  "p_line_min", "p_corner_min", "n_box")
BALL_FREE_SIGNALS = tuple(k for k in SLOW_SIGNALS if k != "b_vis")


def ball_free_columns(names):
    """Indices of the features that do not depend on the ball in any way."""
    keep = set(BALL_FREE_BASE)
    for key in BALL_FREE_SIGNALS:
        for w in SLOW_WINDOWS:
            keep.add("{}_p{}".format(key, w))
            keep.add("{}_f{}".format(key, w))
    return [i for i, name in enumerate(names) if name in keep]
