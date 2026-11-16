"""Fast streaming ATD loader, Smart Tagging loader, and ball glitch cleaner.

Pure standard library + NumPy, keeping the pipeline dependency-free beyond
requirements.txt and deterministic across Python versions.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import team_names, team_of_each_track  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"


def load_atd_arrays(game=GAME, data_dir=None, source="tactical"):
    """Stream raw ATD and metadata into aligned float32 NumPy arrays.

    Returns a dict with:
        n_frames: int
        fps: int
        pitch: (length_m, width_m)
        team_map: (n_slots,) int8 (0=Home, 1=Away)
        team_names: (home_name, away_name)
        half_blocks: [(half, first_vf, last_vf), ...]
        ball_x, ball_y, ball_s: (n_frames,) float32 (NaN when absent)
        px, py, ps: (n_frames, n_slots) float32 (NaN when absent)
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    team_map = team_of_each_track(meta_path)
    n_slots = len(team_map)

    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_s = np.full(n_frames, np.nan, dtype=np.float32)
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            idx = int(parts[0]) - 1  # ATD frame_id is 1-based; video_frame is 0-based
            if idx < 0 or idx >= n_frames:
                continue

            ball_str = parts[2]
            if ball_str and ball_str[0] != MISSING_CHAR:
                c1 = ball_str.index(",")
                c2 = ball_str.index(",", c1 + 1)
                ball_x[idx] = float(ball_str[:c1])
                ball_y[idx] = float(ball_str[c1 + 1:c2])
                s_val = ball_str[c2 + 1:]
                if s_val and s_val[0] != MISSING_CHAR:
                    ball_s[idx] = float(s_val)

            for slot, grp in enumerate(parts[1].split(GROUP_SEP)):
                if not grp or grp[0] == MISSING_CHAR:
                    continue
                c1 = grp.index(",")
                c2 = grp.index(",", c1 + 1)
                px[idx, slot] = float(grp[:c1])
                py[idx, slot] = float(grp[c1 + 1:c2])
                s_val = grp[c2 + 1:]
                if s_val and s_val[0] != MISSING_CHAR:
                    ps[idx, slot] = float(s_val)

    return {
        "game": game,
        "n_frames": n_frames,
        "fps": GAMES[game]["fps"],
        "pitch": GAMES[game]["pitch"],
        "team_map": team_map,
        "team_names": team_names(meta_path),
        "half_blocks": half_video_blocks(game),
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_s": ball_s,
        "px": px,
        "py": py,
        "ps": ps,
    }


def clean_ball_track(ball_x, ball_y, pitch, max_step_m=2.2, max_return_m=2.5):
    """Suppress impossible 1-2 frame ball tracker glitches and tiny isolated islands.

    Physical motivation: at 25 fps, 2.2 m/frame corresponds to 55 m/s (~198 km/h),
    above realistic football speeds. Tracker false detections (e.g., snapping to
    a player's head/sock for 1-2 frames) jump far away and jump right back.
    """
    bx = ball_x.copy()
    by = ball_y.copy()
    length, width = pitch
    n = len(bx)

    # 1. Remove 1-frame and 2-frame outlier spikes where neighbors agree
    for span in (1, 2):
        if n <= span + 2:
            continue
        for t in range(1, n - span):
            if np.isnan(bx[t]) or np.isnan(bx[t - 1]) or np.isnan(bx[t + span]):
                continue
            d_before = np.hypot((bx[t] - bx[t - 1]) * length, (by[t] - by[t - 1]) * width)
            d_across = np.hypot((bx[t + span] - bx[t - 1]) * length, (by[t + span] - by[t - 1]) * width)
            if d_before > max_step_m and d_across <= max_return_m * span:
                bx[t:t + span] = np.nan
                by[t:t + span] = np.nan

    # 2. Remove tiny isolated islands (<= 4 frames) surrounded by >= 25 frames of NaN
    #    if they jump > 12 m from both surrounding visible anchors.
    vis = ~np.isnan(bx)
    edges = np.diff(np.concatenate(([0], vis.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for idx_seg in range(len(starts)):
        a, b = starts[idx_seg], ends[idx_seg]
        if b - a <= 4:
            left_gap = a - ends[idx_seg - 1] if idx_seg > 0 else 999
            right_gap = starts[idx_seg + 1] - b if idx_seg + 1 < len(starts) else 999
            if left_gap >= 25 and right_gap >= 25:
                far_left = far_right = True
                if idx_seg > 0:
                    p_idx = ends[idx_seg - 1] - 1
                    far_left = np.hypot((bx[a] - bx[p_idx]) * length,
                                        (by[a] - by[p_idx]) * width) > 12.0
                if idx_seg + 1 < len(starts):
                    n_idx = starts[idx_seg + 1]
                    far_right = np.hypot((bx[b - 1] - bx[n_idx]) * length,
                                         (by[b - 1] - by[n_idx]) * width) > 12.0
                if far_left and far_right:
                    bx[a:b] = np.nan
                    by[a:b] = np.nan

    return bx, by


def load_match(game=GAME, data_dir=None):
    """Load tactical ATD (with cleaned ball track) and Smart Tagging events."""
    data = load_atd_arrays(game=game, data_dir=data_dir, source="tactical")
    bx_clean, by_clean = clean_ball_track(data["ball_x"], data["ball_y"], data["pitch"])
    data["ball_x_raw"] = data["ball_x"]
    data["ball_y_raw"] = data["ball_y"]
    data["ball_x"] = bx_clean
    data["ball_y"] = by_clean
    data["events"] = load_smart_tagging(game=game, data_dir=data_dir, fps=data["fps"])
    return data
