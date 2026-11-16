"""Loader for the competition's automatic tracking data (ATD), tactical recording.

Reads the raw FIFA/EPTS text export with numpy only and returns the common match dictionary
(see common.py). The only processing applied is the removal of obvious ball-tracker glitches:
single-frame teleports and tiny isolated detections far from the rest of the trajectory. Nothing
is corrected by hand and no frame-specific constant is used.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .common import FPS, ROOT

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# A ball cannot move 3 m between two consecutive frames (75 m/s); a detection that does, and is
# back on the previous trajectory one or two frames later, is a tracker snap.
SPIKE_JUMP_M = 3.0
# A run of at most this many detections with long blanks either side is an isolated blip.
ISLAND_MAX_FRAMES = 4
ISLAND_MIN_BLANK = 25


def _team_map(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if s == team_ids[0] else 1 for s in slots], dtype=np.int8)


def _despike(bx, by, pitch):
    """NaN-out teleports and isolated blips. Returns cleaned copies and the number removed."""
    length, width = pitch
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    bad = np.zeros(len(x), bool)

    # 1-2 frame snaps: far from the previous point, and the trajectory resumes afterwards.
    for span in (1, 2):
        step_in = np.hypot(x[span:-span] - x[:-2 * span], y[span:-span] - y[:-2 * span])
        bridge = np.hypot(x[2 * span:] - x[:-2 * span], y[2 * span:] - y[:-2 * span])
        snap = (step_in > SPIKE_JUMP_M * span) & (bridge < SPIKE_JUMP_M * span)
        bad[span:-span] |= snap

    visible = np.isfinite(x) & ~bad
    edges = np.diff(np.concatenate(([0], visible.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for i, (a, b) in enumerate(zip(starts, stops)):
        if b - a > ISLAND_MAX_FRAMES:
            continue
        blank_before = a - (stops[i - 1] if i else 0)
        blank_after = (starts[i + 1] if i + 1 < len(starts) else len(x)) - b
        if blank_before >= ISLAND_MIN_BLANK and blank_after >= ISLAND_MIN_BLANK:
            bad[a:b] = True

    out_x, out_y = bx.copy(), by.copy()
    out_x[bad], out_y[bad] = np.nan, np.nan
    return out_x, out_y, int(bad.sum())


def load_atd(game=GAME, data_dir=None, verbose=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team_map = _team_map(meta_path)
    n_slots = len(team_map)

    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            values = values.reshape(n_slots, 3)
            px[i], py[i] = values[:, 0], values[:, 1]
            ball[i] = np.array(parts[2].split(",")[:2], dtype=np.float32)

    pitch = tuple(GAMES[game]["pitch"])
    ball_x, ball_y, removed = _despike(ball[:, 0], ball[:, 1], pitch)
    if verbose:
        print("  ATD: {} frames, {} slots, ball on {:.1f}% of frames, {} glitch frames removed".format(
            n_frames, n_slots, 100.0 * np.isfinite(ball_x).mean(), removed))
    return {"game": game, "n_frames": n_frames, "fps": FPS, "pitch": pitch, "team_map": team_map,
            "px": px, "py": py, "ball_x": ball_x, "ball_y": ball_y, "labels": None,
            "half_blocks": [tuple(b) for b in half_video_blocks(game)]}
