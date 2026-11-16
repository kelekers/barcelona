"""Loader for the IDSSE / DFL open dataset (7 Bundesliga matches, official 25 Hz positions).

The ball FrameSet of the DFL position file carries, on every frame, ``BallStatus`` (1 alive,
0 dead) and ``BallPossession`` (1 home, 2 guest). Those two columns ARE the three-state label of
this challenge, produced by the league's data provider on other games:

    D  where BallStatus == 0
    H  where BallStatus == 1 and BallPossession == 1
    A  where BallStatus == 1 and BallPossession == 2

Unlike the Metrica sample games the ball is tracked while it is dead too, so a model trained here
can learn what a dead ball looks like (out of bounds, stationary at a restart spot) instead of
equating "no ball" with "dead".
"""

import os
import re

import numpy as np

from .common import A, D, EXTERNAL_DIR, FPS, H
from .download_external import DFL_MATCHES

DFL_DIR = os.path.join(EXTERNAL_DIR, "dfl")
MATCH_IDS = tuple(sorted(DFL_MATCHES))

_ATTR = {name: re.compile(rb' ' + name + rb'="([^"]*)"')
         for name in (b"N", b"X", b"Y", b"BallPossession", b"BallStatus")}
_HEAD = {name: re.compile(name + rb'="([^"]*)"') for name in (b"GameSection", b"TeamId", b"PersonId")}


def _paths(match_id):
    comp = DFL_MATCHES[match_id]
    return (
        os.path.join(DFL_DIR, "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml".format(comp, match_id)),
        os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(comp, match_id)),
        os.path.join(DFL_DIR, "cache_{}.npz".format(match_id)),
    )


def _column(block, name, dtype):
    return np.array(_ATTR[name].findall(block), dtype="S").astype(dtype)


def _parse(match_id):
    info_path, pos_path, _ = _paths(match_id)
    with open(info_path, "rb") as handle:
        info = handle.read()
    home_id = re.search(rb'HomeTeamId="([^"]*)"', info).group(1)
    guest_id = re.search(rb'GuestTeamId="([^"]*)"', info).group(1)
    length = float(re.search(rb'PitchX="([^"]*)"', info).group(1))
    width = float(re.search(rb'PitchY="([^"]*)"', info).group(1))

    # Streamed one FrameSet at a time: the file is ~400 MB and one FrameSet is ~15 MB.
    players, ball = [], {}
    head, lines = None, []
    with open(pos_path, "rb") as handle:
        for line in handle:
            if head is None:
                if line.startswith(b"<FrameSet"):
                    head, lines = line, []
                continue
            if not line.startswith(b"</FrameSet"):
                lines.append(line)
                continue
            block = b"".join(lines)
            section = _HEAD[b"GameSection"].search(head).group(1)
            team = _HEAD[b"TeamId"].search(head).group(1)
            person = _HEAD[b"PersonId"].search(head).group(1)
            head, lines = None, []
            if section not in (b"firstHalf", b"secondHalf"):
                continue
            is_ball = b"BallStatus" in block[:400]
            if team not in (home_id, guest_id) and not is_ball:
                continue  # referees
            n = _column(block, b"N", np.int64)
            x = _column(block, b"X", np.float32)
            y = _column(block, b"Y", np.float32)
            if is_ball:
                ball[section] = (n, x, y, _column(block, b"BallPossession", np.int8),
                                 _column(block, b"BallStatus", np.int8))
            else:
                players.append((section, 0 if team == home_id else 1, person, n, x, y))

    # One contiguous timeline: first half then second half, indexed by the ball's frame counter.
    spans, offset = {}, 0
    for section in (b"firstHalf", b"secondHalf"):
        n = ball[section][0]
        spans[section] = (int(n.min()), int(n.max()), offset)
        offset += int(n.max()) - int(n.min()) + 1
    n_frames = offset

    slot_of = {}
    for _, team, person, _, _, _ in players:
        slot_of.setdefault((team, person), len(slot_of))
    px = np.full((n_frames, len(slot_of)), np.nan, np.float32)
    py = np.full((n_frames, len(slot_of)), np.nan, np.float32)
    team_map = np.zeros(len(slot_of), np.int8)
    for section, team, person, n, x, y in players:
        lo, hi, off = spans[section]
        keep = (n >= lo) & (n <= hi)
        rows = n[keep] - lo + off
        slot = slot_of[(team, person)]
        team_map[slot] = team
        px[rows, slot] = x[keep] / length + 0.5
        py[rows, slot] = y[keep] / width + 0.5

    ball_x = np.full(n_frames, np.nan, np.float32)
    ball_y = np.full(n_frames, np.nan, np.float32)
    labels = np.full(n_frames, D, np.int8)
    half_blocks = []
    for half, section in ((1, b"firstHalf"), (2, b"secondHalf")):
        n, x, y, possession, status = ball[section]
        lo, hi, off = spans[section]
        rows = n - lo + off
        ball_x[rows] = x / length + 0.5
        ball_y[rows] = y / width + 0.5
        labels[rows] = np.where(status == 0, D, np.where(possession == 1, H, A))
        half_blocks.append((half, off, off + hi - lo))

    return {"n_frames": n_frames, "pitch": np.array([length, width]), "team_map": team_map,
            "px": px, "py": py, "ball_x": ball_x, "ball_y": ball_y, "labels": labels,
            "half_blocks": np.array(half_blocks, np.int64)}


def load_dfl_match(match_id):
    """Match dictionary (see common.py) for one DFL open match. Parsed once, then cached."""
    cache = _paths(match_id)[2]
    if os.path.exists(cache):
        z = np.load(cache)
        out = {k: z[k] for k in z.files}
    else:
        out = _parse(match_id)
        np.savez_compressed(cache, **out)
    out["n_frames"] = int(out["n_frames"])
    out["pitch"] = (float(out["pitch"][0]), float(out["pitch"][1]))
    out["half_blocks"] = [tuple(int(v) for v in row) for row in out["half_blocks"]]
    out["game"] = "DFL-" + match_id
    out["fps"] = FPS
    # This provider parks the ball at a fixed point during stoppages; degrade.py replaces it.
    out["dead_ball_frozen"] = True
    return out
