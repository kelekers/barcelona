"""Smart Tagging as side information for the adaptation student (adapt.py).

EXPERIMENTAL -- NOT PART OF THE SUBMISSION: only reached through `run_pipeline.py --adapt`, and
the adaptation step did not survive validation on external games.

The tags are automatic and exist only for the competition match, so they cannot be part of the
external model, and nobody can tell from other games how much to trust them. They are therefore
never turned into labels and never given a hand-set weight. They enter as per-frame features of
the STUDENT, which is trained on the external model's pseudo-labels: whatever relation the tags
have to possession on this match is learned from the match itself, and a tag that carries no
information simply goes unused.

Per frame:
  * seconds since / until the nearest SET PIECES tag (start and end), and whether one is open
  * seconds since the last GOALS tag and until the next kick-off tag (celebration stretches)
  * for each team: how many attacking-phase tags and how many defending-phase tags are open

Built twice -- as tagged, and with the two teams exchanged -- because the student is trained on
both team orientations.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .common import FPS, ROOT

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, require, source_frames  # noqa: E402

ATTACK_CODES = {"build up", "progression", "creating chances", "attacking transition",
                "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFEND_CODES = {"defending in attacking third", "defending in middle third",
                "defending in defensive third", "defensive transition"}
CAP_S = 30.0
LONG_CAP_S = 120.0

NAMES = ("sp_start_since", "sp_start_until", "sp_end_since", "sp_end_until", "sp_open",
         "goal_since", "kickoff_until", "attack_H", "attack_A", "defend_H", "defend_A")
SWAP = {"attack_H": "attack_A", "attack_A": "attack_H", "defend_H": "defend_A", "defend_A": "defend_H"}


def _since_until(marks, n, cap):
    """Seconds since the last mark and until the next one, capped; `cap` where there is none."""
    marks = np.unique(np.clip(np.asarray(marks, np.int64), 0, n - 1))
    index = np.arange(n)
    flag = np.zeros(n, bool)
    flag[marks] = True
    last = np.maximum.accumulate(np.where(flag, index, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(flag, index, 10 ** 9)[::-1])[::-1]
    since = np.minimum((index - last) / FPS, cap).astype(np.float32)
    until = np.minimum((nxt - index) / FPS, cap).astype(np.float32)
    return since, until


def tag_features(game=GAME, data_dir=None):
    """(forward [n, k], mirrored [n, k], names). Empty tags never raise: they give constants."""
    meta_path, tag_path = require(["atd_meta", "smart_tagging"], game, data_dir)
    n = source_frames("tactical", game)
    home_name = [team.findtext("Name") for team in ET.parse(meta_path).getroot().iter("Team")][0]

    columns = {name: np.zeros(n, np.float32) for name in NAMES}
    sp_starts, sp_ends, goals, kickoffs = [], [], [], []
    for instance in ET.parse(tag_path).iter("instance"):
        code = (instance.findtext("code") or "").strip().lower()
        first = int(round(float(instance.findtext("start")) * FPS))
        stop = int(round(float(instance.findtext("end")) * FPS))
        tags = {label.findtext("group"): (label.findtext("text") or "").strip()
                for label in instance.findall("label")}
        first, stop = max(0, min(n, first)), max(0, min(n, stop))
        team = tags.get("Team", "")
        side = None if team in ("", "N/A") else ("H" if team == home_name else "A")
        if code == "set pieces":
            sp_starts.append(first)
            sp_ends.append(stop)
            columns["sp_open"][first:stop] = 1.0
            if tags.get("Type", "").lower().startswith("kick off"):
                kickoffs.append(first)
        elif code == "goals":
            goals.append(first)
        elif side is not None and code in ATTACK_CODES:
            columns["attack_" + side][first:stop] += 1.0
        elif side is not None and code in DEFEND_CODES:
            columns["defend_" + side][first:stop] += 1.0

    columns["sp_start_since"], columns["sp_start_until"] = _since_until(sp_starts, n, CAP_S) \
        if sp_starts else (np.full(n, CAP_S, np.float32),) * 2
    columns["sp_end_since"], columns["sp_end_until"] = _since_until(sp_ends, n, CAP_S) \
        if sp_ends else (np.full(n, CAP_S, np.float32),) * 2
    columns["goal_since"] = _since_until(goals, n, LONG_CAP_S)[0] if goals \
        else np.full(n, LONG_CAP_S, np.float32)
    columns["kickoff_until"] = _since_until(kickoffs, n, LONG_CAP_S)[1] if kickoffs \
        else np.full(n, LONG_CAP_S, np.float32)

    forward = np.stack([columns[name] for name in NAMES], axis=1)
    mirrored = np.stack([columns[SWAP.get(name, name)] for name in NAMES], axis=1)
    return forward, mirrored, list(NAMES)
