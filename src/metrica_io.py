"""Metrica Sports open sample data (github.com/metrica-sports/sample-data) loader.

Ported from this team's ``trio`` branch (same event -> label conversion, so validation numbers
are comparable), adapted to the folder layout written by download_external.py.

Produces the common match dictionary (see common.py) plus frame labels H/A/D/X derived
deterministically from Metrica's human-produced event log. These are labels on OTHER games, which
RULES.md explicitly allows. In this branch the Metrica games are the VALIDATION set: the model is
trained on a different provider's matches (DFL) and scored here.

Label derivation (fully deterministic, no person decides any frame):
  * Possession team at frame t = team of the most recent on-ball event (PASS, RECOVERY,
    BALL LOST, SET PIECE, SHOT, BALL OUT, CHALLENGE-WON) whose start frame <= t.
  * Dead ball = [stop, next SET PIECE start) where stop is the end frame of a BALL OUT, a SHOT
    that went out / was a goal, the start frame of a FAULT RECEIVED, or -- for set pieces with
    none of these -- the end frame of the last preceding event.
    Before the first kick-off of each period is dead; frames outside periods are X.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .common import A, D, EXTERNAL_DIR, FPS, H, X  # noqa: F401

METRICA_DIR = os.path.join(EXTERNAL_DIR, "metrica")
PITCH = (105.0, 68.0)

CONTROL_TYPES = {"PASS", "RECOVERY", "BALL LOST", "SET PIECE", "SHOT", "BALL OUT"}


# ---------------------------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------------------------

def _read_csv_tracking(path):
    """Metrica CSV tracking (games 1, 2): returns period, frame, (n, k) x, (n, k) y, ball x/y."""
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    period = raw["Period"].values.astype(np.int64)
    frame = raw["Frame"].values.astype(np.int64)
    xs, ys = [], []
    ball_x = ball_y = None
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            xs.append(raw.iloc[:, i].values.astype(np.float32))
            ys.append(raw.iloc[:, i + 1].values.astype(np.float32))
        elif c == "Ball":
            ball_x = raw.iloc[:, i].values.astype(np.float32)
            ball_y = raw.iloc[:, i + 1].values.astype(np.float32)
    return period, frame, np.stack(xs, 1), np.stack(ys, 1), ball_x, ball_y


def _mpath(game_no, filename):
    nested = os.path.join(METRICA_DIR, "Sample_Game_{}".format(game_no), filename)
    return nested if os.path.exists(nested) else os.path.join(METRICA_DIR, filename)


def _load_game_csv(game_no):
    prefix = "Sample_Game_{}".format(game_no)
    per_h, fr_h, xh, yh, bx, by = _read_csv_tracking(_mpath(game_no, prefix + "_RawTrackingData_Home_Team.csv"))
    per_a, fr_a, xa, ya, _, _ = _read_csv_tracking(_mpath(game_no, prefix + "_RawTrackingData_Away_Team.csv"))
    assert np.array_equal(fr_h, fr_a)
    px = np.concatenate([xh, xa], axis=1)
    py = np.concatenate([yh, ya], axis=1)
    team_map = np.array([0] * xh.shape[1] + [1] * xa.shape[1], dtype=np.int8)
    return per_h, fr_h, px, py, bx, by, team_map


def _load_game3():
    meta = ET.parse(_mpath(3, "Sample_Game_3_metadata.xml")).getroot()
    teams = [t.get("id") for t in meta.iter("Team")]
    home_id = teams[0]
    team_of_player = {p.get("id"): (0 if p.get("teamId") == home_id else 1) for p in meta.iter("Player")}
    chan_player = {pc.get("id"): pc.get("playerId") for pc in meta.iter("PlayerChannel")}

    # One DataFormatSpecification per substitution window; map each to global player slots.
    all_players = []
    specs = []
    for dfs in meta.iter("DataFormatSpecification"):
        order = []
        for pc in dfs.iter("PlayerChannelRef"):
            pid = chan_player[pc.get("playerChannelId")]
            if pid not in order:
                order.append(pid)
        for pid in order:
            if pid not in all_players:
                all_players.append(pid)
        specs.append((int(dfs.get("startFrame")), int(dfs.get("endFrame")), order))
    slot = {pid: k for k, pid in enumerate(all_players)}
    specs = [(s, e, np.array([slot[p] for p in order])) for s, e, order in specs]
    team_map = np.array([team_of_player[p] for p in all_players], dtype=np.int8)
    n_slots = len(all_players)

    params = {}
    for pp in meta.iter("ProviderParameter"):
        params[pp.findtext("Name")] = pp.findtext("Value")
    half_ranges = [(1, int(params["first_half_start"]), int(params["first_half_end"])),
                   (2, int(params["second_half_start"]), int(params["second_half_end"]))]

    with open(_mpath(3, "Sample_Game_3_tracking.txt")) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln]
    n = len(lines)
    frame = np.empty(n, dtype=np.int64)
    px = np.full((n, n_slots), np.nan, np.float32)
    py = np.full((n, n_slots), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    for i, line in enumerate(lines):
        f_str, p_str, b_str = line.split(":")
        fr = int(f_str)
        frame[i] = fr
        slots = specs[-1][2]
        for s, e, sl in specs:
            if s <= fr <= e:
                slots = sl
                break
        vals = np.array(p_str.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 2)
        k = min(len(slots), len(vals))
        px[i, slots[:k]] = vals[:k, 0]
        py[i, slots[:k]] = vals[:k, 1]
        bv = b_str.split(",")
        bx[i] = float(bv[0])
        by[i] = float(bv[1])
    period = np.zeros(n, dtype=np.int64)
    for per, s, e in half_ranges:
        period[(frame >= s) & (frame <= e)] = per
    return period, frame, px, py, bx, by, team_map


# ---------------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------------

def _events_csv(game_no):
    ev = pd.read_csv(_mpath(game_no, "Sample_Game_{}_RawEventsData.csv".format(game_no)))
    out = []
    for r in ev.itertuples(index=False):
        out.append({
            "team": 0 if r[0] == "Home" else 1,
            "type": str(r[1]).upper(),
            "subtype": "" if pd.isna(r[2]) else str(r[2]).upper(),
            "period": int(r[3]),
            "start": int(r[4]),
            "end": int(r[6]),
        })
    return out


def _events_json():
    with open(_mpath(3, "Sample_Game_3_events.json")) as fh:
        js = json.load(fh)
    out = []
    for e in js["data"]:
        team_name = (e.get("team") or {}).get("name", "")
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s.get("name", "") for s in sub)
        elif isinstance(sub, dict):
            sub = sub.get("name", "")
        else:
            sub = ""
        out.append({
            "team_name": team_name,
            "type": (e.get("type") or {}).get("name", "").upper(),
            "subtype": (sub or "").upper(),
            "period": int(e.get("period") or 0),
            "start": int((e.get("start") or {}).get("frame") or 0),
            "end": int((e.get("end") or {}).get("frame") or 0),
        })
    return out


def events_to_labels(events, frame, period):
    """Deterministic event -> frame label conversion (see module docstring)."""
    n = len(frame)
    f0 = int(frame[0])
    lab = np.full(n, X, dtype=np.int8)

    def idx(fr):
        return int(np.clip(fr - f0, 0, n - 1))

    for per in sorted(set(int(p) for p in np.unique(period) if p > 0)):
        pev = sorted([e for e in events if e["period"] == per], key=lambda e: (e["start"], e["end"]))
        if not pev:
            continue
        pmask = np.flatnonzero(period == per)
        p_s, p_e = int(pmask[0]), int(pmask[-1]) + 1
        seg = np.full(p_e - p_s, D, dtype=np.int8)

        # 1. Possession team from the latest control event
        marks = []
        for e in pev:
            is_ctrl = (
                e["type"] in CONTROL_TYPES
                or e["type"] == "CARRY"
                or (e["type"] == "CHALLENGE" and "WON" in e["subtype"] and "FAULT" not in e["subtype"])
            )
            if is_ctrl:
                marks.append((idx(e["start"]), e["team"]))
        first_ko = min(
            (idx(e["start"]) for e in pev if e["type"] == "SET PIECE" and "KICK OFF" in e["subtype"]),
            default=min((idx(e["start"]) for e in pev if e["type"] == "SET PIECE"), default=p_s),
        )
        marks.sort()
        for k, (t, tm) in enumerate(marks):
            t_next = marks[k + 1][0] if k + 1 < len(marks) else p_e
            a, b = max(t, p_s), min(t_next, p_e)
            if b > a:
                seg[a - p_s:b - p_s] = tm
        last_ev_end = max(idx(max(e["end"], e["start"])) for e in pev)
        seg[:max(0, first_ko - p_s)] = D
        seg[max(0, last_ev_end + 1 - p_s):] = D

        # 2a. Forward dead spans from explicit stoppage events (BALL OUT, SHOT OUT/GOAL, END HALF)
        for k, e in enumerate(pev):
            sub_toks = set(e["subtype"].replace("-", " ").split())
            is_ball_out = e["type"] == "BALL OUT"
            is_shot_dead = e["type"] == "SHOT" and ("OUT" in sub_toks or "GOAL" in sub_toks)
            is_end_half = "END HALF" in e["subtype"]
            if not (is_ball_out or is_shot_dead or is_end_half):
                continue
            stop = idx(max(e["end"], e["start"]))
            t_next = p_e
            if not is_end_half:
                for m in range(k + 1, len(pev)):
                    em = pev[m]
                    if em["type"] in ("CARD", "CARRY"):
                        continue
                    if em["type"] == "CHALLENGE" and em["start"] <= e["end"] + 3:
                        continue
                    if em["type"] == "BALL LOST" and em["start"] <= e["end"] + 3:
                        continue
                    tm_next = idx(em["start"])
                    if tm_next > stop:
                        t_next = tm_next
                        break
            a, b = max(p_s, stop), min(p_e, t_next)
            if b > a:
                seg[a - p_s:b - p_s] = D

        # 2b. Dead spans ending at each SET PIECE or GOAL KICK
        sp_idx = [
            k for k, e in enumerate(pev)
            if e["type"] == "SET PIECE" or ("GOAL KICK" in e["subtype"] and e["type"] != "BALL OUT")
        ]
        for k in sp_idx:
            e_sp = pev[k]
            t_sp = idx(e_sp["start"])
            t_restart = t_sp
            if e_sp["type"] == "SET PIECE":
                for m in range(k + 1, min(len(pev), k + 4)):
                    em = pev[m]
                    if em["type"] in ("PASS", "BALL LOST", "SHOT"):
                        cand = idx(em["start"])
                        if cand <= t_sp + 60:
                            t_restart = max(t_restart, cand)
                        break
                    if idx(em["start"]) > t_sp + 5:
                        break
            else:
                # Overlapping PASS / BALL LOST both tagged GOAL KICK on the same kick: skip second.
                if k > 0 and "GOAL KICK" in pev[k - 1]["subtype"] and idx(pev[k - 1]["end"]) >= t_sp - 10:
                    continue

            stop = None
            prev_end = None
            for j in range(k - 1, -1, -1):
                e = pev[j]
                if e["type"] in ("CARD", "CARRY"):
                    continue
                if e["type"] == "SET PIECE" or "GOAL KICK" in e["subtype"]:
                    break
                if idx(e["start"]) >= t_sp:
                    continue
                e_end = idx(max(e["end"], e["start"]))
                sub_toks = set(e["subtype"].replace("-", " ").split())
                if e["type"] == "BALL OUT" or (e["type"] == "SHOT" and ("OUT" in sub_toks or "GOAL" in sub_toks)):
                    if prev_end is None or e_end >= prev_end - 5:
                        stop = e_end
                    break
                if e["type"] == "FAULT RECEIVED" or "FAULT" in e["subtype"]:
                    if prev_end is None or idx(e["start"]) >= prev_end - 5:
                        stop = idx(e["start"])
                    break
                if e["type"] == "CHALLENGE":
                    continue
                # Live event (PASS, BALL LOST, SHOT, RECOVERY): bound stoppage unless an explicit
                # BALL OUT/FAULT coincides within 5 frames of this live event.
                if prev_end is None:
                    prev_end = min(e_end, t_sp)
                elif e_end < prev_end - 5:
                    break
            if stop is None:
                stop = prev_end if prev_end is not None else t_sp
            stop = max(p_s, min(stop, t_restart))
            seg[stop - p_s:t_restart - p_s] = D
        lab[p_s:p_e] = seg
    return lab


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------

def load_metrica_game(game_no):
    """Return a dict compatible with the BAR-ATH loader plus labels."""
    if game_no in (1, 2):
        period, frame, px, py, bx, by, team_map = _load_game_csv(game_no)
        events = _events_csv(game_no)
    else:
        period, frame, px, py, bx, by, team_map = _load_game3()
        events = _events_json()
        # Metadata lists FIFATMA ("Team A") first -> home slot team 0.
        for e in events:
            e["team"] = 0 if e["team_name"] == "Team A" else 1

    labels = events_to_labels(events, frame, period)
    # Half blocks [(half, first, last)] over the period frames
    half_blocks = []
    for per in sorted(set(int(p) for p in np.unique(period) if p > 0)):
        m = np.flatnonzero(period == per)
        half_blocks.append((per, int(m[0]), int(m[-1])))
    return {
        "game": "METRICA-{}".format(game_no),
        "n_frames": len(frame),
        "fps": FPS,
        "pitch": PITCH,
        "team_map": team_map,
        "half_blocks": half_blocks,
        "ball_x": bx,
        "ball_y": by,
        "px": px,
        "py": py,
        "ps": None,
        "labels": labels,
        "events": events,
    }
