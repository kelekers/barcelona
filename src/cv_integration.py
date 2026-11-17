"""Lightweight pure-NumPy CV artifact integration for Sprint 3, 4, and 5."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import A, D, H

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_BALL_NPZ = CACHE_DIR / "cv_ball_recovery_BAR-ATH.npz"
DEFAULT_JERSEY_NPZ = CACHE_DIR / "cv_jersey_BAR-ATH.npz"
DEFAULT_BOUNDARY_NPZ = CACHE_DIR / "cv_boundary_shifts_BAR-ATH.npz"
DEFAULT_WEIGHTS = ROOT / "models" / "ball_yolo26s_stage_b.pt"


def ensure_pre_decode_cv(
    rebuild: bool = False,
    ball_npz: Path = DEFAULT_BALL_NPZ,
    jersey_npz: Path = DEFAULT_JERSEY_NPZ,
    weights_path: Path = DEFAULT_WEIGHTS,
) -> tuple[Path, Path]:
    """Ensure Sprint 3 ball recovery and Sprint 4 jersey swap caches exist (build lazily if needed)."""
    if rebuild or not ball_npz.exists():
        if not weights_path.exists():
            raise FileNotFoundError(f"CV ball detector weights missing: {weights_path}")
        print("[CV Integration] Running Sprint 3 YOLO26s ball recovery (--full)...")
        cmd = [sys.executable, str(ROOT / "src" / "cv_recover_ball.py"), "--full", "--weights", str(weights_path)]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    if rebuild or not jersey_npz.exists():
        print("[CV Integration] Running Sprint 4 self-supervised jersey audit...")
        cmd = [sys.executable, str(ROOT / "src" / "cv_jersey.py")]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    return ball_npz, jersey_npz


def apply_pre_decode_cv(match: dict, ball_npz: Path = DEFAULT_BALL_NPZ, jersey_npz: Path = DEFAULT_JERSEY_NPZ) -> dict:
    """Inject Sprint 3 recovered ball coordinates and Sprint 4 isolated tracklet-swap repairs."""
    out = dict(match)
    if ball_npz.exists():
        d = np.load(ball_npz)
        bx = out["ball_x"].astype(np.float32).copy()
        by = out["ball_y"].astype(np.float32).copy()
        fill = d["recovered_mask"] & (~np.isfinite(bx))
        bx[fill] = d["ball_x"][fill]
        by[fill] = d["ball_y"][fill]
        out["ball_x"] = bx
        out["ball_y"] = by
        print(f"[CV S3] filled {int(fill.sum())} ball-NaN frames")
    if jersey_npz.exists():
        swap_mask = np.load(jersey_npz)["swap_mask"]
        n_frames, n_slots = out["px"].shape
        if swap_mask.shape == (n_frames, n_slots) and swap_mask.any():
            px = out["px"].astype(np.float32).copy()
            py = out["py"].astype(np.float32).copy()
            tm = out["team_map"].copy()
            extra_x = np.full((n_frames, 4), np.nan, dtype=np.float32)
            extra_y = np.full((n_frames, 4), np.nan, dtype=np.float32)
            moved = 0
            for s in range(n_slots):
                rows = np.flatnonzero(swap_mask[:, s])
                if len(rows) == 0:
                    continue
                target_team = 1 - int(tm[s])
                cands = (0, 1) if target_team == 0 else (2, 3)
                for r in rows:
                    for c_idx in cands:
                        if not np.isfinite(extra_x[r, c_idx]):
                            extra_x[r, c_idx] = px[r, s]
                            extra_y[r, c_idx] = py[r, s]
                            break
                    px[r, s], py[r, s] = np.nan, np.nan
                    moved += 1
            out["px"] = np.concatenate([px, extra_x], axis=1)
            out["py"] = np.concatenate([py, extra_y], axis=1)
            out["team_map"] = np.concatenate([tm, np.array([0, 0, 1, 1], dtype=tm.dtype)])
            print(f"[CV S4] reassigned {moved} swapped tracklet player-frames")
    return out


def apply_boundary_motion_cv(
    codes: np.ndarray,
    submission_csv_path: str,
    rebuild: bool = False,
    boundary_npz: Path = DEFAULT_BOUNDARY_NPZ,
    weights_path: Path = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """Apply Sprint 5 visual kick-onset snapping at D -> Live boundaries."""
    if rebuild or not boundary_npz.exists():
        if not weights_path.exists():
            raise FileNotFoundError(f"CV ball detector weights missing: {weights_path}")
        print("[CV Integration] Running Sprint 5 visual kick-onset boundary check...")
        cmd = [
            sys.executable,
            str(ROOT / "src" / "cv_boundary_motion.py"),
            "--in-csv",
            os.path.abspath(submission_csv_path),
            "--out-npz",
            str(boundary_npz),
            "--weights",
            str(weights_path),
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    if not boundary_npz.exists():
        return codes

    raw_json = str(np.load(boundary_npz)["shifts_json"])
    shifts = json.loads(raw_json)
    out = codes.copy()
    applied = 0
    for item in shifts:
        t_tr = int(item["t_trans"])
        t_snap = int(item["t_snap"])
        delta = t_snap - t_tr
        if 0 < t_tr < len(out) and 0 < t_snap < len(out) and out[t_tr] in (H, A) and out[t_tr - 1] == D:
            if delta > 0:
                out[t_tr:t_snap] = D
            elif delta < 0:
                out[t_snap:t_tr] = out[t_tr]
            applied += 1
    print(f"[CV S5] aligned {applied} D->Live ground-kick boundaries")
    return out
