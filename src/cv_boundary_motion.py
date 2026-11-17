r"""Sprint 5: Visual Ball-Motion Kick-Onset Check at D <-> Live Boundaries.

At every D -> Live restart boundary (where dead_head lags/leads because dead_head is ball-free),
runs unsmoothed frame-by-frame (stride=1) YOLO26s Stage-B 1-tile ROI tracking over
[t_trans - 8 .. t_trans + 34] to locate the exact physical frame `t_kick` when a stationary
pitch ball launches into play.

Zero human labels or GT peeking. Outputs refined predictions + visual audit frames under
`data/cache/cv_runs/boundary_preview/`.

Usage:
    .\.venv\Scripts\python.exe -u src/cv_boundary_motion.py --in-csv data/cache/eval_s3/hybrid_s3.csv --out-npz data/cache/cv_boundary_shifts_BAR-ATH.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

warnings.filterwarnings("ignore", message=".*half.*deprecated.*")
LOGGER.setLevel(logging.ERROR)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "starter"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import games  # noqa: E402
import load_homography  # noqa: E402
from src.atd_io import load_atd  # noqa: E402

CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_WEIGHTS = ROOT / "models" / "ball_yolo26s_stage_b.pt"
REC_NPZ = CACHE_DIR / "cv_ball_recovery_BAR-ATH.npz"
OUT_NPZ = CACHE_DIR / "cv_boundary_shifts_BAR-ATH.npz"
PREVIEW_DIR = CACHE_DIR / "cv_runs" / "boundary_preview"

TILE_SIZE = 640
TACTICAL_W = 1920
TACTICAL_H = 1080
BATCH_SIZE = 8


def _load_ball_track() -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    m = load_atd(verbose=False)
    bx, by = m["ball_x"].astype(np.float32).copy(), m["ball_y"].astype(np.float32).copy()
    if REC_NPZ.exists():
        d = np.load(REC_NPZ)
        fill = d["recovered_mask"] & (~np.isfinite(bx))
        bx[fill], by[fill] = d["ball_x"][fill], d["ball_y"][fill]
    return bx, by, tuple(float(x) for x in m["pitch"])


def _find_d_to_live_transitions(labels: np.ndarray, min_seg: int = 50) -> list[int]:
    """Return frames `t` where `labels[t-1] == 'D'` and `labels[t] in ('H', 'A')` with stable flanking segments."""
    n = len(labels)
    out = []
    # Compute segment lengths cleanly
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    for i in range(1, len(starts)):
        prev_lbl = labels[starts[i - 1]]
        cur_lbl = labels[starts[i]]
        prev_len = int(ends[i - 1] - starts[i - 1])
        cur_len = int(ends[i] - starts[i])
        if prev_lbl == "D" and cur_lbl in ("H", "A") and prev_len >= min_seg and cur_len >= min_seg:
            out.append(int(starts[i]))
    return out


def _detect_kick_onset(
    t_trans: int,
    obs: dict[int, tuple[float, float, float, float, float]],  # fr -> (xm, ym, ux, uy, conf)
    search_lo: int = -4,
    search_hi: int = 30,
) -> tuple[int | None, float | None]:
    """Locate last stationary frame `tk` before a pitch ball launches into sustained movement."""
    best_tk = None
    best_score = -1e9
    for dt in range(search_lo, search_hi + 1):
        tk = t_trans + dt
        pre = [obs[f] for f in range(tk - 10, tk + 1) if f in obs]
        post = [obs[f] for f in range(tk + 2, tk + 14) if f in obs]
        if len(pre) < 4 or len(post) < 4 or tk not in obs:
            continue

        pre_x = np.array([p[0] for p in pre])
        pre_y = np.array([p[1] for p in pre])
        ux_med = float(np.median([p[2] for p in pre]))
        uy_med = float(np.median([p[3] for p in pre]))
        if not (0.015 <= ux_med <= 0.985 and 0.015 <= uy_med <= 0.985):
            continue

        x0, y0 = float(np.median(pre_x)), float(np.median(pre_y))
        pre_dev = np.hypot(pre_x - x0, pre_y - y0)
        cur_dev = float(((obs[tk][0] - x0) ** 2 + (obs[tk][1] - y0) ** 2) ** 0.5)
        if float(np.max(pre_dev)) > 0.42 or cur_dev > 0.35:
            continue
        # Guard against post-kick pass-traps: all observed positions since (t_trans - 2) must stay near (x0, y0)
        since_pre = [obs[f] for f in range(min(t_trans - 2, tk - 10), tk + 1) if f in obs]
        if since_pre and max((((p[0] - x0) ** 2 + (p[1] - y0) ** 2) ** 0.5) for p in since_pre) > 0.65:
            continue

        post_early = [obs[f] for f in range(tk + 2, tk + 7) if f in obs]
        post_late = [obs[f] for f in range(tk + 7, tk + 15) if f in obs]
        if not post_early or not post_late:
            continue
        early_max = max((((p[0] - x0) ** 2 + (p[1] - y0) ** 2) ** 0.5) for p in post_early)
        late_min = min((((p[0] - x0) ** 2 + (p[1] - y0) ** 2) ** 0.5) for p in post_late)
        if early_max >= 0.75 and late_min >= 1.50:
            score = float(len(pre)) * 0.2 + late_min - float(np.max(pre_dev)) * 2.0
            # Keep the LAST frame tk that is still strictly inside stationary cluster before launch
            best_tk = tk
            best_score = score
    return best_tk, (round(best_score, 3) if best_tk is not None else None)


def run_boundary_motion_check(
    in_csv: Path,
    out_npz: Path = OUT_NPZ,
    out_csv: Path | None = None,
    weights_path: Path = DEFAULT_WEIGHTS,
) -> dict:
    t0 = time.time()
    print("=== Sprint 5: Visual Kick-Onset Check at D -> Live Boundaries ===")
    df = pd.read_csv(in_csv)
    col = df.columns[1]
    labels = df[col].values.copy()

    bx, by, (pitch_w, pitch_h) = _load_ball_track()
    H_mats, _, valid_h = load_homography.load()
    n_frames = len(labels)
    d2l_frames = _find_d_to_live_transitions(labels, min_seg=50)
    print(f"[Plan] Candidate D->Live restarts (seg>=50f): {len(d2l_frames)}")

    # Gather local windows [t - 10 .. t + 36] only where anchor ball coords exist nearby
    job_targets: dict[int, tuple[int, float, float]] = {}  # fr -> (t_trans, roi_cx, roi_cy)
    trans_windows: dict[int, list[int]] = {}
    for t_tr in d2l_frames:
        win_lo = max(1, t_tr - 14)
        win_hi = min(n_frames - 1, t_tr + 44)
        # Find nearest valid ball coord inside/near window to center 640x640 ROI
        sub = np.arange(max(0, t_tr - 40), min(n_frames, t_tr + 45))
        fin = sub[np.isfinite(bx[sub]) & np.isfinite(by[sub]) & valid_h[sub]]
        if len(fin) == 0:
            continue
        nearest_f = int(fin[np.argmin(np.abs(fin - t_tr))])
        ux_ref, uy_ref = float(bx[nearest_f]), float(by[nearest_f])
        w_list = []
        for fr in range(win_lo, win_hi + 1):
            if not valid_h[fr]:
                continue
            # Follow finite ball if present on fr, else project reference coords (set piece!)
            if np.isfinite(bx[fr]) and np.isfinite(by[fr]):
                cx_ref, cy_ref = load_homography.project(H_mats[fr], float(bx[fr]), float(by[fr]))
            else:
                cx_ref, cy_ref = load_homography.project(H_mats[fr], ux_ref, uy_ref)
            if -40.0 <= float(cx_ref) <= TACTICAL_W + 40.0 and -40.0 <= float(cy_ref) <= TACTICAL_H + 40.0:
                job_targets[fr] = (t_tr, float(cx_ref), float(cy_ref))
                w_list.append(fr)
        if w_list:
            trans_windows[t_tr] = w_list

    job_frames = sorted(job_targets.keys())
    print(f"[Video Scan] Stride-1 ROI frames to evaluate: {len(job_frames)}")

    model = YOLO(str(weights_path))
    vpath = games.video_path(game="BAR-ATH", source="tactical")
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vpath}")

    raw_obs: dict[int, tuple[float, float, float, float, float]] = {}  # fr -> (xm, ym, ux, uy, conf)
    px_obs: dict[int, tuple[float, float, float]] = {}  # fr -> (px, py, conf) for preview
    queue: list[tuple[int, int, int, np.ndarray]] = []

    def flush() -> None:
        if not queue:
            return
        for i in range(0, len(queue), BATCH_SIZE):
            chunk = queue[i : i + BATCH_SIZE]
            results = model.predict(
                [c[3] for c in chunk], imgsz=TILE_SIZE, batch=len(chunk), half=True, conf=0.22, iou=0.45, device=0, verbose=False
            )
            for (fr, x0, y0, _), res in zip(chunk, results):
                boxes = getattr(res, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                best = None
                best_c = -1.0
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    w, h = float(x2 - x1), float(y2 - y1)
                    if not (4.0 <= w <= 28.0 and 4.0 <= h <= 28.0 and 0.45 <= w / max(1e-6, h) <= 2.2):
                        continue
                    cx, cy = 0.5 * (float(x1) + float(x2)) + x0, 0.5 * (float(y1) + float(y2)) + y0
                    ux, uy = load_homography.unproject(H_mats[fr], cx, cy)
                    uxf, uyf = float(ux), float(uy)
                    if -0.02 <= uxf <= 1.02 and -0.02 <= uyf <= 1.02 and float(c) > best_c:
                        best_c = float(c)
                        best = (uxf * pitch_w, uyf * pitch_h, uxf, uyf, float(c), cx, cy)
                if best is not None:
                    raw_obs[fr] = best[:5]
                    px_obs[fr] = (best[5], best[6], best[4])
        queue.clear()

    cur_pos = -1000
    for idx_j, fr in enumerate(job_frames):
        if fr - cur_pos > 75 or fr < cur_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            cur_pos = fr - 1
        while cur_pos < fr - 1:
            cap.grab()
            cur_pos += 1
        ok, img = cap.read()
        cur_pos += 1
        if not ok:
            continue
        _, rcx, rcy = job_targets[fr]
        x0 = int(np.clip(round(rcx - TILE_SIZE * 0.5), 0, TACTICAL_W - TILE_SIZE))
        y0 = int(np.clip(round(rcy - TILE_SIZE * 0.5), 0, TACTICAL_H - TILE_SIZE))
        queue.append((fr, x0, y0, img[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]))
        if len(queue) >= BATCH_SIZE * 4:
            flush()
        if (idx_j + 1) % 1500 == 0 or idx_j + 1 == len(job_frames):
            print(f"  [Scan {idx_j + 1}/{len(job_frames)}] elapsed={time.time() - t0:.1f}s")
    flush()
    cap.release()

    # Evaluate kick onset per D -> Live boundary
    shifts: list[dict] = []
    new_labels = labels.copy()
    for t_tr in sorted(trans_windows.keys()):
        tk, sc = _detect_kick_onset(t_tr, raw_obs, search_lo=-4, search_hi=28)
        if tk is None:
            continue
        # Moving kick frame starts right after last stationary frame (tk + 1)
        t_snap = tk + 1
        delta = t_snap - t_tr
        if abs(delta) >= 2 and -4 <= delta <= 28:
            # Verify live team label remains valid around t_snap
            if delta > 0:
                new_labels[t_tr:t_snap] = "D"
            else:
                team_after = labels[t_tr]
                new_labels[t_snap:t_tr] = team_after
            shifts.append({"t_trans": t_tr, "t_snap": t_snap, "delta": delta, "score": sc})

    # Save visual strips of first 6 boundary snaps (showing before/after kick frames)
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    if shifts:
        cap = cv2.VideoCapture(str(vpath))
        for k, sh in enumerate(shifts[:6]):
            ts = sh["t_snap"]
            panels = []
            focus = px_obs.get(ts, px_obs.get(ts - 1))
            if focus is None:
                continue
            cx0 = int(np.clip(round(focus[0]) - 140, 0, TACTICAL_W - 280))
            cy0 = int(np.clip(round(focus[1]) - 140, 0, TACTICAL_H - 280))
            for f_off, tag in ((-6, "stat"), (-2, "stat"), (0, "KICK"), (4, "move")):
                fr = ts + f_off
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, im = cap.read()
                if not ok:
                    continue
                crop = im[cy0 : cy0 + 280, cx0 : cx0 + 280].copy()
                if fr in px_obs:
                    bxp, byp, _ = px_obs[fr]
                    cv2.circle(crop, (int(round(bxp - cx0)), int(round(byp - cy0))), 8, (0, 255, 0), 2)
                cv2.putText(crop, f"{tag} f={fr}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                panels.append(crop)
            if len(panels) == 4:
                cv2.imwrite(str(PREVIEW_DIR / f"kick_{k:02d}_t{ts:06d}_d{sh['delta']:+d}.jpg"), np.hstack(panels))
        cap.release()

    np.savez_compressed(out_npz, shifts_json=json.dumps(shifts))
    if out_csv is not None:
        df[col] = new_labels
        df.to_csv(out_csv, index=False)

    summary = {
        "elapsed_s": round(time.time() - t0, 1),
        "candidates": len(d2l_frames),
        "scanned": len(trans_windows),
        "snapped_count": len(shifts),
        "mean_delta_frames": round(float(np.mean([s["delta"] for s in shifts])), 2) if shifts else 0.0,
        "median_delta_frames": round(float(np.median([s["delta"] for s in shifts])), 1) if shifts else 0.0,
        "sample_shifts": shifts[:15],
    }
    (PREVIEW_DIR / "boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Sprint 5 DONE] {json.dumps(summary, indent=2)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 5: Visual kick-onset alignment at D->Live boundaries.")
    parser.add_argument("--in-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-npz", type=Path, default=OUT_NPZ)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    run_boundary_motion_check(args.in_csv, args.out_npz, args.out_csv, args.weights)


if __name__ == "__main__":
    main()
