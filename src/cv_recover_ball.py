r"""Sprint 3: Domain-adapted YOLO26s Ball Recovery on BAR-ATH ball-NaN gaps.

Two-pass hardware-optimal pipeline (all batched at batch=8 FP16 = ~123 tiles/s on RTX 3050 6GB):
  Pass 1 (Keyframe Full-Pitch Seeder): samples adaptive keyframes inside ATD ball-NaN gaps,
      filters tiles via homography pitch footprint, runs 640x640 Stage-B detector, unprojects to
      metric pitch coords, and links physically valid trajectories per gap (DP/DAG + ATD anchors).
  Pass 2 (Bidirectional ROI Densifier): crops single 640x640 ROI tiles at stride=2 along
      verified Pass-1 tracks, accepts detections within 3.2m corridor at conf>=0.18, bridges
      short <=6f occlusions, and applies light Gaussian smoothing (sigma=1.6) matching ATD.

ZERO human labels and ZERO ground-truth labels are touched. Outputs deterministic cache under
`data/cache/cv_ball_recovery_BAR-ATH.npz` plus visual audit crops in `data/cache/cv_runs/`.

Usage:
    .\.venv\Scripts\python.exe -u src/cv_recover_ball.py --pilot
    .\.venv\Scripts\python.exe -u src/cv_recover_ball.py --full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
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
from src.model import play_mask  # noqa: E402
from src.smart_priors import compute_physical_cues  # noqa: E402

CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_WEIGHTS = ROOT / "models" / "ball_yolo26s_stage_b.pt"
OUT_NPZ = CACHE_DIR / "cv_ball_recovery_BAR-ATH.npz"
PILOT_NPZ = CACHE_DIR / "cv_ball_recovery_pilot.npz"
PREVIEW_DIR = CACHE_DIR / "cv_runs" / "recovery_preview"

TILE_SIZE = 640
TACTICAL_W = 1920
TACTICAL_H = 1080
TILE_XS = (0, 427, 853, 1280)
TILE_YS = (0, 440)
BATCH_SIZE = 8
FPS = 25.0

# Precomputed pitch footprint sample grid in normalized [0..1] coords (with small margin)
_GX, _GY = np.meshgrid(np.linspace(-0.02, 1.02, 11), np.linspace(-0.02, 1.02, 7))
PITCH_GRID_HOMOG = np.vstack([_GX.ravel(), _GY.ravel(), np.ones(_GX.size, dtype=np.float64)])


def _active_tile_coords(H_mat: np.ndarray) -> list[tuple[int, int]]:
    """Return subset of the 8 640x640 tile origins overlapping the normalized pitch."""
    proj = H_mat @ PITCH_GRID_HOMOG
    z = proj[2]
    ok = np.abs(z) > 1e-7
    px = proj[0, ok] / z[ok]
    py = proj[1, ok] / z[ok]
    out: list[tuple[int, int]] = []
    pad = 24.0
    for y0 in TILE_YS:
        for x0 in TILE_XS:
            if np.any((px >= x0 - pad) & (px <= x0 + TILE_SIZE + pad) & (py >= y0 - pad) & (py <= y0 + TILE_SIZE + pad)):
                out.append((x0, y0))
    return out


def _nan_gaps(vis: np.ndarray, pm: np.ndarray, max_frame: int | None = None) -> list[tuple[int, int]]:
    """Return contiguous [a, b) gaps of play_mask & ~vis."""
    target = pm & (~vis)
    if max_frame is not None:
        target = target.copy()
        target[max_frame + 1 :] = False
    edges = np.diff(np.concatenate(([np.int8(0)], target.astype(np.int8), [np.int8(0)])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _merge_frame_dets(
    dets: list[tuple[float, float, float, float, float]],
    iou_thresh: float = 0.30,
    center_thresh: float = 14.0,
) -> list[tuple[float, float, float, float, float]]:
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: d[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    for cand in ordered:
        cx1, cy1, cx2, cy2, _ = cand
        ccx, ccy = 0.5 * (cx1 + cx2), 0.5 * (cy1 + cy2)
        carea = max(1e-6, (cx2 - cx1) * (cy2 - cy1))
        dup = False
        for kx1, ky1, kx2, ky2, _ in kept:
            kcx, kcy = 0.5 * (kx1 + kx2), 0.5 * (ky1 + ky2)
            if ((ccx - kcx) ** 2 + (ccy - kcy) ** 2) ** 0.5 <= center_thresh:
                dup = True
                break
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > 0.0:
                karea = max(1e-6, (kx2 - kx1) * (ky2 - ky1))
                if inter / (carea + karea - inter) >= iou_thresh:
                    dup = True
                    break
        if not dup:
            kept.append(cand)
    return kept


def _predict_batch_tiles(
    model: YOLO,
    tiles: list[np.ndarray],
    conf: float,
) -> list[list[tuple[float, float, float, float, float]]]:
    """Run YOLO26s on tiles chunked in optimal batch=8 FP16."""
    all_out: list[list[tuple[float, float, float, float, float]]] = []
    for i in range(0, len(tiles), BATCH_SIZE):
        chunk = tiles[i : i + BATCH_SIZE]
        results = model.predict(
            chunk, imgsz=TILE_SIZE, batch=len(chunk), half=True, conf=conf, iou=0.45, device=0, verbose=False
        )
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                all_out.append([])
            else:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                valid_list = []
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    w, h = float(x2 - x1), float(y2 - y1)
                    if 4.0 <= w <= 28.0 and 4.0 <= h <= 28.0 and 0.45 <= w / max(1e-6, h) <= 2.2:
                        valid_list.append((float(x1), float(y1), float(x2), float(y2), float(c)))
                all_out.append(valid_list)
    return all_out


def _plan_pass1_keyframes(
    gaps: list[tuple[int, int]],
    mps_fast: np.ndarray,
    valid_h: np.ndarray,
) -> tuple[list[int], dict[int, int]]:
    """Build sorted keyframe list & gap index map."""
    frame_to_gap: dict[int, int] = {}
    for g_idx, (a, b) in enumerate(gaps):
        if b - a <= 0:
            continue
        frames = set()
        # Boundary probes
        for k in (a, min(b - 1, a + 2), max(a, b - 3), b - 1):
            frames.add(k)
        t = a
        while t < b:
            frames.add(t)
            near_edge = (t - a <= 60) or (b - t <= 60)
            step = 6 if (near_edge or float(mps_fast[t]) >= 1.10) else 18
            t += step
        for fr in frames:
            if valid_h[fr]:
                frame_to_gap[fr] = g_idx
    return sorted(frame_to_gap.keys()), frame_to_gap


def _read_frames_pass(
    video_path: Path,
    frame_jobs: list[int],
    on_frame,
    seek_thresh: int = 75,
    log_every: int = 800,
    tag: str = "Pass",
) -> None:
    """Linear sequential video sweep executing callback(fr, bgr_img) cleanly."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cur_pos = -1000
    t0 = time.time()
    n = len(frame_jobs)
    for i, target_f in enumerate(frame_jobs):
        if target_f - cur_pos > seek_thresh or target_f < cur_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            cur_pos = target_f - 1
        while cur_pos < target_f - 1:
            if not cap.grab():
                break
            cur_pos += 1
        ok, frame = cap.read()
        cur_pos += 1
        if ok:
            on_frame(target_f, frame)
        if (i + 1) % log_every == 0 or i + 1 == n:
            elapsed = max(1e-3, time.time() - t0)
            print(f"  [{tag}] {i + 1}/{n} frames ({(i + 1) / elapsed:.1f} fr/s, {elapsed:.1f}s)")
    cap.release()


def _extract_gap_chains(
    gap_a: int,
    gap_b: int,
    det_by_frame: dict[int, list[dict]],
    bx: np.ndarray,
    by: np.ndarray,
    vis: np.ndarray,
    pitch_w: float,
    pitch_h: float,
) -> list[list[dict]]:
    """Link Pass-1 keyframe detections inside [gap_a, gap_b) using metric DAG chaining."""
    frames = sorted(det_by_frame.keys())
    if not frames:
        return []

    # Optional finite ATD endpoints in metric coords
    left_anchor = None
    if gap_a > 0 and vis[gap_a - 1]:
        left_anchor = (gap_a - 1, float(bx[gap_a - 1]) * pitch_w, float(by[gap_a - 1]) * pitch_h)
    right_anchor = None
    if gap_b < len(bx) and vis[gap_b]:
        right_anchor = (gap_b, float(bx[gap_b]) * pitch_w, float(by[gap_b]) * pitch_h)

    # Flatten nodes
    nodes: list[dict] = []
    for t in frames:
        for d in det_by_frame[t]:
            nodes.append(d)
    if not nodes:
        return []

    used = [False] * len(nodes)
    accepted_chains: list[list[dict]] = []

    while True:
        avail = [i for i in range(len(nodes)) if not used[i]]
        if not avail:
            break
        dp_score = {i: (nodes[i]["conf"] - 0.15) for i in avail}
        prev_ptr: dict[int, int | None] = {i: None for i in avail}

        for idx_pos, i in enumerate(avail):
            ni = nodes[i]
            ti = ni["frame"]
            # ATD left-anchor bonus if trajectory connects smoothly from gap_a - 1
            if left_anchor is not None:
                dt_l = (ti - left_anchor[0]) / FPS
                if 0.0 < dt_l <= 0.8:
                    dl = ((ni["xm"] - left_anchor[1]) ** 2 + (ni["ym"] - left_anchor[2]) ** 2) ** 0.5
                    if dl <= max(2.2, 36.0 * dt_l + 1.2):
                        dp_score[i] += 0.65 - 0.05 * dl

            for j in avail[idx_pos + 1 :]:
                nj = nodes[j]
                tj = nj["frame"]
                if tj == ti:
                    continue
                dt_s = (tj - ti) / FPS
                if dt_s > 2.2:
                    break
                dist_m = ((nj["xm"] - ni["xm"]) ** 2 + (nj["ym"] - ni["ym"]) ** 2) ** 0.5
                max_jump = max(2.4, 36.0 * dt_s + 1.4)
                if dist_m > max_jump:
                    continue
                cand_score = dp_score[i] + (nj["conf"] - 0.15) - 0.04 * dist_m - 0.12 * dt_s
                if cand_score > dp_score[j]:
                    dp_score[j] = cand_score
                    prev_ptr[j] = i

        # Add right-anchor affinity at end scores
        end_scores: dict[int, float] = {}
        for i in avail:
            s = dp_score[i]
            if right_anchor is not None:
                dt_r = (right_anchor[0] - nodes[i]["frame"]) / FPS
                if 0.0 < dt_r <= 0.8:
                    dr = ((nodes[i]["xm"] - right_anchor[1]) ** 2 + (nodes[i]["ym"] - right_anchor[2]) ** 2) ** 0.5
                    if dr <= max(2.2, 36.0 * dt_r + 1.2):
                        s += 0.65 - 0.05 * dr
            end_scores[i] = s

        best_end = max(end_scores, key=lambda k: end_scores[k])
        chain_indices: list[int] = []
        cur: int | None = best_end
        while cur is not None:
            chain_indices.append(cur)
            cur = prev_ptr[cur]
        chain_indices.reverse()

        chain = [nodes[i] for i in chain_indices]
        chain_frames = {d["frame"] for d in chain}
        # Mark chain nodes (and same-frame nodes in this span if conflicting lower-score distractor within chain)
        for i in chain_indices:
            used[i] = True

        mean_conf = float(np.mean([d["conf"] for d in chain]))
        span_f = chain[-1]["frame"] - chain[0]["frame"] + 1
        n_hits = len(chain)

        anchored_left = False
        if left_anchor is not None:
            dt_l = (chain[0]["frame"] - left_anchor[0]) / FPS
            dl = ((chain[0]["xm"] - left_anchor[1]) ** 2 + (chain[0]["ym"] - left_anchor[2]) ** 2) ** 0.5
            anchored_left = 0.0 < dt_l <= 0.8 and dl <= max(2.4, 36.0 * dt_l + 1.2)
        anchored_right = False
        if right_anchor is not None:
            dt_r = (right_anchor[0] - chain[-1]["frame"]) / FPS
            dr = ((chain[-1]["xm"] - right_anchor[1]) ** 2 + (chain[-1]["ym"] - right_anchor[2]) ** 2) ** 0.5
            anchored_right = 0.0 < dt_r <= 0.8 and dr <= max(2.4, 36.0 * dt_r + 1.2)
        anchored = anchored_left or anchored_right

        valid = False
        if anchored and n_hits >= 3 and mean_conf >= 0.40:
            valid = True
        elif n_hits >= 5 and span_f >= 10 and mean_conf >= 0.60:
            valid = True

        if valid:
            # Remove overlapping already-covered frames from remaining pool so competing socks don't fork duplicate track
            t_lo, t_hi = chain[0]["frame"], chain[-1]["frame"]
            for i in avail:
                if t_lo <= nodes[i]["frame"] <= t_hi:
                    used[i] = True
            accepted_chains.append(chain)
        elif n_hits <= 1:
            break

        # Also remove singletons at covered frames just in case
        _ = chain_frames
    accepted_chains.sort(key=lambda c: c[0]["frame"])
    return accepted_chains


def run_recovery(max_frame: int | None = None, weights_path: Path = DEFAULT_WEIGHTS, out_npz: Path = OUT_NPZ) -> Path:
    if not weights_path.exists():
        raise FileNotFoundError(f"Stage-B weights missing: {weights_path}")

    t_start = time.time()
    print(f"=== Sprint 3: YOLO26s Ball Recovery ({'PILOT' if max_frame else 'FULL MATCH'}) ===")
    match = load_atd(verbose=False)
    cues = compute_physical_cues(match)
    H_mats, _, valid_h = load_homography.load()
    pm = play_mask(match)
    vis = cues["vis"]
    bx = match["ball_x"].astype(np.float64).copy()
    by = match["ball_y"].astype(np.float64).copy()
    pitch_w, pitch_h = match["pitch"]
    n_frames = match["n_frames"]

    gaps = _nan_gaps(vis, pm, max_frame=max_frame)
    keyframes, frame_to_gap = _plan_pass1_keyframes(gaps, cues["mps_fast"], valid_h)
    total_gap_frames = sum(b - a for a, b in gaps)
    print(f"[Plan] gaps={len(gaps)}, total_gap_frames={total_gap_frames}, pass1_keyframes={len(keyframes)}")

    model = YOLO(str(weights_path))
    vpath = games.video_path(game="BAR-ATH", source="tactical")

    # Pass 1: Batched full-pitch keyframe scanning
    gap_dets: dict[int, dict[int, list[dict]]] = {g: {} for g in range(len(gaps))}
    tile_queue: list[tuple[int, int, int, np.ndarray]] = []  # (fr, x0, y0, tile_bgr)
    pass1_tiles = 0

    def flush_pass1() -> None:
        nonlocal pass1_tiles
        if not tile_queue:
            return
        imgs = [item[3] for item in tile_queue]
        preds = _predict_batch_tiles(model, imgs, conf=0.25)
        pass1_tiles += len(imgs)
        by_frame: dict[int, list[tuple[float, float, float, float, float]]] = {}
        for (fr, x0, y0, _), boxes in zip(tile_queue, preds):
            flist = by_frame.setdefault(fr, [])
            for x1, y1, x2, y2, c in boxes:
                flist.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, c))
        for fr, raw_boxes in by_frame.items():
            merged = _merge_frame_dets(raw_boxes)
            kept_nodes = []
            H_fr = H_mats[fr]
            for x1, y1, x2, y2, c in merged:
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                ux, uy = load_homography.unproject(H_fr, cx, cy)
                uxf, uyf = float(ux), float(uy)
                if -0.025 <= uxf <= 1.025 and -0.025 <= uyf <= 1.025:
                    kept_nodes.append({
                        "frame": fr, "cx": cx, "cy": cy, "w": x2 - x1, "h": y2 - y1,
                        "conf": c, "ux": uxf, "uy": uyf, "xm": uxf * pitch_w, "ym": uyf * pitch_h,
                    })
            if kept_nodes:
                g_idx = frame_to_gap[fr]
                gap_dets[g_idx].setdefault(fr, []).extend(kept_nodes)
        tile_queue.clear()

    def on_pass1_frame(fr: int, frame: np.ndarray) -> None:
        for x0, y0 in _active_tile_coords(H_mats[fr]):
            tile_queue.append((fr, x0, y0, frame[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]))
            if len(tile_queue) >= BATCH_SIZE * 4:
                flush_pass1()

    print("[Pass 1] Scanning adaptive keyframes inside ball-NaN gaps...")
    _read_frames_pass(vpath, keyframes, on_pass1_frame, tag="Pass1")
    flush_pass1()

    # Chain verified trajectories inside each gap
    all_chains: list[tuple[int, int, list[dict]]] = []
    for g_idx, (a, b) in enumerate(gaps):
        chains = _extract_gap_chains(a, b, gap_dets[g_idx], bx, by, vis, pitch_w, pitch_h)
        for ch in chains:
            all_chains.append((a, b, ch))

    pass1_span_frames = sum(ch[-1]["frame"] - ch[0]["frame"] + 1 for _, _, ch in all_chains)
    print(f"[Pass 1 DONE] tiles={pass1_tiles} | accepted_chains={len(all_chains)} | span_frames={pass1_span_frames}")

    # Pass 2: Stride-2 1-tile ROI Densifier along verified chains
    roi_targets: dict[int, tuple[float, float, float, float]] = {}  # fr -> (pred_px, pred_py, pred_xm, pred_ym)
    for a, b, ch in all_chains:
        c_frames = np.array([d["frame"] for d in ch], dtype=np.float64)
        c_xm = np.array([d["xm"] for d in ch], dtype=np.float64)
        c_ym = np.array([d["ym"] for d in ch], dtype=np.float64)
        t_lo = max(a, ch[0]["frame"] - (4 if a > 0 and vis[a - 1] and ch[0]["frame"] - a <= 10 else 0))
        t_hi = min(b - 1, ch[-1]["frame"] + (4 if b < n_frames and vis[b] and b - ch[-1]["frame"] <= 10 else 0))
        for fr in range(t_lo, t_hi + 1):
            if (fr % 2 != 0) and (fr != t_lo) and (fr != t_hi):
                continue
            if not valid_h[fr]:
                continue
            pxm = float(np.interp(fr, c_frames, c_xm))
            pym = float(np.interp(fr, c_frames, c_ym))
            ppx, ppy = load_homography.project(H_mats[fr], pxm / pitch_w, pym / pitch_h)
            if 0.0 <= ppx <= TACTICAL_W and 0.0 <= ppy <= TACTICAL_H:
                roi_targets[fr] = (float(ppx), float(ppy), pxm, pym)

    roi_jobs = sorted(roi_targets.keys())
    print(f"[Pass 2] ROI densification jobs (stride=2, 1-tile/job): {len(roi_jobs)}")

    dense_hits: dict[int, tuple[float, float, float, float, float]] = {}  # fr -> (ux, uy, cx, cy, conf)
    roi_queue: list[tuple[int, int, int, float, float, np.ndarray]] = []
    pass2_tiles = 0

    def flush_pass2() -> None:
        nonlocal pass2_tiles
        if not roi_queue:
            return
        imgs = [item[5] for item in roi_queue]
        preds = _predict_batch_tiles(model, imgs, conf=0.18)
        pass2_tiles += len(imgs)
        for (fr, x0, y0, pxm, pym, _), boxes in zip(roi_queue, preds):
            H_fr = H_mats[fr]
            best = None
            best_score = -1e9
            for x1, y1, x2, y2, c in boxes:
                cx = 0.5 * (x1 + x2) + x0
                cy = 0.5 * (y1 + y2) + y0
                ux, uy = load_homography.unproject(H_fr, cx, cy)
                uxf, uyf = float(ux), float(uy)
                if not (-0.025 <= uxf <= 1.025 and -0.025 <= uyf <= 1.025):
                    continue
                dist_m = (((uxf * pitch_w) - pxm) ** 2 + ((uyf * pitch_h) - pym) ** 2) ** 0.5
                if dist_m <= 3.5:
                    score = c - 0.06 * dist_m
                    if score > best_score:
                        best_score = score
                        best = (uxf, uyf, cx, cy, c)
            if best is not None:
                dense_hits[fr] = best
        roi_queue.clear()

    def on_pass2_frame(fr: int, frame: np.ndarray) -> None:
        ppx, ppy, pxm, pym = roi_targets[fr]
        x0 = int(np.clip(round(ppx - TILE_SIZE * 0.5), 0, TACTICAL_W - TILE_SIZE))
        y0 = int(np.clip(round(ppy - TILE_SIZE * 0.5), 0, TACTICAL_H - TILE_SIZE))
        roi_queue.append((fr, x0, y0, pxm, pym, frame[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]))
        if len(roi_queue) >= BATCH_SIZE * 4:
            flush_pass2()

    _read_frames_pass(vpath, roi_jobs, on_pass2_frame, tag="Pass2")
    flush_pass2()

    # Merge Pass-1 chain hits with Pass-2 dense hits inside each gap & bridge <= 6f
    rec_bx = np.full(n_frames, np.nan, dtype=np.float32)
    rec_by = np.full(n_frames, np.nan, dtype=np.float32)
    rec_conf = np.zeros(n_frames, dtype=np.float32)

    for fr, (uxf, uyf, _, _, c) in dense_hits.items():
        rec_bx[fr] = uxf
        rec_by[fr] = uyf
        rec_conf[fr] = c
    for _, _, ch in all_chains:
        for d in ch:
            fr = d["frame"]
            if not np.isfinite(rec_bx[fr]) or d["conf"] > rec_conf[fr]:
                rec_bx[fr] = d["ux"]
                rec_by[fr] = d["uy"]
                rec_conf[fr] = d["conf"]

    # Bridge short <= 8f missing intervals between recovered detections (handles stride=2 odd frames cleanly!)
    raw_rec = np.isfinite(rec_bx)
    for a, b in gaps:
        sub_vis = raw_rec[a:b]
        hits = np.flatnonzero(sub_vis) + a
        if len(hits) == 0:
            continue
        # Connect internal gaps <= 8 frames
        for k in range(len(hits) - 1):
            f0, f1 = int(hits[k]), int(hits[k + 1])
            gap_k = f1 - f0 - 1
            if 1 <= gap_k <= 8:
                dist_m = (((rec_bx[f1] - rec_bx[f0]) * pitch_w) ** 2 + ((rec_by[f1] - rec_by[f0]) * pitch_h) ** 2) ** 0.5
                if dist_m <= max(2.4, 38.0 * ((f1 - f0) / FPS) + 1.0):
                    alpha = np.linspace(0.0, 1.0, gap_k + 2, dtype=np.float32)[1:-1]
                    rec_bx[f0 + 1 : f1] = rec_bx[f0] + alpha * (rec_bx[f1] - rec_bx[f0])
                    rec_by[f0 + 1 : f1] = rec_by[f0] + alpha * (rec_by[f1] - rec_by[f0])

        # Also smoothly bridge <= 4 frames directly to finite ATD boundaries a-1 / b
        if a > 0 and vis[a - 1]:
            f_first = int(hits[0])
            if 1 <= f_first - a + 1 <= 6:
                dm = (((rec_bx[f_first] - bx[a - 1]) * pitch_w) ** 2 + ((rec_by[f_first] - by[a - 1]) * pitch_h) ** 2) ** 0.5
                if dm <= 4.0:
                    gk = f_first - a
                    if gk > 0:
                        al = np.linspace(0.0, 1.0, gk + 2, dtype=np.float32)[1:-1]
                        rec_bx[a:f_first] = bx[a - 1] + al * (rec_bx[f_first] - bx[a - 1])
                        rec_by[a:f_first] = by[a - 1] + al * (rec_by[f_first] - by[a - 1])
        if b < n_frames and vis[b]:
            f_last = int(hits[-1])
            if 1 <= b - f_last <= 6:
                dm = (((bx[b] - rec_bx[f_last]) * pitch_w) ** 2 + ((by[b] - rec_by[f_last]) * pitch_h) ** 2) ** 0.5
                if dm <= 4.0:
                    gk = b - f_last - 1
                    if gk > 0:
                        al = np.linspace(0.0, 1.0, gk + 2, dtype=np.float32)[1:-1]
                        rec_bx[f_last + 1 : b] = rec_bx[f_last] + al * (bx[b] - rec_bx[f_last])
                        rec_by[f_last + 1 : b] = rec_by[f_last] + al * (by[b] - rec_by[f_last])

    # Smooth contiguous recovered spans (sigma=1.6 frames) & enforce strict mask (~vis & pm)
    recovered_mask = (~vis) & pm & np.isfinite(rec_bx) & np.isfinite(rec_by)
    edges = np.diff(np.concatenate(([np.int8(0)], recovered_mask.astype(np.int8), [np.int8(0)])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s >= 6:
            rec_bx[s:e] = gaussian_filter1d(rec_bx[s:e].astype(np.float64), sigma=1.6, mode="nearest").astype(np.float32)
            rec_by[s:e] = gaussian_filter1d(rec_by[s:e].astype(np.float64), sigma=1.6, mode="nearest").astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    n_recovered = int(recovered_mask.sum())
    np.savez_compressed(
        out_npz,
        recovered_mask=recovered_mask,
        ball_x=rec_bx,
        ball_y=rec_by,
        conf=rec_conf,
    )

    # Save visual audit previews of recovered detections across different recovered chains
    if PREVIEW_DIR.exists():
        import shutil
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    hit_frames = sorted(dense_hits.keys())
    if hit_frames:
        sample_idx = np.linspace(0, len(hit_frames) - 1, min(10, len(hit_frames)), dtype=int)
        preview_frames = [hit_frames[i] for i in sample_idx]
        cap = cv2.VideoCapture(str(vpath))
        for k, fr in enumerate(preview_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            uxf, uyf, cx, cy, c = dense_hits[fr]
            sx, sy = float(rec_bx[fr]), float(rec_by[fr])
            spx, spy = load_homography.project(H_mats[fr], sx, sy)
            vis_im = img.copy()
            cv2.circle(vis_im, (int(round(cx)), int(round(cy))), 9, (0, 255, 0), 2)
            cv2.drawMarker(vis_im, (int(round(float(spx))), int(round(float(spy)))), (255, 220, 0), cv2.MARKER_DIAMOND, 14, 2)
            cv2.putText(
                vis_im, f"RECOVERED f={fr} c={c:.2f} norm=({uxf:.3f},{uyf:.3f})",
                (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
            fx = int(np.clip(round(cx) - 180, 0, TACTICAL_W - 360))
            fy = int(np.clip(round(cy) - 180, 0, TACTICAL_H - 360))
            cv2.imwrite(str(PREVIEW_DIR / f"rec_{k:02d}_f{fr:06d}_crop.jpg"), vis_im[fy : fy + 360, fx : fx + 360])
            if k < 4:
                cv2.imwrite(str(PREVIEW_DIR / f"rec_{k:02d}_f{fr:06d}_full.jpg"), vis_im)
        cap.release()

    elapsed = time.time() - t_start
    rec_pct = 100.0 * n_recovered / max(1, total_gap_frames)
    oob_rec = int(
        (recovered_mask & ((rec_bx < 0.0) | (rec_bx > 1.0) | (rec_by < 0.0) | (rec_by > 1.0))).sum()
    )
    stats = {
        "mode": "pilot" if max_frame else "full",
        "max_frame": max_frame,
        "elapsed_s": round(elapsed, 1),
        "total_gap_frames": int(total_gap_frames),
        "recovered_frames": n_recovered,
        "recovered_pct_of_nan": round(rec_pct, 2),
        "recovered_oob_edge_frames": oob_rec,
        "accepted_chains": len(all_chains),
        "pass1_tiles": pass1_tiles,
        "pass2_tiles": pass2_tiles,
        "out_npz": str(out_npz),
    }
    print(f"[Sprint 3 DONE] {json.dumps(stats, indent=2)}")
    (PREVIEW_DIR / "recovery_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out_npz


def inject_recovered_ball(match: dict, npz_path: Path = OUT_NPZ) -> dict:
    """Return shallow copy of ATD match dict with verified YOLO26s coordinates filled into ball-NaN frames."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Ball recovery file not found: {npz_path}")
    data = np.load(npz_path)
    mask = data["recovered_mask"]
    if len(mask) != match["n_frames"]:
        raise ValueError(f"Recovery length {len(mask)} != match n_frames {match['n_frames']}")
    bx = match["ball_x"].astype(np.float32).copy()
    by = match["ball_y"].astype(np.float32).copy()
    fill = mask & (~np.isfinite(bx))
    bx[fill] = data["ball_x"][fill]
    by[fill] = data["ball_y"][fill]
    out = dict(match)
    out["ball_x"] = bx
    out["ball_y"] = by
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3: Recover BAR-ATH ball trajectories with Stage-B YOLO26s.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Fast ~30s pilot on frames <= 26000 for visual audit.")
    group.add_argument("--full", action="store_true", help="Full match recovery across H1 + H2.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    if args.pilot:
        run_recovery(max_frame=26000, weights_path=args.weights, out_npz=PILOT_NPZ)
    else:
        run_recovery(max_frame=None, weights_path=args.weights, out_npz=OUT_NPZ)


if __name__ == "__main__":
    main()
