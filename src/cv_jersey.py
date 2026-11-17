r"""Sprint 4: Self-supervised Jersey Color Disambiguation (Home vs Away) for BAR-ATH.

Fits a 100% label-free visual jersey classifier (FC Barcelona dark blaugrana vs Athletic Club
pale pink/white) using ONLY isolated uncrowded ATD tracklets strictly outside public GT
(fr > 19000, nearest neighbor >= 3.0 m, interior [0.15..0.85] excluding green/yellow GKs/refs).

Then audits:
  1. Held-out generalization accuracy on isolated opening-5m players (fr <= 19000, purely ATD slot consistency).
  2. Full-match tracklet team-swap detection (continuous ATD tracklets whose jersey conflicts with team_map).
  3. All tight duel frames (|d_h - d_a| < 1.5 m & min_d < 2.8 m) using Sprint-3 CV-recovered ball.

Usage:
    .\.venv\Scripts\python.exe -u src/cv_jersey.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "starter"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import games  # noqa: E402
import load_homography  # noqa: E402
from src.atd_io import load_atd  # noqa: E402
from src.model import play_mask  # noqa: E402

CACHE_DIR = ROOT / "data" / "cache"
RECOVERY_NPZ = CACHE_DIR / "cv_ball_recovery_BAR-ATH.npz"
OUT_NPZ = CACHE_DIR / "cv_jersey_BAR-ATH.npz"
PREVIEW_DIR = CACHE_DIR / "cv_runs" / "jersey_preview"
TACTICAL_W = 1920
TACTICAL_H = 1080


def _inject_cv_ball_fast(match: dict) -> dict:
    if not RECOVERY_NPZ.exists():
        return match
    d = np.load(RECOVERY_NPZ)
    bx = match["ball_x"].astype(np.float32).copy()
    by = match["ball_y"].astype(np.float32).copy()
    fill = d["recovered_mask"] & (~np.isfinite(bx))
    bx[fill] = d["ball_x"][fill]
    by[fill] = d["ball_y"][fill]
    out = dict(match)
    out["ball_x"] = bx
    out["ball_y"] = by
    return out


def _player_px_scale(H_mat: np.ndarray, nx: float, ny: float, pitch_w: float) -> tuple[float, float, float]:
    """Return foot (fx, fy) and local horizontal meters->pixels scale sx."""
    fx, fy = load_homography.project(H_mat, nx, ny)
    xx, xy = load_homography.project(H_mat, nx + 1.0 / pitch_w, ny)
    sx = float(np.hypot(xx - fx, xy - fy))
    return float(fx), float(fy), sx


def torso_patch_coords(fx: float, fy: float, sx: float) -> tuple[int, int, int, int] | None:
    """Compute tight upper-chest shirt box (x1, y1, x2, y2) avoiding shorts."""
    if sx < 12.0 or sx > 65.0:
        return None
    h_px = 1.65 * sx
    x1 = int(round(fx - 0.24 * sx))
    x2 = int(round(fx + 0.24 * sx))
    y1 = int(round(fy - 0.84 * h_px))
    y2 = int(round(fy - 0.50 * h_px))
    if x1 < 0 or y1 < 0 or x2 > TACTICAL_W or y2 > TACTICAL_H or (x2 - x1) < 5 or (y2 - y1) < 6:
        return None
    return x1, y1, x2, y2


def extract_torso_features(img_bgr: np.ndarray, fx: float, fy: float, sx: float) -> tuple[np.ndarray | None, float]:
    """Extract shadow-invariant grass-relative CIE-Lab + HSV chest jersey features."""
    box = torso_patch_coords(fx, fy, sx)
    if box is None:
        return None, 0.0
    x1, y1, x2, y2 = box
    patch = img_bgr[y1:y2, x1:x2]
    b, g, r = patch[:, :, 0].astype(np.float32), patch[:, :, 1].astype(np.float32), patch[:, :, 2].astype(np.float32)

    ex_green = g - np.maximum(r, b)
    is_grass = (ex_green > 8.0) & (g > 55.0)
    non_grass = ~is_grass
    n_valid = int(non_grass.sum())
    valid_ratio = n_valid / max(1.0, float(non_grass.size))
    if n_valid < 12 or valid_ratio < 0.20:
        return None, valid_ratio

    # Measure local ambient grass lightness around player (removes stadium roof shadow shifts!)
    h_px = 1.65 * sx
    bx1 = max(0, int(round(fx - 0.9 * sx)))
    bx2 = min(TACTICAL_W, int(round(fx + 0.9 * sx)))
    by1 = max(0, int(round(fy - 0.9 * h_px)))
    by2 = min(TACTICAL_H, int(round(fy + 0.15 * h_px)))
    bg_crop = img_bgr[by1:by2, bx1:bx2]
    bb, bg_g, br = bg_crop[:, :, 0].astype(np.float32), bg_crop[:, :, 1].astype(np.float32), bg_crop[:, :, 2].astype(np.float32)
    bg_grass_mask = ((bg_g - np.maximum(br, bb)) > 8.0) & (bg_g > 50.0)

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)

    L = lab[:, :, 0][non_grass]
    A = lab[:, :, 1][non_grass]
    B = lab[:, :, 2][non_grass]
    H = hsv[:, :, 0][non_grass]
    S = hsv[:, :, 1][non_grass]
    V = hsv[:, :, 2][non_grass]

    if int(bg_grass_mask.sum()) >= 30:
        bg_lab = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_hsv = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        L_bg = float(np.median(bg_lab[:, :, 0][bg_grass_mask]))
        V_bg = float(np.median(bg_hsv[:, :, 2][bg_grass_mask]))
    else:
        L_bg, V_bg = 120.0, 120.0

    # Reject neon yellow referees & lime goalkeepers cleanly on chest box
    neon_yellow = float(np.mean((H >= 17.0) & (H <= 44.0) & (S >= 100.0) & (V >= 115.0)))
    lime_gk = float(np.mean((H >= 42.0) & (H <= 85.0) & (S >= 90.0) & (V >= 115.0)))
    if neon_yellow > 0.22 or lime_gk > 0.25:
        return None, 0.0

    L_rel = L - L_bg
    V_rel = V - V_bg
    dark_rel_frac = float(np.mean(L_rel < -5.0))
    bright_rel_frac = float(np.mean(L_rel > 22.0))
    feat = np.array([
        float(np.median(L_rel)),
        float(np.percentile(L_rel, 75)),
        float(np.median(V_rel)),
        float(np.median(A)),
        float(np.median(B)),
        float(np.median(S)),
        float(np.median(r[non_grass] - g[non_grass])),
        float(np.median(b[non_grass] - g[non_grass])),
        dark_rel_frac,
        bright_rel_frac,
    ], dtype=np.float32)
    return feat, valid_ratio


def _fit_logistic_np(X: np.ndarray, y: np.ndarray, l2: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Lightweight deterministic L2 logistic regression trained in NumPy (zero sklearn dependency needed)."""
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-4)
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1], dtype=np.float64)
    b = 0.0
    n = float(Xs.shape[0])
    lr = 0.25
    for _ in range(350):
        z = np.clip(Xs @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        gw = (Xs.T @ err) / n + (l2 / n) * w
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb
    return mu.astype(np.float32), sd.astype(np.float32), w.astype(np.float32), float(b)


def _predict_jersey_prob(feat: np.ndarray, model_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> float:
    mu, sd, w, b = model_tuple
    z = float(np.dot((feat - mu) / sd, w) + b)
    z = float(np.clip(z, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-z)))  # P(Home)


def _tracklet_runs(visible: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([np.int8(0)], visible.astype(np.int8), [np.int8(0)])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def run_jersey_audit() -> None:
    t0 = time.time()
    print("=== Sprint 4: Self-Supervised Jersey Disambiguation Audit ===")
    match = _inject_cv_ball_fast(load_atd(verbose=False))
    H_mats, _, valid_h = load_homography.load()
    pm = play_mask(match)
    px, py = match["px"], match["py"]
    bx, by = match["ball_x"], match["ball_y"]
    team_map = match["team_map"]
    pitch_w, pitch_h = match["pitch"]
    n_frames, n_slots = px.shape

    vpath = games.video_path(game="BAR-ATH", source="tactical")
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vpath}")

    # Step 1: Collect isolated interior samples strictly outside public GT (fr > 19000) to fit classifier
    fit_frames = [fr for fr in range(19500, n_frames, 360) if pm[fr] and valid_h[fr]]
    heldout_frames = [fr for fr in range(11185, 19000, 120) if pm[fr] and valid_h[fr]]
    all_sample_frames = sorted(set(fit_frames + heldout_frames + [20434]))

    frame_cache: dict[int, np.ndarray] = {}
    cur_pos = -1000
    for fr in all_sample_frames:
        if fr - cur_pos > 75 or fr < cur_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            cur_pos = fr - 1
        while cur_pos < fr - 1:
            cap.grab()
            cur_pos += 1
        ok, img = cap.read()
        cur_pos += 1
        if ok:
            frame_cache[fr] = img

    def sample_isolated(frames_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for fr in frames_list:
            img = frame_cache.get(fr)
            if img is None:
                continue
            vis_slots = np.flatnonzero(np.isfinite(px[fr]) & np.isfinite(py[fr]))
            if len(vis_slots) < 6:
                continue
            coords_m = np.column_stack([px[fr, vis_slots] * pitch_w, py[fr, vis_slots] * pitch_h])
            dmat = np.hypot(coords_m[:, 0:1] - coords_m[None, :, 0], coords_m[:, 1:2] - coords_m[None, :, 1])
            np.fill_diagonal(dmat, 1e9)
            nn_dist = dmat.min(axis=1)
            for idx_k, s in enumerate(vis_slots):
                nx, ny = float(px[fr, s]), float(py[fr, s])
                # Avoid goal areas where goalkeepers stand (0.14..0.86) and crowded overlaps (<3.2m)
                if nn_dist[idx_k] < 3.2 or not (0.14 <= nx <= 0.86 and 0.04 <= ny <= 0.96):
                    continue
                fx, fy, sx = _player_px_scale(H_mats[fr], nx, ny, pitch_w)
                feat, _ = extract_torso_features(img, fx, fy, sx)
                if feat is not None:
                    Xs.append(feat)
                    ys.append(1.0 if int(team_map[s]) == 0 else 0.0)
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float64)

    X_fit, y_fit = sample_isolated(fit_frames)
    model_tuple = _fit_logistic_np(X_fit, y_fit)
    p_fit = np.array([_predict_jersey_prob(x, model_tuple) for x in X_fit])
    acc_fit = float(((p_fit >= 0.5) == (y_fit >= 0.5)).mean())

    X_hold, y_hold = sample_isolated(heldout_frames)
    p_hold = np.array([_predict_jersey_prob(x, model_tuple) for x in X_hold])
    acc_hold = float(((p_hold >= 0.5) == (y_hold >= 0.5)).mean())
    acc_hold_confident = float(((p_hold >= 0.5) == (y_hold >= 0.5))[(p_hold >= 0.80) | (p_hold <= 0.20)].mean())
    frac_confident = float(((p_hold >= 0.80) | (p_hold <= 0.20)).mean())
    print(
        f"[Self-Sup Fit] train_samples={len(y_fit)} fit_acc={acc_fit:.4f} | "
        f"opening_5m_heldout n={len(y_hold)} acc={acc_hold:.4f} (conf>=0.80 acc={acc_hold_confident:.4f}, cov={frac_confident:.3f})"
    )

    # Save visual probe f=20434 showing torso boxes & jersey prob
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    if 20434 in frame_cache:
        vis_im = frame_cache[20434].copy()
        for s in range(n_slots):
            nx, ny = float(px[20434, s]), float(py[20434, s])
            if np.isfinite(nx) and np.isfinite(ny):
                fx, fy, sx = _player_px_scale(H_mats[20434], nx, ny, pitch_w)
                box = torso_patch_coords(fx, fy, sx)
                feat, _ = extract_torso_features(frame_cache[20434], fx, fy, sx)
                if box is not None and feat is not None:
                    p_h = _predict_jersey_prob(feat, model_tuple)
                    col = (255, 80, 50) if p_h >= 0.5 else (230, 210, 255)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(vis_im, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(vis_im, f"{'H' if team_map[s]==0 else 'A'}:{p_h:.2f}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        cv2.imwrite(str(PREVIEW_DIR / "probe_f020434_jerseys.jpg"), vis_im)
    frame_cache.clear()

    # Step 2: Audit tracklet swaps AND tight duels across full match
    # Identify tight duels (|d_h - d_a| < 1.5 m & min_d < 2.8 m) + subsample frames cleanly
    home_mask = team_map == 0
    away_mask = team_map == 1
    dx_all = (px - bx[:, None]) * pitch_w
    dy_all = (py - by[:, None]) * pitch_h
    dist_all = np.hypot(dx_all, dy_all)
    fin_dist = np.isfinite(dist_all)
    dist_h = np.where(home_mask[None, :] & fin_dist, dist_all, np.inf)
    dist_a = np.where(away_mask[None, :] & fin_dist, dist_all, np.inf)
    idx_h = np.argmin(dist_h, axis=1)
    idx_a = np.argmin(dist_a, axis=1)
    d_h = dist_h[np.arange(n_frames), idx_h]
    d_a = dist_a[np.arange(n_frames), idx_a]
    min_d = np.minimum(d_h, d_a)
    vis_b = np.isfinite(bx) & np.isfinite(by)
    both_fin = np.isfinite(d_h) & np.isfinite(d_a)
    diff_ha = np.where(both_fin, np.abs(np.where(both_fin, d_h, 0.0) - np.where(both_fin, d_a, 0.0)), np.inf)
    tight_duel = pm & vis_b & (min_d < 2.8) & (diff_ha < 1.5)
    print(f"[Targets] Tight duel frames (with CV ball): {int(tight_duel.sum())}")

    # Collect tracklet swap probes (every 25 frames on active play for clean fast audit)
    tracklets: list[tuple[int, int, int]] = []  # (slot, a, b)
    for s in range(n_slots):
        vis_s = pm & np.isfinite(px[:, s]) & np.isfinite(py[:, s])
        for a, b in _tracklet_runs(vis_s):
            if b - a >= 25:
                tracklets.append((s, a, b))

    sample_job_frames = [fr for fr in range(n_frames) if pm[fr] and valid_h[fr] and (fr % 25 == 0)]
    print(f"[Video Sweep] Scanning {len(sample_job_frames)} frames for tracklet swaps & duel visual check...")

    # Pre-build fast frame -> job lookup
    duel_set = set(np.flatnonzero(tight_duel).tolist())
    # Tracklet scores: tr_idx -> list[(fr, p_h)]
    slot_track_id = np.full((n_frames, n_slots), -1, dtype=np.int32)
    for tid, (s, a, b) in enumerate(tracklets):
        slot_track_id[a:b, s] = tid
    track_probs: list[list[float]] = [[] for _ in range(len(tracklets))]

    duel_records: list[dict] = []  # sampled tight duel checks
    cur_pos = -1000
    saved_swaps = 0
    for idx_j, fr in enumerate(sample_job_frames):
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

        # Score isolated tracklets on %25 grid
        if fr % 25 == 0:
            vis_slots = np.flatnonzero(np.isfinite(px[fr]) & np.isfinite(py[fr]))
            if len(vis_slots) >= 4:
                coords_m = np.column_stack([px[fr, vis_slots] * pitch_w, py[fr, vis_slots] * pitch_h])
                dmat = np.hypot(coords_m[:, 0:1] - coords_m[None, :, 0], coords_m[:, 1:2] - coords_m[None, :, 1])
                np.fill_diagonal(dmat, 1e9)
                nn_dist = dmat.min(axis=1)
                for idx_k, s in enumerate(vis_slots):
                    tid = int(slot_track_id[fr, s])
                    if tid < 0 or nn_dist[idx_k] < 2.2:
                        continue
                    nx, ny = float(px[fr, s]), float(py[fr, s])
                    if not (0.08 <= nx <= 0.92 and 0.03 <= ny <= 0.97):
                        continue
                    fx, fy, sx = _player_px_scale(H_mats[fr], nx, ny, pitch_w)
                    feat, _ = extract_torso_features(img, fx, fy, sx)
                    if feat is not None:
                        track_probs[tid].append(_predict_jersey_prob(feat, model_tuple))

        # Score closest H & A slots on tight duels
        if fr in duel_set:
            sh, sa = int(idx_h[fr]), int(idx_a[fr])
            fxh, fyh, sxh = _player_px_scale(H_mats[fr], float(px[fr, sh]), float(py[fr, sh]), pitch_w)
            fxa, fya, sxa = _player_px_scale(H_mats[fr], float(px[fr, sa]), float(py[fr, sa]), pitch_w)
            fh, _ = extract_torso_features(img, fxh, fyh, sxh)
            fa, _ = extract_torso_features(img, fxa, fya, sxa)
            ph_h = _predict_jersey_prob(fh, model_tuple) if fh is not None else np.nan
            ph_a = _predict_jersey_prob(fa, model_tuple) if fa is not None else np.nan
            duel_records.append({
                "frame": fr, "sh": sh, "sa": sa,
                "dh": float(d_h[fr]), "da": float(d_a[fr]),
                "ph_labeled_h": float(ph_h), "ph_labeled_a": float(ph_a),
            })

            # Save a few duel crops
            if saved_swaps < 6 and np.isfinite(ph_h) and np.isfinite(ph_a):
                if ph_h < 0.25 or ph_a > 0.75:
                    bpx, bpy = load_homography.project(H_mats[fr], float(bx[fr]), float(by[fr]))
                    cx0 = int(np.clip(round(float(bpx)) - 180, 0, TACTICAL_W - 360))
                    cy0 = int(np.clip(round(float(bpy)) - 180, 0, TACTICAL_H - 360))
                    crop = img[cy0 : cy0 + 360, cx0 : cx0 + 360].copy()
                    for tag_l, fxp, fyp, pprob, tcol in (
                        ("H", fxh, fyh, ph_h, (255, 80, 50)),
                        ("A", fxa, fya, ph_a, (230, 210, 255)),
                    ):
                        box = torso_patch_coords(fxp, fyp, sxh)
                        if box is not None:
                            x1, y1, x2, y2 = box[0] - cx0, box[1] - cy0, box[2] - cx0, box[3] - cy0
                            cv2.rectangle(crop, (x1, y1), (x2, y2), tcol, 2)
                            cv2.putText(crop, f"ATD:{tag_l} pH={pprob:.2f}", (max(4, x1), max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tcol, 1)
                    cv2.circle(crop, (int(round(float(bpx) - cx0)), int(round(float(bpy) - cy0))), 6, (0, 255, 0), 2)
                    cv2.imwrite(str(PREVIEW_DIR / f"duel_mismatch_{saved_swaps:02d}_f{fr:06d}.jpg"), crop)
                    saved_swaps += 1

        if (idx_j + 1) % 3000 == 0 or idx_j + 1 == len(sample_job_frames):
            print(f"  [Sweep {idx_j + 1}/{len(sample_job_frames)}] elapsed={time.time() - t0:.1f}s")

    cap.release()

    # Summarize tracklet-level ATD team_map swaps!
    n_swapped_tracklets = 0
    swapped_frames = 0
    total_eval_tracklets = 0
    total_eval_frames = 0
    swapped_list = []
    for tid, (s, a, b) in enumerate(tracklets):
        probs = track_probs[tid]
        obs_frac = len(probs) / max(1.0, (b - a) / 25.0)
        if len(probs) < 8 or obs_frac < 0.30:
            continue
        # Exclude pre-H2 warmup bib interval (players wear dark jackets over away shorts before whistle)
        if a >= 110808 and b <= 113000:
            continue
        total_eval_tracklets += 1
        total_eval_frames += (b - a)
        med_p = float(np.median(probs))
        atd_team = int(team_map[s])
        wrong = (atd_team == 0 and med_p <= 0.10) or (atd_team == 1 and med_p >= 0.90 and obs_frac >= 0.65)
        if wrong:
            n_swapped_tracklets += 1
            swapped_frames += (b - a)
            swapped_list.append({
                "slot": s, "team": atd_team, "a": a, "b": b, "len": b - a,
                "n_obs": len(probs), "obs_frac": round(obs_frac, 2), "med_pH": round(med_p, 3),
            })

    # Summarize tight duel checks
    both_ok = [r for r in duel_records if np.isfinite(r["ph_labeled_h"]) and np.isfinite(r["ph_labeled_a"])]
    h_looks_a = sum(1 for r in both_ok if r["ph_labeled_h"] <= 0.25)
    a_looks_h = sum(1 for r in both_ok if r["ph_labeled_a"] >= 0.75)
    both_swapped = sum(1 for r in both_ok if r["ph_labeled_h"] <= 0.25 and r["ph_labeled_a"] >= 0.75)

    # Save per-tracklet swap mask & duel info cleanly
    swap_mask = np.zeros((n_frames, n_slots), dtype=bool)
    for item in swapped_list:
        swap_mask[item["a"] : item["b"], item["slot"]] = True

    duel_fr = np.array([r["frame"] for r in duel_records], dtype=np.int32)
    duel_ph_h = np.array([r["ph_labeled_h"] for r in duel_records], dtype=np.float32)
    duel_ph_a = np.array([r["ph_labeled_a"] for r in duel_records], dtype=np.float32)
    np.savez_compressed(
        OUT_NPZ,
        swap_mask=swap_mask,
        tight_duel=tight_duel,
        duel_frames=duel_fr,
        duel_ph_h=duel_ph_h,
        duel_ph_a=duel_ph_a,
    )
    audit = {
        "elapsed_s": round(time.time() - t0, 1),
        "fit_acc": round(acc_fit, 4),
        "heldout_5m_acc": round(acc_hold, 4),
        "heldout_5m_conf80_acc": round(acc_hold_confident, 4),
        "heldout_5m_conf80_cov": round(frac_confident, 4),
        "eval_tracklets": total_eval_tracklets,
        "eval_tracklet_frames": total_eval_frames,
        "swapped_tracklets": n_swapped_tracklets,
        "swapped_tracklet_frames": swapped_frames,
        "tight_duels_sampled": len(duel_records),
        "tight_duels_both_valid": len(both_ok),
        "duel_labeled_H_wears_A": h_looks_a,
        "duel_labeled_A_wears_H": a_looks_h,
        "duel_both_swapped": both_swapped,
        "top_swaps": swapped_list[:20],
    }
    (PREVIEW_DIR / "jersey_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[Sprint 4 Audit DONE] {json.dumps(audit, indent=2)}")


def repair_team_swaps(match: dict, npz_path: Path = OUT_NPZ) -> dict:
    """Move verified visual team-swapped tracklet frames into matching-team spare slots."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Jersey audit file not found: {npz_path}")
    swap_mask = np.load(npz_path)["swap_mask"]
    n_frames, n_slots = match["px"].shape
    if swap_mask.shape != (n_frames, n_slots):
        raise ValueError(f"swap_mask shape {swap_mask.shape} != {(n_frames, n_slots)}")
    if not swap_mask.any():
        return match
    px = match["px"].astype(np.float32).copy()
    py = match["py"].astype(np.float32).copy()
    team_map = match["team_map"].copy()
    # Allocate 4 spare slots [0, 0, 1, 1] for reassigned frames
    extra_x = np.full((n_frames, 4), np.nan, dtype=np.float32)
    extra_y = np.full((n_frames, 4), np.nan, dtype=np.float32)
    moved = 0
    for s in range(n_slots):
        rows = np.flatnonzero(swap_mask[:, s])
        if len(rows) == 0:
            continue
        target_team = 1 - int(team_map[s])
        cands = (0, 1) if target_team == 0 else (2, 3)
        for r in rows:
            for c_idx in cands:
                if not np.isfinite(extra_x[r, c_idx]):
                    extra_x[r, c_idx] = px[r, s]
                    extra_y[r, c_idx] = py[r, s]
                    break
            px[r, s], py[r, s] = np.nan, np.nan
            moved += 1
    out = dict(match)
    out["px"] = np.concatenate([px, extra_x], axis=1)
    out["py"] = np.concatenate([py, extra_y], axis=1)
    out["team_map"] = np.concatenate([team_map, np.array([0, 0, 1, 1], dtype=team_map.dtype)])
    return out


if __name__ == "__main__":
    run_jersey_audit()
