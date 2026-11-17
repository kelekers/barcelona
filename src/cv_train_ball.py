r"""Sprint 2: Train YOLO26s Stage-A (Roboflow unstretched) & Stage-B (Opsi B BAR-ATH self-supervised).

Usage:
    .\.venv\Scripts\python.exe src/cv_train_ball.py --stage A
    .\.venv\Scripts\python.exe src/cv_train_ball.py --stage B
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "starter"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import games  # noqa: E402
import load_homography  # noqa: E402
from src.atd_io import load_atd  # noqa: E402
from src.smart_priors import compute_physical_cues  # noqa: E402

CACHE_DIR = ROOT / "data" / "cache"
STAGE_A_DIR = CACHE_DIR / "cv_ball_stage_a"
STAGE_A_YAML = STAGE_A_DIR / "data.yaml"
STAGE_B_DIR = CACHE_DIR / "cv_ball_stage_b"
STAGE_B_YAML = STAGE_B_DIR / "data.yaml"
RUNS_DIR = CACHE_DIR / "cv_runs"
BASE_WEIGHTS = CACHE_DIR / "yolo26s.pt"
BAR_PROBE_DIR = CACHE_DIR / "cv_ball_bar_probe"
MODELS_DIR = ROOT / "models"

TILE_SIZE = 640
TACTICAL_W = 1920
TACTICAL_H = 1080
TILE_XS = (0, 427, 853, 1280)
TILE_YS = (0, 440)

BAR_PUBLIC_GT_END = 19000
BAR_H1_END = 90500
BAR_H2_TRAIN_START = 111200
BAR_H2_TRAIN_END = 169000
BAR_VAL_START = 172000
BAR_VAL_END = 190000


def _link_or_copy(src: Path, dst: Path) -> None:
    """Create a hardlink to save disk space, falling back to copy if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_yaml(path: Path, root_dir: Path, train_rel: str, val_rel: str, test_rel: str | None = None) -> None:
    lines = [
        f"path: {root_dir.as_posix()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
    ]
    if test_rel:
        lines.append(f"test: {test_rel}")
    lines.extend(["nc: 1", "names: ['ball']"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_boxes_from_result(result) -> list[tuple[float, float, float, float, float]]:
    """Return list of (x1, y1, x2, y2, conf) from an Ultralytics Result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    return [
        (float(x1), float(y1), float(x2), float(y2), float(c))
        for (x1, y1, x2, y2), c in zip(xyxy, confs)
    ]


def _merge_overlapping_detections(
    dets: list[tuple[float, float, float, float, float]],
    iou_thresh: float = 0.30,
    center_thresh: float = 14.0,
) -> list[tuple[float, float, float, float, float]]:
    """Greedy NMS merging both overlapping IoU boxes and close tiny-ball centers."""
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: d[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    for cand in ordered:
        cx1, cy1, cx2, cy2, _ = cand
        ccx, ccy = 0.5 * (cx1 + cx2), 0.5 * (cy1 + cy2)
        carea = max(1e-6, (cx2 - cx1) * (cy2 - cy1))
        duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2, _ = k
            kcx, kcy = 0.5 * (kx1 + kx2), 0.5 * (ky1 + ky2)
            if ((ccx - kcx) ** 2 + (ccy - kcy) ** 2) ** 0.5 <= center_thresh:
                duplicate = True
                break
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter > 0.0:
                karea = max(1e-6, (kx2 - kx1) * (ky2 - ky1))
                if inter / (carea + karea - inter) >= iou_thresh:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(cand)
    return kept


def predict_tiled_frame(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    max_wh: float = 56.0,
) -> list[tuple[float, float, float, float, float]]:
    """Run batched 8-tile 640x640 inference on a 1920x1080 tactical frame."""
    tiles = []
    offsets = []
    for y0 in TILE_YS:
        for x0 in TILE_XS:
            tiles.append(img_bgr[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE])
            offsets.append((x0, y0))
    results = model.predict(
        tiles, imgsz=TILE_SIZE, batch=len(tiles), conf=conf, iou=0.45, device=0, verbose=False
    )
    raw: list[tuple[float, float, float, float, float]] = []
    for (x0, y0), res in zip(offsets, results):
        for x1, y1, x2, y2, c in _extract_boxes_from_result(res):
            if (x2 - x1) <= max_wh and (y2 - y1) <= max_wh:
                raw.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, c))
    return _merge_overlapping_detections(raw)


def _save_probe_visuals(
    model: YOLO,
    out_dir: Path,
    probe_dir: Path = BAR_PROBE_DIR,
    conf: float = 0.20,
) -> list[dict]:
    """Run tiled inference on 16 BAR-ATH 1080p probe frames and save annotated crops + full frames."""
    if not probe_dir.exists():
        print(f"[WARN] Probe directory not found: {probe_dir}")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    match = load_atd(verbose=False)
    H_mats, _, valid_h = load_homography.load()
    pitch_w, pitch_h = match["pitch"]
    probe_files = sorted(probe_dir.glob("frame_*.jpg"))
    summary = []
    for img_path in probe_files:
        stem = img_path.stem
        try:
            fr = int(stem.split("_")[-1])
        except ValueError:
            fr = -1
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = predict_tiled_frame(model, img, conf=conf)
        atd_px = None
        if 0 <= fr < match["n_frames"] and valid_h[fr]:
            bx, by = float(match["ball_x"][fr]), float(match["ball_y"][fr])
            if np.isfinite(bx) and np.isfinite(by):
                px_x, px_y = load_homography.project(H_mats[fr], bx, by)
                atd_px = (float(px_x), float(px_y))

        vis = img.copy()
        if atd_px is not None:
            ax, ay = int(round(atd_px[0])), int(round(atd_px[1]))
            cv2.drawMarker(vis, (ax, ay), (0, 220, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(
                vis, f"ATD", (ax + 12, max(24, ay - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA
            )
        best_dist_m = None
        for idx, (x1, y1, x2, y2, c) in enumerate(dets):
            color = (0, 255, 0) if idx == 0 else (255, 220, 0)
            cv2.rectangle(vis, (int(round(x1)) - 2, int(round(y1)) - 2), (int(round(x2)) + 2, int(round(y2)) + 2), color, 2)
            cv2.putText(
                vis, f"{c:.2f}", (int(round(x1)), max(20, int(round(y1)) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )
            if idx == 0 and 0 <= fr < match["n_frames"] and valid_h[fr] and atd_px is not None:
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                ux, uy = load_homography.unproject(H_mats[fr], cx, cy)
                dx = (float(ux) - float(match["ball_x"][fr])) * pitch_w
                dy = (float(uy) - float(match["ball_y"][fr])) * pitch_h
                best_dist_m = float((dx * dx + dy * dy) ** 0.5)

        cv2.imwrite(str(out_dir / f"{stem}_full.jpg"), vis)

        # Save zoomed 360x360 ROI crop around best detection or ATD so tiny balls are easy to verify visually
        focus = None
        if dets:
            x1, y1, x2, y2, _ = dets[0]
            focus = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        elif atd_px is not None:
            focus = atd_px
        if focus is not None:
            fx = int(np.clip(round(focus[0]) - 180, 0, TACTICAL_W - 360))
            fy = int(np.clip(round(focus[1]) - 180, 0, TACTICAL_H - 360))
            crop = vis[fy : fy + 360, fx : fx + 360]
            cv2.imwrite(str(out_dir / f"{stem}_crop.jpg"), crop)

        rec = {
            "frame": fr,
            "n_dets": len(dets),
            "top_conf": round(dets[0][4], 4) if dets else 0.0,
            "top_wh": [round(dets[0][2] - dets[0][0], 1), round(dets[0][3] - dets[0][1], 1)] if dets else None,
            "atd_px": [round(atd_px[0], 1), round(atd_px[1], 1)] if atd_px else None,
            "top_px": [round(0.5 * (dets[0][0] + dets[0][2]), 1), round(0.5 * (dets[0][1] + dets[0][3]), 1)] if dets else None,
            "dist_to_atd_m": round(best_dist_m, 3) if best_dist_m is not None else None,
        }
        summary.append(rec)
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    n_hit = sum(1 for r in summary if r["n_dets"] > 0)
    print(f"[PROBE] {out_dir.name}: detected ball on {n_hit}/{len(summary)} 1080p probe frames")
    for r in summary:
        print(f"  f={r['frame']:6d} | dets={r['n_dets']} top={r['top_conf']:.3f} wh={r['top_wh']} d_atd={r['dist_to_atd_m']}m")
    return summary


def _train_yolo(
    weights_path: Path,
    data_yaml: Path,
    run_name: str,
    epochs: int,
    lr0: float = 0.01,
    patience: int = 10,
    close_mosaic: int | None = None,
) -> Path:
    """Train YOLO26s deterministically on tiny-ball friendly augmentations."""
    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if close_mosaic is None:
        close_mosaic = min(8, max(2, epochs // 4))
    model = YOLO(str(weights_path))
    model.train(
        data=str(data_yaml),
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        epochs=epochs,
        patience=patience,
        imgsz=TILE_SIZE,
        batch=16,
        workers=4,
        device=0,
        amp=True,
        optimizer="auto",
        lr0=lr0,
        mixup=0.0,
        scale=0.25,
        mosaic=0.5,
        close_mosaic=close_mosaic,
        degrees=0.0,
        fliplr=0.5,
        flipud=0.0,
        cache=False,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"
    return best_pt


def _eval_split(weights: Path, data_yaml: Path, split: str, name_tag: str) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=TILE_SIZE,
        batch=16,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
    )
    out = {
        "split": split,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
    }
    print(f"[VAL:{name_tag}:{split}] P={out['precision']:.4f} R={out['recall']:.4f} mAP50={out['mAP50']:.4f} mAP50-95={out['mAP50_95']:.4f}")
    return out


def run_stage_a() -> Path:
    if not STAGE_A_YAML.exists():
        raise FileNotFoundError(f"Stage-A dataset missing: {STAGE_A_YAML}. Run src/cv_ball_data.py first.")
    if not BASE_WEIGHTS.exists():
        raise FileNotFoundError(f"Base weights missing: {BASE_WEIGHTS}")

    print("=== Sprint 2 Stage-A: Training YOLO26s on clean unstretched football_tactical ===")
    best_a = _train_yolo(BASE_WEIGHTS, STAGE_A_YAML, "stage_a_yolo26s", epochs=35, lr0=0.01)
    m_val = _eval_split(best_a, STAGE_A_YAML, "val", "StageA")
    m_test = _eval_split(best_a, STAGE_A_YAML, "test", "StageA")
    model_a = YOLO(str(best_a))
    probe_summary = _save_probe_visuals(model_a, RUNS_DIR / "stage_a_probe", conf=0.20)
    (RUNS_DIR / "stage_a_yolo26s" / "eval_summary.json").write_text(
        json.dumps({"val": m_val, "test": m_test, "probe": probe_summary}, indent=2), encoding="utf-8"
    )
    print(f"[Stage-A DONE] Best weight: {best_a}")
    return best_a


def _select_bar_candidates() -> tuple[list[int], list[int]]:
    """Select clean isolated active-play BAR-ATH frames strictly outside public GT."""
    match = load_atd(verbose=False)
    cues = compute_physical_cues(match)
    H_mats, _, valid_h = load_homography.load()

    bx_arr = match["ball_x"]
    by_arr = match["ball_y"]
    vis = cues["vis"]
    oob = cues["oob"]
    min_d = np.minimum(cues["d_h"], cues["d_a"])
    b_spd = cues["b_spd"]
    n_frames = match["n_frames"]

    def collect_regime(
        start_f: int,
        end_f: int,
        min_d_lo: float,
        min_d_hi: float,
        spd_lo: float,
        spd_hi: float,
        min_step: int,
        max_keep: int,
    ) -> list[int]:
        out = []
        last_kept = -10**9
        for fr in range(max(BAR_PUBLIC_GT_END + 100, start_f + 3), min(n_frames - 3, end_f), 1):
            if fr - last_kept < min_step:
                continue
            ok = True
            for t in (fr - 2, fr, fr + 2):
                if not (
                    valid_h[t]
                    and vis[t]
                    and (not oob[t])
                    and min_d_lo <= min_d[t] < min_d_hi
                    and spd_lo <= b_spd[t] <= spd_hi
                ):
                    ok = False
                    break
                px_x, px_y = load_homography.project(H_mats[t], float(bx_arr[t]), float(by_arr[t]))
                if not (64.0 <= px_x <= TACTICAL_W - 64.0 and 64.0 <= px_y <= TACTICAL_H - 64.0):
                    ok = False
                    break
            if ok:
                out.append(fr)
                last_kept = fr
        if len(out) > max_keep:
            idx = np.linspace(0, len(out) - 1, max_keep, dtype=int)
            out = [out[i] for i in idx]
        return out

    open_train = collect_regime(BAR_PUBLIC_GT_END + 500, BAR_H1_END, 2.0, 999.0, 1.5, 22.0, 18, 450) + collect_regime(
        BAR_H2_TRAIN_START, BAR_H2_TRAIN_END, 2.0, 999.0, 1.5, 22.0, 18, 450
    )
    near_train = collect_regime(BAR_PUBLIC_GT_END + 500, BAR_H1_END, 0.20, 2.0, 0.5, 18.0, 16, 450) + collect_regime(
        BAR_H2_TRAIN_START, BAR_H2_TRAIN_END, 0.20, 2.0, 0.5, 18.0, 16, 450
    )
    open_val = collect_regime(BAR_VAL_START, BAR_VAL_END, 2.0, 999.0, 1.5, 22.0, 18, 160)
    near_val = collect_regime(BAR_VAL_START, BAR_VAL_END, 0.20, 2.0, 0.5, 18.0, 16, 160)

    regime_map: dict[int, str] = {}
    for f in open_train + open_val:
        regime_map[f] = "open"
    for f in near_train + near_val:
        regime_map[f] = "near"
    train_anchors = sorted(set(open_train + near_train))
    val_anchors = sorted(set(open_val + near_val))
    return train_anchors, val_anchors, regime_map


def _best_box_near_atd(
    dets: list[tuple[float, float, float, float, float]],
    H_mat: np.ndarray,
    bx_norm: float,
    by_norm: float,
    pitch_w: float,
    pitch_h: float,
    max_dist_m: float = 2.6,
    min_conf: float = 0.42,
    min_margin: float = 0.0,
) -> tuple[float, float, float, float, float] | None:
    """Return unique unambiguous Stage-A ball box matching size constraints and pitch proximity to ATD."""
    cands: list[tuple[float, tuple[float, float, float, float, float]]] = []
    near_cluster: list[tuple[float, float, float]] = []
    for x1, y1, x2, y2, conf in dets:
        w, h = x2 - x1, y2 - y1
        if not (4.0 <= w <= 28.0 and 4.0 <= h <= 28.0):
            continue
        ratio = w / max(1e-6, h)
        if not (0.50 <= ratio <= 2.0):
            continue
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        ux, uy = load_homography.unproject(H_mat, cx, cy)
        dx = (float(ux) - bx_norm) * pitch_w
        dy = (float(uy) - by_norm) * pitch_h
        dist_m = (dx * dx + dy * dy) ** 0.5
        if dist_m <= max(3.6, max_dist_m + 1.2):
            near_cluster.append((cx, cy, conf))
        if conf >= min_conf and dist_m <= max_dist_m:
            score = conf - 0.03 * dist_m
            cands.append((score, (x1, y1, x2, y2, conf)))
    if not cands:
        return None
    cands.sort(key=lambda item: item[0], reverse=True)
    best = cands[0][1]
    if min_margin > 0.0:
        bcx, bcy = 0.5 * (best[0] + best[2]), 0.5 * (best[1] + best[3])
        for ox, oy, oconf in near_cluster:
            if ((ox - bcx) ** 2 + (oy - bcy) ** 2) ** 0.5 > 12.0:
                if oconf >= best[4] - min_margin:
                    return None
    return best


def _harvest_bar_opsi_b(stage_a_weights: Path) -> tuple[dict, dict]:
    """Harvest Stage-A + 3-frame collinear verified BAR-ATH tiles and build Stage-B dataset."""
    if STAGE_B_DIR.exists():
        shutil.rmtree(STAGE_B_DIR)
    for split in ("train", "val", "val_bar", "val_rf", "test"):
        (STAGE_B_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (STAGE_B_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Link balanced Stage-A RF replay (2500 train so BAR-ATH ~35-40% isn't drowned out) + all RF val/test
    rf_train_imgs = sorted((STAGE_A_DIR / "images" / "train").glob("*.jpg"))
    if len(rf_train_imgs) > 2500:
        keep_idx = np.linspace(0, len(rf_train_imgs) - 1, 2500, dtype=int)
        rf_train_imgs = [rf_train_imgs[i] for i in keep_idx]
    for img_p in rf_train_imgs:
        lbl_p = STAGE_A_DIR / "labels" / "train" / f"{img_p.stem}.txt"
        _link_or_copy(img_p, STAGE_B_DIR / "images" / "train" / f"rf_{img_p.name}")
        if lbl_p.exists():
            _link_or_copy(lbl_p, STAGE_B_DIR / "labels" / "train" / f"rf_{lbl_p.name}")

    for rf_split, dst_splits in (("val", ("val_rf", "val")), ("test", ("test",))):
        src_img_dir = STAGE_A_DIR / "images" / rf_split
        src_lbl_dir = STAGE_A_DIR / "labels" / rf_split
        for img_p in sorted(src_img_dir.glob("*.jpg")):
            lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
            for ds in dst_splits:
                _link_or_copy(img_p, STAGE_B_DIR / "images" / ds / f"rf_{img_p.name}")
                if lbl_p.exists():
                    _link_or_copy(lbl_p, STAGE_B_DIR / "labels" / ds / f"rf_{lbl_p.name}")

    train_anchors, val_anchors, regime_map = _select_bar_candidates()
    all_tasks = [(fr, "train") for fr in train_anchors] + [(fr, "val_bar") for fr in val_anchors]
    all_tasks.sort(key=lambda item: item[0])
    print(f"[Opsi-B] RF replay train={len(rf_train_imgs)} | Candidate BAR triplets: train={len(train_anchors)}, val_bar={len(val_anchors)}")

    needed_frames = sorted({t for fr, _ in all_tasks for t in (fr - 2, fr, fr + 2)})
    needed_set = set(needed_frames)

    vpath = games.video_path(game="BAR-ATH", source="tactical")
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vpath}")

    match = load_atd(verbose=False)
    H_mats, _, _ = load_homography.load()
    pitch_w, pitch_h = match["pitch"]
    bx_arr, by_arr = match["ball_x"], match["ball_y"]
    px_arr, py_arr = match["px"], match["py"]
    model_a = YOLO(str(stage_a_weights))
    rng = np.random.default_rng(42)

    # Linear sequential pass over needed frames with smart grab/seek
    frame_buf: dict[int, np.ndarray] = {}
    stats = {
        "train_open": 0, "train_near": 0, "train_neg": 0, "train_fp": 0,
        "val_bar_open": 0, "val_bar_near": 0, "val_bar_neg": 0, "val_bar_fp": 0,
        "rejected": 0,
    }
    preview_dir = RUNS_DIR / "stage_b_harvest_preview"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved_previews = {"open": 0, "near": 0}

    def process_triplet(fr: int, split: str) -> None:
        imgs = [frame_buf.pop(t) for t in (fr - 2, fr, fr + 2)]
        regime = regime_map.get(fr, "open")
        is_near = regime == "near"
        px_x, px_y = load_homography.project(H_mats[fr], float(bx_arr[fr]), float(by_arr[fr]))
        jx = int(rng.integers(-210, 211))
        jy = int(rng.integers(-210, 211))
        x0 = int(np.clip(round(px_x - TILE_SIZE / 2 + jx), 0, TACTICAL_W - TILE_SIZE))
        y0 = int(np.clip(round(px_y - TILE_SIZE / 2 + jy), 0, TACTICAL_H - TILE_SIZE))

        tiles = [im[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] for im in imgs]
        results = model_a.predict(tiles, imgsz=TILE_SIZE, batch=3, conf=0.32, iou=0.45, device=0, verbose=False)

        max_dm = 1.8 if is_near else 2.6
        min_marg = 0.14 if is_near else 0.08
        max_err_px = 6.5 if is_near else 8.5

        picked: list[tuple[float, float, float, float, float] | None] = []
        for idx, t in enumerate((fr - 2, fr, fr + 2)):
            dets_global = [
                (x1 + x0, y1 + y0, x2 + x0, y2 + y0, c)
                for x1, y1, x2, y2, c in _extract_boxes_from_result(results[idx])
            ]
            if is_near:
                min_c = 0.54 if idx == 1 else 0.40
            else:
                min_c = 0.48 if idx == 1 else 0.36
            box = _best_box_near_atd(
                dets_global, H_mats[t], float(bx_arr[t]), float(by_arr[t]),
                pitch_w, pitch_h, max_dist_m=max_dm, min_conf=min_c, min_margin=min_marg
            )
            picked.append(box)

        if any(b is None for b in picked):
            stats["rejected"] += 1
            return

        centers = [(0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])) for b in picked]  # type: ignore[index]
        mx = 0.5 * (centers[0][0] + centers[2][0])
        my = 0.5 * (centers[0][1] + centers[2][1])
        err_px = ((centers[1][0] - mx) ** 2 + (centers[1][1] - my) ** 2) ** 0.5
        if err_px > max_err_px:
            stats["rejected"] += 1
            return

        bx1, by1, bx2, by2, bconf = picked[1]  # type: ignore[misc]
        tx1, ty1, tx2, ty2 = bx1 - x0, by1 - y0, bx2 - x0, by2 - y0
        if not (6.0 <= tx1 and ty1 >= 6.0 and tx2 <= TILE_SIZE - 6.0 and ty2 <= TILE_SIZE - 6.0):
            stats["rejected"] += 1
            return

        w_px, h_px = tx2 - tx1, ty2 - ty1
        cx_n = (0.5 * (tx1 + tx2)) / TILE_SIZE
        cy_n = (0.5 * (ty1 + ty2)) / TILE_SIZE
        wn = max(4.0, w_px) / TILE_SIZE
        hn = max(4.0, h_px) / TILE_SIZE

        fname = f"bar_{regime}_f{fr:06d}"
        out_splits = ("train",) if split == "train" else ("val_bar", "val")
        for ds in out_splits:
            cv2.imwrite(str(STAGE_B_DIR / "images" / ds / f"{fname}.jpg"), tiles[1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (STAGE_B_DIR / "labels" / ds / f"{fname}.txt").write_text(
                f"0 {cx_n:.6f} {cy_n:.6f} {wn:.6f} {hn:.6f}\n", encoding="utf-8"
            )
        stats[f"{split}_{regime}"] += 1

        if split == "train" and saved_previews[regime] < 5:
            prev = tiles[1].copy()
            cv2.rectangle(prev, (int(round(tx1)) - 2, int(round(ty1)) - 2), (int(round(tx2)) + 2, int(round(ty2)) + 2), (0, 255, 0), 2)
            cv2.putText(prev, f"{regime} f={fr} c={bconf:.2f} e={err_px:.1f}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(preview_dir / f"opsi_b_{regime}_{saved_previews[regime]:02d}_f{fr:06d}.jpg"), prev)
            saved_previews[regime] += 1

        # Player-centered & pitch-interior negative + hard-FP mining (>= 18m from live ball)
        if rng.random() < 0.45:
            far_players = []
            bx_f, by_f = float(bx_arr[fr]), float(by_arr[fr])
            for p_idx in range(px_arr.shape[1]):
                px_v, py_v = float(px_arr[fr, p_idx]), float(py_arr[fr, p_idx])
                if np.isfinite(px_v) and np.isfinite(py_v) and 0.18 <= px_v <= 0.82 and 0.20 <= py_v <= 0.80:
                    d_m = (((px_v - bx_f) * pitch_w) ** 2 + ((py_v - by_f) * pitch_h) ** 2) ** 0.5
                    if d_m >= 18.0:
                        far_players.append((px_v, py_v))
            if far_players and rng.random() < 0.80:
                nx_norm, ny_norm = far_players[int(rng.integers(0, len(far_players)))]
                ncx, ncy = load_homography.project(H_mats[fr], nx_norm, ny_norm)
                ncx += float(rng.integers(-110, 111))
                ncy += float(rng.integers(-110, 111))
            else:
                nx_norm = float(rng.uniform(0.20, 0.80))
                ny_norm = float(rng.uniform(0.24, 0.78))
                ncx, ncy = load_homography.project(H_mats[fr], nx_norm, ny_norm)

            neg_x0 = int(np.clip(round(ncx - TILE_SIZE * 0.5), 0, TACTICAL_W - TILE_SIZE))
            neg_y0 = int(np.clip(round(ncy - TILE_SIZE * 0.5), 0, TACTICAL_H - TILE_SIZE))
            bx_c, by_c = centers[1]
            if not (neg_x0 - 24 <= bx_c <= neg_x0 + TILE_SIZE + 24 and neg_y0 - 24 <= by_c <= neg_y0 + TILE_SIZE + 24):
                neg_tile = imgs[1][neg_y0 : neg_y0 + TILE_SIZE, neg_x0 : neg_x0 + TILE_SIZE]
                neg_res = model_a.predict(neg_tile, imgsz=TILE_SIZE, conf=0.18, device=0, verbose=False)[0]
                fp_boxes = _extract_boxes_from_result(neg_res)
                if len(fp_boxes) == 0:
                    if rng.random() < 0.40:
                        nname = f"bar_neg_f{fr:06d}"
                        for ds in out_splits:
                            cv2.imwrite(str(STAGE_B_DIR / "images" / ds / f"{nname}.jpg"), neg_tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            (STAGE_B_DIR / "labels" / ds / f"{nname}.txt").write_text("", encoding="utf-8")
                        stats[f"{split}_neg"] += 1
                else:
                    all_interior = True
                    for x1, y1, x2, y2, _ in fp_boxes:
                        ux, uy = load_homography.unproject(H_mats[fr], neg_x0 + 0.5 * (x1 + x2), neg_y0 + 0.5 * (y1 + y2))
                        if not (0.06 <= float(ux) <= 0.94 and 0.06 <= float(uy) <= 0.94):
                            all_interior = False
                            break
                    if all_interior:
                        nname = f"bar_fp_f{fr:06d}"
                        for ds in out_splits:
                            cv2.imwrite(str(STAGE_B_DIR / "images" / ds / f"{nname}.jpg"), neg_tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            (STAGE_B_DIR / "labels" / ds / f"{nname}.txt").write_text("", encoding="utf-8")
                        stats[f"{split}_fp"] += 1

    task_by_end = {fr + 2: (fr, split) for fr, split in all_tasks}
    cur_pos = -1000
    for idx_t, target_f in enumerate(needed_frames):
        if target_f - cur_pos > 75 or target_f < cur_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            cur_pos = target_f - 1
        while cur_pos < target_f - 1:
            if not cap.grab():
                break
            cur_pos += 1
        ok, frame = cap.read()
        cur_pos += 1
        if not ok:
            continue
        if target_f in needed_set:
            frame_buf[target_f] = frame
        if target_f in task_by_end:
            fr, split = task_by_end[target_f]
            if all(t in frame_buf for t in (fr - 2, fr, fr + 2)):
                process_triplet(fr, split)
        stale = [k for k in frame_buf if k < target_f - 6]
        for k in stale:
            frame_buf.pop(k, None)
        if (idx_t + 1) % 600 == 0 or idx_t + 1 == len(needed_frames):
            print(
                f"  [Harvest {idx_t + 1}/{len(needed_frames)}] "
                f"tr(open={stats['train_open']},near={stats['train_near']},-={stats['train_neg']},fp={stats['train_fp']}) "
                f"val(open={stats['val_bar_open']},near={stats['val_bar_near']},-={stats['val_bar_neg']},fp={stats['val_bar_fp']}) "
                f"rej={stats['rejected']}"
            )
    cap.release()

    _write_yaml(STAGE_B_YAML, STAGE_B_DIR, "images/train", "images/val", "images/test")
    _write_yaml(STAGE_B_DIR / "data_val_bar.yaml", STAGE_B_DIR, "images/train", "images/val_bar", "images/test")
    _write_yaml(STAGE_B_DIR / "data_val_rf.yaml", STAGE_B_DIR, "images/train", "images/val_rf", "images/test")
    return stats, {"train_candidates": len(train_anchors), "val_candidates": len(val_anchors)}


def run_stage_b() -> Path:
    best_a = RUNS_DIR / "stage_a_yolo26s" / "weights" / "best.pt"
    if not best_a.exists():
        raise FileNotFoundError(f"Stage-A weights not found at {best_a}. Run `--stage A` first.")

    print("=== Sprint 2 Stage-B: Harvesting Opsi-B self-supervised BAR-ATH tiles ===")
    harvest_stats, cand_info = _harvest_bar_opsi_b(best_a)
    print(f"[Harvest DONE] {harvest_stats}")

    print("=== Sprint 2 Stage-B: Fine-tuning YOLO26s on RF + BAR-ATH Opsi-B ===")
    best_b = _train_yolo(best_a, STAGE_B_YAML, "stage_b_yolo26s", epochs=16, lr0=0.002, patience=8, close_mosaic=4)
    m_val_bar = _eval_split(best_b, STAGE_B_DIR / "data_val_bar.yaml", "val", "StageB_BAR")
    m_val_rf = _eval_split(best_b, STAGE_B_DIR / "data_val_rf.yaml", "val", "StageB_RF")
    m_test_rf = _eval_split(best_b, STAGE_B_YAML, "test", "StageB_RF")

    # Compare Stage-A vs Stage-B directly on held-out val_bar too!
    m_a_on_bar = _eval_split(best_a, STAGE_B_DIR / "data_val_bar.yaml", "val", "StageA_on_BAR")

    model_b = YOLO(str(best_b))
    probe_b = _save_probe_visuals(model_b, RUNS_DIR / "stage_b_probe", conf=0.20)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    exported = MODELS_DIR / "ball_yolo26s_stage_b.pt"
    shutil.copy2(best_b, exported)
    exported_a = MODELS_DIR / "ball_yolo26s_stage_a.pt"
    shutil.copy2(best_a, exported_a)

    summary = {
        "candidates": cand_info,
        "harvest": harvest_stats,
        "stage_a_on_val_bar": m_a_on_bar,
        "stage_b_val_bar": m_val_bar,
        "stage_b_val_rf": m_val_rf,
        "stage_b_test_rf": m_test_rf,
        "stage_b_probe": probe_b,
        "exported_weights": str(exported),
    }
    (RUNS_DIR / "stage_b_yolo26s" / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Stage-B DONE] Exported clean weights to {exported}")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26s Stage-A and Stage-B tiny ball detector.")
    parser.add_argument("--stage", choices=("A", "B"), required=True, help="Which sprint training stage to run.")
    args = parser.parse_args()
    print(f"CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if args.stage == "A":
        run_stage_a()
    else:
        run_stage_b()


if __name__ == "__main__":
    main()
