"""Sprint 1: Clean football_tactical dataset (unify poly+bbox, unstretch 16:9, multi-scale 640x640 tiles)
and prepare unlabeled BAR-ATH visual probes & Stage-B Opsi B helpers.

All generated image/label caches stay strictly under data/cache/ (gitignored per RULES.md).

Usage (inside .venv with OpenCV):
    python src/cv_ball_data.py
"""

import glob
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import asset_paths, half_video_blocks, video_path  # noqa: E402
from src.atd_io import load_atd  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402

ROBOFLOW_DIR = os.path.join(ROOT, "data", "external", "football_tactical")
STAGE_A_DIR = os.path.join(ROOT, "data", "cache", "cv_ball_stage_a")
BAR_PROBE_DIR = os.path.join(ROOT, "data", "cache", "cv_ball_bar_probe")

UNSTRETCH_W = 1138   # round(640 * 16 / 9) restores 1:1 round geometry from 640x640 stretched input
UNSTRETCH_H = 640
TILE_SIZE = 640
MAX_RAW_BOX_PX = 32.0


def parse_ball_boxes(label_path):
    """Parse class 0 ('ball') annotations supporting both YOLO bbox (5 cols) and polygon (>5 cols).

    Returns (clean_boxes_norm, had_outlier) where each box is (cx, cy, bw, bh) in [0, 1].
    """
    boxes = []
    had_outlier = False
    with open(label_path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts or int(parts[0]) != 0:
                continue
            if len(parts) == 5:
                cx, cy, bw, bh = map(float, parts[1:5])
            elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                coords = [float(v) for v in parts[1:]]
                xs, ys = coords[0::2], coords[1::2]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                cx, cy, bw, bh = 0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0
            else:
                continue
            if bw <= 0.0 or bh <= 0.0:
                continue
            if max(bw * 640.0, bh * 640.0) > MAX_RAW_BOX_PX:
                had_outlier = True
                continue
            boxes.append((cx, cy, bw, bh))
    return boxes, had_outlier


def _choose_crop(ux, uy, uw, uh, crop_size, rng, random_jitter=True):
    """Choose (ox, oy) crop origin inside 1138x640 covering (ux, uy) safely."""
    max_ox = UNSTRETCH_W - crop_size
    max_oy = UNSTRETCH_H - crop_size
    margin = 24.0
    ox_lo = max(0, int(np.ceil(ux + uw * 0.5 + margin - crop_size)))
    ox_hi = min(max_ox, int(np.floor(ux - uw * 0.5 - margin)))
    if ox_hi < ox_lo:
        ox = int(np.clip(round(ux - crop_size * 0.5), 0, max_ox))
    elif random_jitter:
        ox = rng.randint(ox_lo, ox_hi)
    else:
        ox = (ox_lo + ox_hi) // 2

    if max_oy <= 0:
        oy = 0
    else:
        oy_lo = max(0, int(np.ceil(uy + uh * 0.5 + margin - crop_size)))
        oy_hi = min(max_oy, int(np.floor(uy - uh * 0.5 - margin)))
        if oy_hi < oy_lo:
            oy = int(np.clip(round(uy - crop_size * 0.5), 0, max_oy))
        elif random_jitter:
            oy = rng.randint(oy_lo, oy_hi)
        else:
            oy = (oy_lo + oy_hi) // 2
    return ox, oy


def _labels_in_crop(px_boxes, ox, oy, crop_size):
    """Convert unstretched pixel boxes into normalized YOLO lines inside crop (ox, oy, crop_size)."""
    out = []
    for ux, uy, uw, uh in px_boxes:
        lx, ly = ux - ox, uy - oy
        if (2.0 < lx < crop_size - 2.0) and (2.0 < ly < crop_size - 2.0):
            x0 = max(0.0, lx - 0.5 * uw)
            x1 = min(float(crop_size), lx + 0.5 * uw)
            y0 = max(0.0, ly - 0.5 * uh)
            y1 = min(float(crop_size), ly + 0.5 * uh)
            if x1 - x0 >= 1.5 and y1 - y0 >= 1.5:
                cx = 0.5 * (x0 + x1) / crop_size
                cy = 0.5 * (y0 + y1) / crop_size
                bw = (x1 - x0) / crop_size
                bh = (y1 - y0) / crop_size
                out.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def prepare_stage_a_dataset(seed=42, zoom_frac=0.35, neg_frac=0.15):
    """Clean & unstretch football_tactical into 1:1 640x640 Stage-A YOLO dataset."""
    rng = random.Random(seed)
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for dst_sp in split_map.values():
        os.makedirs(os.path.join(STAGE_A_DIR, "images", dst_sp), exist_ok=True)
        os.makedirs(os.path.join(STAGE_A_DIR, "labels", dst_sp), exist_ok=True)
    prev_dir = os.path.join(STAGE_A_DIR, "preview")
    os.makedirs(prev_dir, exist_ok=True)

    stats = {}
    preview_cnt = 0

    for src_sp, dst_sp in split_map.items():
        lbl_paths = sorted(glob.glob(os.path.join(ROBOFLOW_DIR, src_sp, "labels", "*.txt")))
        n_pos = n_zoom = n_neg = n_skip_outlier = n_skip_noball = 0
        ws, hs = [], []

        for lp in lbl_paths:
            boxes, had_outlier = parse_ball_boxes(lp)
            if had_outlier:
                n_skip_outlier += 1
                continue
            if not boxes:
                n_skip_noball += 1
                continue

            stem = os.path.splitext(os.path.basename(lp))[0]
            img_p = os.path.join(ROBOFLOW_DIR, src_sp, "images", stem + ".jpg")
            if not os.path.exists(img_p):
                continue
            img = cv2.imread(img_p)
            if img is None:
                continue

            img_u = cv2.resize(img, (UNSTRETCH_W, UNSTRETCH_H), interpolation=cv2.INTER_LINEAR)
            px_boxes = [(cx * UNSTRETCH_W, cy * UNSTRETCH_H, bw * UNSTRETCH_W, bh * UNSTRETCH_H)
                        for cx, cy, bw, bh in boxes]
            ux0, uy0, uw0, uh0 = px_boxes[0]

            # 1. Primary 640x640 unstretched tile covering the ball
            is_train = (src_sp == "train")
            crop_sz = 480 if (is_train and rng.random() < zoom_frac) else TILE_SIZE
            ox, oy = _choose_crop(ux0, uy0, uw0, uh0, crop_sz, rng, random_jitter=is_train)
            tile = img_u[oy:oy + crop_sz, ox:ox + crop_sz]
            if crop_sz != TILE_SIZE:
                tile = cv2.resize(tile, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
                n_zoom += 1
            lbl_lines = _labels_in_crop(px_boxes, ox, oy, crop_sz)
            if not lbl_lines:
                continue

            for ln in lbl_lines:
                ps = ln.split()
                ws.append(float(ps[3]) * TILE_SIZE)
                hs.append(float(ps[4]) * TILE_SIZE)

            out_img = os.path.join(STAGE_A_DIR, "images", dst_sp, f"{stem}.jpg")
            out_lbl = os.path.join(STAGE_A_DIR, "labels", dst_sp, f"{stem}.txt")
            cv2.imwrite(out_img, tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out_lbl, "w") as out_f:
                out_f.write("\n".join(lbl_lines) + "\n")
            n_pos += 1

            if preview_cnt < 12 and (dst_sp == "val" or crop_sz == 480):
                vis = tile.copy()
                for ln in lbl_lines:
                    _, cx, cy, bw, bh = map(float, ln.split())
                    x0 = int(round((cx - 0.5 * bw) * TILE_SIZE)) - 3
                    y0 = int(round((cy - 0.5 * bh) * TILE_SIZE)) - 3
                    x1 = int(round((cx + 0.5 * bw) * TILE_SIZE)) + 3
                    y1 = int(round((cy + 0.5 * bh) * TILE_SIZE)) + 3
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
                cv2.putText(vis, f"{dst_sp} c={crop_sz}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(prev_dir, f"stage_a_{preview_cnt:02d}_{dst_sp}.jpg"), vis)
                preview_cnt += 1

            # 2. Verified ball-free opposite-half negative tile (no risk of unannotated ball!)
            if len(px_boxes) == 1 and rng.random() < neg_frac:
                if ux0 < 390.0:
                    nox = UNSTRETCH_W - TILE_SIZE
                elif ux0 > 748.0:
                    nox = 0
                else:
                    nox = None
                if nox is not None:
                    neg_tile = img_u[0:TILE_SIZE, nox:nox + TILE_SIZE]
                    cv2.imwrite(os.path.join(STAGE_A_DIR, "images", dst_sp, f"{stem}_neg.jpg"),
                                neg_tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(os.path.join(STAGE_A_DIR, "labels", dst_sp, f"{stem}_neg.txt"), "w") as out_f:
                        out_f.write("")
                    n_neg += 1

        stats[dst_sp] = {
            "pos": n_pos, "zoom480": n_zoom, "neg": n_neg,
            "skip_outlier": n_skip_outlier, "skip_noball": n_skip_noball,
            "med_w_px": float(np.median(ws)) if ws else 0.0,
            "med_h_px": float(np.median(hs)) if hs else 0.0,
        }
        print(
            f"Stage-A {dst_sp:<5}: pos={n_pos} (zoom480={n_zoom}), neg={n_neg} | "
            f"skipped_outlier={n_skip_outlier}, skipped_noball={n_skip_noball} | "
            f"med_ball={stats[dst_sp]['med_w_px']:.1f}x{stats[dst_sp]['med_h_px']:.1f} px"
        )

    yaml_path = os.path.join(STAGE_A_DIR, "data.yaml")
    with open(yaml_path, "w", newline="\n") as handle:
        handle.write(
            f"path: {STAGE_A_DIR.replace(os.sep, '/')}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names: ['ball']\n"
        )
    print(f"Wrote Stage-A YAML -> {yaml_path}")
    return stats


def prepare_bar_probe_frames():
    """Save 16 unlabeled 1080p BAR-ATH frames (outside public 5-min GT) for Stage-A/B visual checks."""
    os.makedirs(BAR_PROBE_DIR, exist_ok=True)
    # Include frames we audited earlier (20434, 28387, 38004, 45306, 57585, 68771) + diverse spans
    probe_frames = [
        20434, 28387, 38004, 45306, 57585, 68771,
        24500, 33000, 52000, 78000, 86000,
        118000, 130000, 145000, 165000, 180000,
    ]
    cap = cv2.VideoCapture(video_path())
    saved = 0
    for fr in probe_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, img = cap.read()
        if ok:
            cv2.imwrite(os.path.join(BAR_PROBE_DIR, f"frame_{fr:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
    cap.release()
    print(f"Saved {saved} 1080p BAR-ATH probe frames -> {BAR_PROBE_DIR}")


if __name__ == "__main__":
    prepare_stage_a_dataset()
    prepare_bar_probe_frames()
