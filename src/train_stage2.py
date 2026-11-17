"""Train Hybrid Stage-2 temporal context stackers, boundary heads, and MLP ensemble.

Uses Stage-1 DFL external model predictions (`models/dead_head.txt`, `models/team_head.txt`)
paired with the 3 Metrica sample games under ATD-calibrated `light` and `heavy` degradation
(`degrade.py`) — never touching any competition label.

Two invariants preserve cross-provider generalization:
  * `dead2` and `bound_dl` see ONLY ball-blind features (`logit(p_dead1)` + player speeds),
    never ball visibility or Metrica's `ball = NaN` dead-ball convention.
  * `bound_ha` is trained on live frames only and fed to Stage-2 out-of-fold across games.

Outputs in `models/`:
  bound_ha.txt, bound_dl.txt, dead2.txt, team2.txt, stage2_mlp.npz, stage2_meta.json

Usage:
    python src/train_stage2.py              # run 3-fold Nested LOGO CV, then fit final -> models/
    python src/train_stage2.py --no-cv      # fit final -> models/ directly (~90s)
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import possession_metrics as pm  # noqa: E402
from src import model as frame_model  # noqa: E402
from src.common import EXTERNAL_DIR, MODEL_DIR, X  # noqa: E402
from src.decoder import HYBRID_V2_ADVANCE_FRAMES, HYBRID_V2_PENALTIES, decode_hybrid  # noqa: E402
from src.degrade import degrade  # noqa: E402
from src.metrica_io import load_metrica_game  # noqa: E402
from src.smart_priors import compute_physical_cues  # noqa: E402
from src.stage2 import (  # noqa: E402
    build_bound_features, compute_targets, fit_stage2_models, predict_stage2, save_stage2,
)

SETTINGS = (("light", 1), ("heavy", 2))


def truth_codes(match):
    codes = np.full(match["n_frames"], X, np.int8)
    for _, first, last in match["half_blocks"]:
        codes[first:last + 1] = match["labels"][first:last + 1]
    return codes


def load_stage2_items(model_dir=MODEL_DIR, cache_dir=None, recompute=False):
    """Build Stage-1 probabilities + physical cues + boundary features for Metrica 1..3 (light, heavy)."""
    booster, meta = frame_model.load(model_dir)
    names = meta["feature_names"]
    if cache_dir is None:
        cache_dir = os.path.join(EXTERNAL_DIR, "prob_cache_dev")

    items = []
    for gi, g in enumerate((1, 2, 3)):
        raw_match = load_metrica_game(g)
        for setting, si in SETTINGS:
            seed = 5000 + 10 * gi + si
            match = degrade(raw_match, setting, seed=seed)
            cpath = os.path.join(cache_dir, f"metrica{g}_{setting}.npz")
            if (not recompute) and os.path.exists(cpath):
                z = np.load(cpath)
                prob = z["prob"].astype(np.float64)
                truth = z["truth"]
                blocks = [tuple(int(v) for v in r) for r in z["half_blocks"]]
            else:
                prob = frame_model.predict_proba(booster, match, names).astype(np.float64)
                truth = truth_codes(match)
                blocks = match["half_blocks"]
            cues = compute_physical_cues(match)
            xb = build_bound_features(match, prob, cues)
            y_ha, y_dl, y_band = compute_targets(truth, blocks)
            items.append((g, {
                "game_idx": g,
                "setting": setting,
                "match": match,
                "prob": prob,
                "truth": truth,
                "blocks": blocks,
                "cues": cues,
                "xb": xb,
                "y_ha": y_ha,
                "y_dl": y_dl,
                "y_band": y_band,
            }))
    return items


def run_nested_cv(items):
    """Strict 3-fold Nested Leave-One-Game-Out CV on Metrica (light + heavy)."""
    rows = []
    for val_g in (1, 2, 3):
        tr_nested = [(g, it) for g, it in items if g != val_g]
        val_items = [it for g, it in items if g == val_g]
        bundle = fit_stage2_models(None, nested_oof_items=tr_nested)
        for it in val_items:
            prob2, pbha, pbdl = predict_stage2(bundle, it["match"], it["prob"], it["cues"])
            pred = decode_hybrid(
                prob2, it["blocks"], it["cues"], smart=None,
                penalties=HYBRID_V2_PENALTIES, pbha=pbha, pbdl=pbdl,
                use_mbr=True, advance_frames=HYBRID_V2_ADVANCE_FRAMES,
            )
            res = pm.evaluate(it["truth"], pred)
            row = (
                f"metrica{val_g}", it["setting"],
                float(res["spell_f1_mean"]), float(res["macro_f1"]),
                int(res["n_pred_spells"]), int(res["n_gt_spells"]),
                float(res["transitions"][25]["f1"]),
                float(res["transitions"][25]["mean_latency"]),
            )
            rows.append(row)
            print("  val {:<9} {:<6} spellF1={:.4f} macroF1={:.4f} pred={}/{} tr25={:.4f} lat={:+.2f}".format(*row))
            sys.stdout.flush()

    cv_summary = {}
    for s in ("light", "heavy"):
        sub = [r for r in rows if r[1] == s]
        sp = float(np.mean([r[2] for r in sub]))
        ma = float(np.mean([r[3] for r in sub]))
        cv_summary[s] = {"spell_f1": sp, "macro_f1": ma}
        print("  Nested CV mean {:<6}: spellF1={:.4f} macroF1={:.4f}".format(s, sp, ma))
    return cv_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--no-cv", action="store_true", help="skip 3-fold Nested LOGO CV")
    parser.add_argument("--recompute", action="store_true", help="ignore prob_cache_dev and recompute Stage-1")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading Stage-1 probabilities + features for Metrica 1..3 (light, heavy) ...")
    items = load_stage2_items(model_dir=args.model_dir, recompute=args.recompute)
    print(f"Prepared {len(items)} items in {time.time() - t0:.1f}s")

    cv_summary = {}
    if not args.no_cv:
        print("\nRunning strict 3-fold Nested LOGO CV ...")
        cv_summary = run_nested_cv(items)

    print("\nTraining final Stage-2 + Boundary + MLP models (out-of-fold Bound_ha across 3 games) ...")
    t1 = time.time()
    final_bundle = fit_stage2_models(None, nested_oof_items=items)
    save_stage2(final_bundle, {
        "training_games": ["Metrica-1", "Metrica-2", "Metrica-3"],
        "settings": ["light", "heavy"],
        "penalties": HYBRID_V2_PENALTIES,
        "advance_frames": HYBRID_V2_ADVANCE_FRAMES,
        "nested_cv": cv_summary,
    }, model_dir=args.model_dir)
    print(f"Saved Stage-2 models -> {args.model_dir} ({time.time() - t1:.1f}s)")
