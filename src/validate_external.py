"""Score the method on labelled games it was not trained on -- never on the competition match.

Two held-out sets:

  * the three Metrica sample games: a different provider and a different labelling process from
    the DFL matches the model learned from, and the same family the competition's ground truth
    comes from. This is the main number.
  * any DFL match left out of training (--dfl-holdout), as an in-provider check.

Each game is scored as provided (raw) and after the two corruption settings of degrade.py, using the
official scorer (evaluation/possession_metrics.py). The decoder's three penalties are chosen
here, on the average over these games, and frozen in decoder.DEFAULT_PENALTIES.

    python src/train.py --holdout J03WR9 --out models_dev
    python src/validate_external.py --model-dir models_dev --dfl-holdout J03WR9 --grid
"""

import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from src import model as frame_model  # noqa: E402
from src.common import EXTERNAL_DIR, MODEL_DIR, X  # noqa: E402
from src.decoder import DEFAULT_PENALTIES, HYBRID_PENALTIES, decode, decode_hybrid  # noqa: E402
from src.degrade import degrade  # noqa: E402
from src.dfl_io import load_dfl_match  # noqa: E402
from src.metrica_io import load_metrica_game  # noqa: E402
from src.smart_priors import compute_physical_cues  # noqa: E402

SETTINGS = ("raw", "light", "heavy")
GRID = {"switch": (1.0, 2.0, 3.0, 4.5, 6.0, 9.0), "stop": (2.0, 4.0, 8.0),
        "restart": (2.0, 4.0, 8.0)}


def truth_codes(match):
    codes = np.full(match["n_frames"], X, np.int8)
    for _, first, last in match["half_blocks"]:
        codes[first:last + 1] = match["labels"][first:last + 1]
    return codes


def score(truth, prob, half_blocks, penalties):
    pred = decode(prob, half_blocks, penalties)
    result = pm.evaluate(truth, pred)
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]


def cached_probabilities(tag, loader, setting, booster, names, cache_dir, seed):
    path = os.path.join(cache_dir, "{}_{}.npz".format(tag, setting))
    if os.path.exists(path):
        z = np.load(path)
        return z["prob"], z["truth"], [tuple(int(v) for v in r) for r in z["half_blocks"]]
    match = degrade(loader(), setting, seed=seed)
    prob = frame_model.predict_proba(booster, match, names).astype(np.float32)
    truth = truth_codes(match)
    np.savez_compressed(path, prob=prob, truth=truth, half_blocks=np.array(match["half_blocks"]))
    return prob, truth, match["half_blocks"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--dfl-holdout", nargs="*", default=[])
    parser.add_argument("--grid", action="store_true", help="search the decoder penalties")
    parser.add_argument("--tag", default="dev", help="cache namespace")
    parser.add_argument("--adapt", nargs="*", default=None, metavar="SETTING",
                        help="also run the label-free adaptation (adapt.py) on these settings "
                             "(no value = all) and report before/after")
    parser.add_argument("--adapt-rounds", type=int, default=1)
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS,
                        help="corruption settings to score (default: all)")
    parser.add_argument("--hybrid", action="store_true",
                        help="evaluate hybrid decoder (physical cues + touch modulation + MBR)")
    parser.add_argument("--no-mbr", action="store_true",
                        help="with --hybrid: use touch-modulated Viterbi instead of MBR")
    args = parser.parse_args()

    booster, meta = frame_model.load(args.model_dir)
    names = meta["feature_names"]
    cache_dir = os.path.join(EXTERNAL_DIR, "prob_cache_" + args.tag)
    os.makedirs(cache_dir, exist_ok=True)

    games = [("metrica{}".format(g), (lambda g=g: load_metrica_game(g))) for g in (1, 2, 3)]
    games += [("dfl" + m, (lambda m=m: load_dfl_match(m))) for m in args.dfl_holdout]
    entries = []
    for gi, (tag, loader) in enumerate(games):
        for si, setting in enumerate(SETTINGS):
            if setting not in args.settings:
                continue
            prob, truth, blocks = cached_probabilities(tag, loader, setting, booster, names,
                                                       cache_dir, seed=5000 + 10 * gi + si)
            entries.append((tag, setting, prob, truth, blocks))
            graded = truth != X
            acc = float((prob[graded].argmax(1) == truth[graded]).mean())
            print("{:<10} {:<6} frame accuracy before decoding {:.3f}".format(tag, setting, acc))
            sys.stdout.flush()

    def table(penalties):
        rows = [(tag, setting) + score(truth, prob, blocks, penalties)
                for tag, setting, prob, truth, blocks in entries]
        return rows

    def mean_of(rows, prefix):
        vals = [r[2] for r in rows if r[0].startswith(prefix)]
        return float(np.mean(vals)) if vals else float("nan")

    if args.grid:
        results = []
        for switch, stop, restart in itertools.product(GRID["switch"], GRID["stop"], GRID["restart"]):
            pen = {"switch": switch, "stop": stop, "restart": restart}
            rows = table(pen)
            results.append((mean_of(rows, "metrica"), mean_of(rows, "dfl"), pen))
        results.sort(key=lambda r: -r[0])
        print("\npenalty grid, ranked by mean spellF1 over the Metrica games (all settings):")
        for metrica, dfl, pen in results[:12]:
            print("  metrica {:.4f}  dfl {:.4f}  {}".format(metrica, dfl, pen))
        print("  ...\n  worst: metrica {:.4f}  {}".format(results[-1][0], results[-1][2]))
        penalties = results[0][2]
    else:
        penalties = DEFAULT_PENALTIES

    print("\nper game with penalties {}:".format(penalties))
    print("  {:<10} {:<6} {:>8} {:>8} {:>6} {:>6}".format("game", "noise", "spellF1", "macroF1", "pred", "true"))
    rows = table(penalties)
    for tag, setting, spell, macro, n_pred, n_true in rows:
        print("  {:<10} {:<6} {:>8.4f} {:>8.4f} {:>6} {:>6}".format(tag, setting, spell, macro, n_pred, n_true))
    for setting in args.settings:
        sub = [r for r in rows if r[1] == setting and r[0].startswith("metrica")]
        print("  Metrica mean, {:<6}: spellF1 {:.4f}  macroF1 {:.4f}".format(
            setting, np.mean([r[2] for r in sub]), np.mean([r[3] for r in sub])))

    if args.hybrid:
        print("\nHYBRID decoder (penalties={}, use_mbr={}):".format(HYBRID_PENALTIES, not args.no_mbr))
        print("  {:<10} {:<6} {:>8} {:>8} {:>6} {:>6}".format("game", "noise", "spellF1", "macroF1", "pred", "true"))
        hrows = []
        for gi, (tag, loader) in enumerate(games):
            for si, setting in enumerate(SETTINGS):
                if setting not in args.settings:
                    continue
                prob, truth, blocks = [e[2:] for e in entries if e[0] == tag and e[1] == setting][0]
                match = degrade(loader(), setting, seed=5000 + 10 * gi + si)
                cues = compute_physical_cues(match)
                pred = decode_hybrid(prob, blocks, cues, smart=None, penalties=HYBRID_PENALTIES,
                                     use_mbr=(not args.no_mbr))
                res = pm.evaluate(truth, pred)
                row = (tag, setting, res["spell_f1_mean"], res["macro_f1"],
                       res["n_pred_spells"], res["n_gt_spells"])
                hrows.append(row)
                print("  {:<10} {:<6} {:>8.4f} {:>8.4f} {:>6} {:>6}".format(*row))
        for setting in args.settings:
            sub = [r for r in hrows if r[1] == setting and r[0].startswith("metrica")]
            print("  Hybrid Metrica mean, {:<6}: spellF1 {:.4f}  macroF1 {:.4f}".format(
                setting, np.mean([r[2] for r in sub]), np.mean([r[3] for r in sub])))

    if args.adapt is not None:
        from src.adapt import adapt
        wanted = [x for x in (args.adapt or list(SETTINGS)) if x in args.settings]
        print()
        print("label-free adaptation, scored against labels it never saw (same penalties):")
        print("  {:<10} {:<6} {:>9} {:>9} {:>8}   {:>9} {:>9}".format(
            "game", "noise", "teacher", "adapted", "delta", "macro t.", "macro a."))
        deltas = []
        for gi, (tag, loader) in enumerate(games):
            for si, setting in enumerate(SETTINGS):
                if setting not in wanted:
                    continue
                prob, truth, blocks = [e[2:] for e in entries if e[0] == tag and e[1] == setting][0]
                path = os.path.join(cache_dir, "{}_{}_adapt{}.npz".format(tag, setting, args.adapt_rounds))
                if os.path.exists(path):
                    fused = np.load(path)["prob"]
                else:
                    match = degrade(loader(), setting, seed=5000 + 10 * gi + si)
                    fused = adapt(match, prob.astype(np.float64), penalties,
                                  rounds=args.adapt_rounds).astype(np.float32)
                    np.savez_compressed(path, prob=fused)
                before = score(truth, prob, blocks, penalties)
                after = score(truth, fused, blocks, penalties)
                deltas.append(after[0] - before[0])
                print("  {:<10} {:<6} {:>9.4f} {:>9.4f} {:>+8.4f}   {:>9.4f} {:>9.4f}".format(
                    tag, setting, before[0], after[0], after[0] - before[0], before[1], after[1]))
                sys.stdout.flush()
        print("  mean change in spellF1: {:+.4f}  (helped on {} of {})".format(
            float(np.mean(deltas)), sum(d > 0 for d in deltas), len(deltas)))
