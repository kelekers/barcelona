"""Infer possession for the competition match using the Hybrid pipeline:

  1. DFL-trained external models (ball-blind dead/alive head + symmetric team head)
  2. Physical ball-state cues (oob, live-ball rescue, touch velocity modulation)
  3. Smart Tagging soft priors & restart anchors (GOALS, SET PIECES, offensive phases)
  4. Touch-modulated Forward-Backward + FFBS Minimum Bayes Risk (MBR) decoding

Usage:
    python src/run_pipeline.py --out evaluation/results/submission.csv
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME  # noqa: E402
from src import model as frame_model  # noqa: E402
from src.adapt import adapt  # noqa: E402
from src.atd_io import load_atd  # noqa: E402
from src.common import A, D, H, LABEL_CHARS, MODEL_DIR, X  # noqa: E402
from src.cv_integration import apply_boundary_motion_cv, apply_pre_decode_cv, ensure_pre_decode_cv  # noqa: E402
from src.decoder import (  # noqa: E402
    DEFAULT_PENALTIES, HYBRID_PENALTIES, HYBRID_V2_ADVANCE_FRAMES, HYBRID_V2_PENALTIES,
    decode, decode_hybrid,
)
from src.smart_priors import compute_physical_cues, compute_smart_tagging_priors  # noqa: E402
from src.stage2 import load_stage2, predict_stage2, stage2_available  # noqa: E402
from src.tagging import tag_features  # noqa: E402

# FAQ.md: the second half is graded from frame 110,808 but does not kick off until 113,742, and
# every graded frame before that kick-off is dead ball.
SECOND_HALF_KICKOFF = {"BAR-ATH": (110808, 113741)}
ADAPT_ROUNDS = 1


def write_submission(path, codes):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    letters = np.array(LABEL_CHARS)[codes]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(letters))


def summary(codes, half_blocks, smart=None):
    graded = np.zeros(len(codes), bool)
    for _, first, last in half_blocks:
        graded[first:last + 1] = True
    g = codes[graded]
    change = np.flatnonzero(np.diff(codes) != 0) + 1
    bounds = np.concatenate(([0], change, [len(codes)]))
    spells = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if codes[a] in (H, A)]
    lengths = np.array([b - a for a, b in spells]) / 25.0
    print("graded frames {}:  H {:.1f}%  A {:.1f}%  D {:.1f}%".format(
        len(g), 100 * (g == H).mean(), 100 * (g == A).mean(), 100 * (g == D).mean()))
    print("possessions {}  mean {:.1f}s  median {:.1f}s  home share of live time {:.1f}%".format(
        len(spells), lengths.mean(), np.median(lengths),
        100.0 * (g == H).sum() / max(1, ((g == H) | (g == A)).sum())))

    if smart is not None:
        print("\nGoal celebration dead-ball alignment:")
        for i, (stop_start, t_ko, _) in enumerate(smart["goal_windows"]):
            d_frac = float(np.mean(codes[stop_start:t_ko] == D))
            post = int(codes[min(len(codes) - 1, t_ko + 10)])
            print("  Goal {} [{:>6}..{:>6}): D={:5.1f}%  post-KO state={}".format(
                i + 1, stop_start, t_ko, 100.0 * d_frac, ("H", "A", "D", "X")[post]))
        sp = smart["sp_restarts"]
        n_tm = sum(1 for _, _, tm, _ in sp if tm is not None)
        d_bef = sum(1 for _, tr, _, _ in sp if np.any(codes[max(0, tr - 50):tr + 5] == D))
        tm_match = sum(1 for _, tr, tm, _ in sp
                       if tm is not None and np.any(codes[tr:min(len(codes), tr + 50)] == tm))
        print("Set Pieces alignment ({} non-KO, {} with team): pre-D={}/{}, team_match={}/{}".format(
            len(sp), n_tm, d_bef, len(sp), tm_match, n_tm))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "submission.csv"))
    parser.add_argument("--save-prob", default=None, help="optional .npy dump of the frame probabilities")
    parser.add_argument("--load-prob", default=None,
                        help="reuse external-model probabilities saved earlier")
    parser.add_argument("--save-teacher", default=None, help="optional .npy dump of the external model's probabilities")
    parser.add_argument("--legacy-crossdomain", action="store_true",
                        help="run original constant-Viterbi crossdomain without physical/MBR/Smart Tagging")
    parser.add_argument("--no-smart-tags", action="store_true",
                        help="disable Smart Tagging anchors in hybrid mode")
    parser.add_argument("--no-stage2", action="store_true",
                        help="disable Stage-2 Temporal Stacker + Boundary + MLP ensemble (use v1)")
    parser.add_argument("--no-mbr", action="store_true",
                        help="use touch-modulated Viterbi instead of MBR in hybrid mode")
    parser.add_argument("--no-cv", action="store_true",
                        help="disable Tahap-2 CV enhancements (run Tahap-1 Hybrid v2 on raw ATD)")
    parser.add_argument("--rebuild-cv", action="store_true",
                        help="recompute Sprint 3/4/5 CV artifacts from raw tactical video and YOLO26s weights")
    parser.add_argument("--adapt", action="store_true",
                        help="EXPERIMENTAL: add label-free co-training adaptation")
    parser.add_argument("--no-tags", action="store_true",
                        help="with --adapt: leave Smart Tagging side features out")
    parser.add_argument("--rounds", type=int, default=ADAPT_ROUNDS, help="with --adapt: rounds")
    args = parser.parse_args()

    started = time.time()
    print("loading tracking ...")
    match = load_atd(game=args.game, data_dir=args.data_dir)
    use_cv = (not args.no_cv) and (not args.legacy_crossdomain) and (args.game == GAME)
    if use_cv:
        ensure_pre_decode_cv(rebuild=args.rebuild_cv)
        match = apply_pre_decode_cv(match)

    if args.load_prob:
        prob = np.load(args.load_prob).astype(np.float64)
    else:
        bundle, meta = frame_model.load(args.model_dir)
        print("external DFL model ({} features, both team orientations) ...".format(len(meta["feature_names"])))
        prob = frame_model.predict_proba(bundle, match, meta["feature_names"])
    if args.save_teacher:
        np.save(args.save_teacher, prob.astype(np.float64))

    forced = [SECOND_HALF_KICKOFF[args.game]] if args.game in SECOND_HALF_KICKOFF else []
    if args.adapt:
        extra = None
        if not args.no_tags:
            forward, mirrored, tag_names = tag_features(args.game, args.data_dir)
            extra = (forward, mirrored)
            print("Smart Tagging side features for the student: {}".format(", ".join(tag_names)))
        print("label-free adaptation to this match (cross-fitted co-training, {} round(s)) ...".format(
            args.rounds))
        prob = adapt(match, prob, DEFAULT_PENALTIES, forced_dead=forced, extra=extra,
                     rounds=args.rounds, verbose=True)
    if args.save_prob:
        np.save(args.save_prob, prob.astype(np.float32))

    smart = None
    if args.legacy_crossdomain:
        codes = decode(prob, match["half_blocks"], DEFAULT_PENALTIES, forced_dead=forced)
    else:
        print("building physical ball cues & Smart Tagging priors ...")
        cues = compute_physical_cues(match)
        pbha = pbdl = None
        if (not args.no_stage2) and stage2_available(args.model_dir):
            s2_bundle, s2_meta = load_stage2(args.model_dir)
            print("Stage-2 temporal stacker + Boundary + MLP ensemble (w_mlp={:.2f}) ...".format(
                s2_bundle["w_mlp"]))
            prob, pbha, pbdl = predict_stage2(s2_bundle, match, prob, cues)
        if not args.no_smart_tags:
            smart = compute_smart_tagging_priors(match, cues, data_dir=args.data_dir)
        penalties = HYBRID_V2_PENALTIES if pbha is not None else HYBRID_PENALTIES
        adv = HYBRID_V2_ADVANCE_FRAMES if pbha is not None else 2
        print("running hybrid {} decoder (pen={}) ...".format(
            "Viterbi" if args.no_mbr else "Forward-Backward MBR", penalties))
        codes = decode_hybrid(
            prob, match["half_blocks"], cues, smart=smart, forced_dead=forced,
            penalties=penalties, pbha=pbha, pbdl=pbdl,
            use_mbr=(not args.no_mbr), advance_frames=adv,
        )

    write_submission(args.out, codes)
    if use_cv:
        codes = apply_boundary_motion_cv(codes, args.out, rebuild=args.rebuild_cv)
        write_submission(args.out, codes)
    print("wrote {}  ({:.1f}s)".format(args.out, time.time() - started))
    summary(codes, match["half_blocks"], smart=smart)


if __name__ == "__main__":
    main()
