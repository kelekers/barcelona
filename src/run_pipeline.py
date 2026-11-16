"""Generate the submission from the provided tactical tracking.

    python src/run_pipeline.py --out evaluation/results/submission.csv

Deterministic: the same inputs and the pinned requirements reproduce the same file. The trained
model ships in models/ (regenerate it with src/train.py); no competition label is read anywhere.
The default path is inference only: features -> external model -> Viterbi decode.
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
from src.decoder import DEFAULT_PENALTIES, decode  # noqa: E402
from src.tagging import tag_features  # noqa: E402

# FAQ.md: the second half is graded from frame 110,808 but does not kick off until 113,742, and
# every graded frame before that kick-off is dead ball. A published fact about the recording,
# not a label derived from the ground truth.
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


def summary(codes, half_blocks):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "submission.csv"))
    parser.add_argument("--save-prob", default=None, help="optional .npy dump of the frame probabilities")
    parser.add_argument("--load-prob", default=None,
                        help="reuse external-model probabilities saved earlier with --save-teacher")
    parser.add_argument("--save-teacher", default=None, help="optional .npy dump of the external model's probabilities")
    parser.add_argument("--adapt", action="store_true",
                        help="EXPERIMENTAL, not part of the submission: add the label-free adaptation "
                             "step (adapt.py). It lowered spellF1 on 5 of 6 held-out external games.")
    parser.add_argument("--no-tags", action="store_true",
                        help="with --adapt: leave the Smart Tagging side features out")
    parser.add_argument("--rounds", type=int, default=ADAPT_ROUNDS, help="with --adapt: rounds")
    args = parser.parse_args()

    started = time.time()
    print("loading tracking ...")
    match = load_atd(game=args.game, data_dir=args.data_dir)
    if args.load_prob:
        prob = np.load(args.load_prob).astype(np.float64)
    else:
        bundle, meta = frame_model.load(args.model_dir)
        print("external model ({} features, both team orientations) ...".format(len(meta["feature_names"])))
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
    codes = decode(prob, match["half_blocks"], DEFAULT_PENALTIES, forced_dead=forced)
    write_submission(args.out, codes)
    print("wrote {}  ({:.0f}s)".format(args.out, time.time() - started))
    summary(codes, match["half_blocks"])


if __name__ == "__main__":
    main()
