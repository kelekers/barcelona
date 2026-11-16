"""End-to-end deterministic possession inference runner.

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
from starter.baseline import write_submission  # noqa: E402
from src.atd_io import load_match  # noqa: E402
from src.decoder import A, D, H, X, decode_match  # noqa: E402
from src.features import build_frame_evidence  # noqa: E402


def print_full_match_diagnostics(data, features, codes):
    """Print unlabeled full-match diagnostics (no GT accessed)."""
    graded = np.zeros(data["n_frames"], dtype=bool)
    for _, s, e in data["half_blocks"]:
        graded[s:e + 1] = True

    g_codes = codes[graded]
    total_g = len(g_codes)
    print("\nFull-match graded summary ({} frames):".format(total_g))
    for code, name in ((H, "H"), (A, "A"), (D, "D"), (X, "X")):
        cnt = int(np.sum(g_codes == code))
        print("  {}: {:>7} ({:5.1f}%)".format(name, cnt, 100.0 * cnt / total_g))

    alive = int(np.sum((g_codes == H) | (g_codes == A)))
    h_alive = int(np.sum(g_codes == H))
    if alive:
        print("  Home share of alive time: {:.1f}%".format(100.0 * h_alive / alive))

    # Check Goal celebrations
    print("\nGoal celebration dead-ball alignment:")
    for i, (stop_start, t_ko, conc_team) in enumerate(features["goal_dead_windows"]):
        d_frac = float(np.mean(codes[stop_start:t_ko] == D))
        post = int(codes[min(len(codes) - 1, t_ko + 10)])
        print("  Goal {} [{:>6}..{:>6}): D={:5.1f}%  post-KO state={}".format(
            i + 1, stop_start, t_ko, 100.0 * d_frac, ("H", "A", "D", "X")[post]))

    # Check SET PIECES restarts: how many have D before t_restart and target team after?
    sp = features["sp_restarts"]
    n_with_team = sum(1 for _, _, tm, _ in sp if tm is not None)
    d_before = sum(1 for _, tr, _, _ in sp if np.any(codes[max(0, tr - 50):tr + 5] == D))
    team_match = sum(
        1 for _, tr, tm, _ in sp
        if tm is not None and np.any(codes[tr:min(len(codes), tr + 50)] == tm)
    )
    print("\nSet Pieces alignment ({} non-KO detected, {} with team):".format(len(sp), n_with_team))
    print("  Had D around pre-restart [-50..+5]: {}/{}".format(d_before, len(sp)))
    print("  Entered restarting team within +2s : {}/{}".format(team_match, n_with_team))


def main():
    parser = argparse.ArgumentParser(description="Generate possession submission CSV.")
    parser.add_argument("--game", default=GAME, help="Game ID (default: %(default)s)")
    parser.add_argument("--data-dir", default=None, help="Optional override for data directory")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "submission.csv"),
                        help="Output CSV path")
    parser.add_argument("--p-ha", type=float, default=54.0, help="H<->A switch penalty")
    parser.add_argument("--p-ad", type=float, default=34.0, help="Alive->Dead penalty")
    parser.add_argument("--p-da", type=float, default=22.0, help="Dead->Alive penalty")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading match data ...")
    data = load_match(game=args.game, data_dir=args.data_dir)
    t1 = time.time()
    print("Loaded in {:.2f}s. Building physical & tactical features ...".format(t1 - t0))
    features = build_frame_evidence(data)
    t2 = time.time()
    print("Features built in {:.2f}s. Running 3-state Viterbi decoder ...".format(t2 - t1))
    labels, codes = decode_match(data, features, p_ha=args.p_ha, p_ad=args.p_ad, p_da=args.p_da)
    t3 = time.time()
    print("Decoded in {:.2f}s (total {:.2f}s).".format(t3 - t2, t3 - t0))

    write_submission(args.out, labels)
    print("Wrote submission -> {}".format(args.out))
    print_full_match_diagnostics(data, features, codes)


if __name__ == "__main__":
    main()
