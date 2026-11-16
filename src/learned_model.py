"""2-stage Metrica-supervised possession inference + BAR-ATH physical fusion + MBR decoding.

Combines:
  1. Stage-1 & Stage-2 LightGBM models trained strictly on Metrica Sports open data
     (Games 1, 2, 3 with ATD observation augmentation and Home<->Away team-swap symmetry):
       - Dead_players: pure collective player kinematics + PELT change-point regimes (ball-blind)
       - Team_live:    symmetrized H vs A team ownership
       - Bound_ha:     H <-> A turnover boundary detector
  2. Unlabeled BAR-ATH physical cues (out-of-bounds, live in-bounds ball rescue, stationary
     uncontested ball) and Smart Tagging priors (GOALS celebrations, SET PIECES restarts,
     single-team offensive phases).
  3. Exact Forward-Backward + FFBS Posterior Sampling + Minimum Bayes Risk (MBR) Decoding
     directly maximizing expected spellF1.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.decoder import (  # noqa: E402
    A, D, H, LABEL_CHARS, X,
    cleanup_short_segments, filter_fast_sandwiches, snap_transitions,
)
from src.mbr_decoder import (  # noqa: E402
    extract_spells, fast_spell_f1, ffbs_sample, forward_log_alpha, viterbi_tv,
)
from src.ml_features import compute_ml_features  # noqa: E402
from src.train_metrica import build_stage2_features  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")


def swap_teams(data):
    """Same game with Home/Away exchanged -- enforces exact team symmetry."""
    out = dict(data)
    out["team_map"] = 1 - np.asarray(data["team_map"])
    if "labels" in data:
        lab = data["labels"].copy()
        lab[data["labels"] == H] = A
        lab[data["labels"] == A] = H
        out["labels"] = lab
    return out


def _sigmoid_logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log(1.0 - p)


def _sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_learned_bundle(model_dir=MODEL_DIR):
    """Load all 6 LightGBM boosters and feature column metadata."""
    import lightgbm as lgb

    with open(os.path.join(model_dir, "feature_meta.json"), "r") as fh:
        meta = json.load(fh)

    names = meta["feature_names"]
    groups = meta["groups"]
    ci = {k: i for i, k in enumerate(names)}
    bundle = {
        "p_cols": [ci[k] for k in groups["player"]],
        "t_cols": [ci[k] for k in groups["team"]],
        "b_cols": [ci[k] for k in groups["boundary"]],
        "m_dead1": lgb.Booster(model_file=os.path.join(model_dir, "stage1_dead_players.txt")),
        "m_team1": lgb.Booster(model_file=os.path.join(model_dir, "stage1_team_live.txt")),
        "m_bound1": lgb.Booster(model_file=os.path.join(model_dir, "stage1_bound_ha.txt")),
        "m_dead2": lgb.Booster(model_file=os.path.join(model_dir, "stage2_dead_players.txt")),
        "m_team2": lgb.Booster(model_file=os.path.join(model_dir, "stage2_team_live.txt")),
        "m_bound2": lgb.Booster(model_file=os.path.join(model_dir, "stage2_bound_ha.txt")),
    }
    return bundle


def predict_stage2_probs(data, ko2, bundle=None):
    """Run Stage-1 + Stage-2 Metrica-trained models on BAR-ATH (play blocks only for PELT)."""
    if bundle is None:
        bundle = load_learned_bundle()

    # Restrict PELT standardization/segmentation to true match-play spans (exclude pre-KO setup).
    hb_orig = data["half_blocks"]
    play_blocks = [(1, hb_orig[0][1], hb_orig[0][2]), (2, ko2, hb_orig[1][2])]
    d_fwd = dict(data)
    d_fwd["half_blocks"] = play_blocks
    d_rev = swap_teams(d_fwd)

    F_fwd, _, _ = compute_ml_features(d_fwd)
    F_rev, _, _ = compute_ml_features(d_rev)

    p_dead1 = np.clip(bundle["m_dead1"].predict(F_fwd[:, bundle["p_cols"]]), 1e-4, 1.0 - 1e-4)
    p_h1_f = bundle["m_team1"].predict(F_fwd[:, bundle["t_cols"]])
    p_h1_r = bundle["m_team1"].predict(F_rev[:, bundle["t_cols"]])
    p_h1 = np.clip(0.5 * (p_h1_f + (1.0 - p_h1_r)), 1e-4, 1.0 - 1e-4)
    p_bha1 = np.clip(bundle["m_bound1"].predict(F_fwd[:, bundle["b_cols"]]), 1e-4, 1.0 - 1e-4)

    Xd, Xt, Xb = build_stage2_features(p_dead1, p_h1, p_bha1)
    _, Xts, _ = build_stage2_features(p_dead1, 1.0 - p_h1, p_bha1)

    p_dead2 = np.clip(bundle["m_dead2"].predict(Xd), 1e-4, 1.0 - 1e-4)
    p_h2_f = bundle["m_team2"].predict(Xt)
    p_h2_r = bundle["m_team2"].predict(Xts)
    p_h2 = np.clip(0.5 * (p_h2_f + (1.0 - p_h2_r)), 1e-4, 1.0 - 1e-4)
    p_bha2 = np.clip(bundle["m_bound2"].predict(Xb), 1e-4, 1.0 - 1e-4)

    return p_dead2, p_h2, p_bha2


def snap_transitions_learned(codes, pbha, dv, d_h, d_a, oob, start_idx, end_idx, min_seg_len=14):
    """Refine transition boundaries using learned boundary peaks (H<->A) and physical OOB onset (L->D)."""
    out = codes.copy()
    seg = out[start_idx:end_idx]
    if len(seg) == 0:
        return out

    breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(seg)]))
    last_cut = start_idx
    for j in range(1, len(bounds) - 1):
        t_cur = start_idx + int(bounds[j])
        next_start = start_idx + int(bounds[j + 1])
        prev_st = int(out[t_cur - 1])
        next_st = int(out[t_cur])
        w0 = max(last_cut + min_seg_len, t_cur - 8)
        w1 = min(next_start - min_seg_len, t_cur + 8)
        best_t = t_cur
        if w1 > w0 + 2:
            if prev_st in (H, A) and next_st in (H, A):
                if float(np.max(pbha[w0:w1])) > 0.20:
                    best_t = w0 + int(np.argmax(pbha[w0:w1]))
                else:
                    d_next = d_h[w0:w1] if next_st == H else d_a[w0:w1]
                    d_prev = d_a[w0:w1] if next_st == H else d_h[w0:w1]
                    if float(np.max(dv[w0:w1])) > 2.5 or float(np.min(d_next)) < 2.2:
                        sc = (dv[w0:w1] / 4.0
                              + np.exp(-np.nan_to_num(d_next, nan=99.0) / 1.5)
                              - 0.5 * np.exp(-np.nan_to_num(d_prev, nan=99.0) / 1.5))
                        best_t = w0 + int(np.argmax(sc))
            elif prev_st in (H, A) and next_st == D:
                w1_oob = min(next_start - min_seg_len, t_cur + 12)
                cand_oob = np.flatnonzero(oob[w0:w1_oob])
                if len(cand_oob):
                    best_t = w0 + int(cand_oob[0])
        if best_t > t_cur:
            out[t_cur:best_t] = prev_st
        elif best_t < t_cur:
            out[best_t:t_cur] = next_st
        last_cut = best_t
    return out


def decode_match_learned(data, features, bundle=None, w0=2.5, w1=4.0, p_min=1.2,
                         p_ad=36.0, p_da=28.0, mbr_temp=1.4, mbr_samples=140, seed=42):
    """Fuse Metrica-trained posteriors with physical & Smart-Tagging priors and decode via MBR."""
    n = data["n_frames"]
    ko2 = features["ko2"]
    half_blocks = data["half_blocks"]

    p_dead_pl, p_h_ml, p_bha = predict_stage2_probs(data, ko2, bundle=bundle)

    # 1. Calibrated log-odds fusion with unlabeled BAR-ATH physics & Smart Tagging
    vis_clean = features["vis_clean"]
    oob = features["oob"]
    b_spd = features["b_spd"]
    d_h = features["d_H"]
    d_a = features["d_A"]
    min_d = np.minimum(d_h, d_a)

    # Metrica-validated live-ball rescue: visible in-bounds ball moving >3.0 m/s near a player
    # is strongly Alive even during slow tactical build-up (offsets player-only dead bias).
    live_in_3 = vis_clean & (~oob) & (b_spd > 3.0) & (min_d < 3.5)
    phys_shift = features["phys_dead_shift"].copy()
    phys_shift = np.where(live_in_3, np.minimum(phys_shift, 0.0) - 2.6, phys_shift)

    dead_logit = _sigmoid_logit(p_dead_pl) + phys_shift
    dead_logit = np.maximum(dead_logit, features["sp_dead_floor"])
    dead_logit = np.minimum(dead_logit, features["sp_live_ceil"])

    vis_i = features["vis_interp"]
    phase_dir = features["phase_dir"]
    team_logit = _sigmoid_logit(p_h_ml)
    team_logit += np.where(vis_i, 0.22 * phase_dir, 1.15 * phase_dir)
    team_logit += features["sp_team_bias"]

    p_dead = _sigmoid(dead_logit)
    p_h = _sigmoid(team_logit)
    emissions = np.column_stack([
        np.log(np.clip(1.0 - p_dead, 1e-6, 1.0)) + np.log(np.clip(p_h, 1e-6, 1.0)),
        np.log(np.clip(1.0 - p_dead, 1e-6, 1.0)) + np.log(np.clip(1.0 - p_h, 1e-6, 1.0)),
        np.log(np.clip(p_dead, 1e-6, 1.0)),
    ])

    # 2. Time-varying transition costs modulated by Stage-2 Boundary Detector + Smart Tagging
    logit_b = _sigmoid_logit(p_bha)
    p_ha_tv = np.clip(w0 - w1 * logit_b, p_min, 32.0)

    sp_boost = features["sp_boost"]
    p_ha_tv = np.where(sp_boost, np.minimum(p_ha_tv, 8.0), p_ha_tv)
    p_ad_tv = np.where(oob, 14.0, p_ad)
    p_da_tv = np.where(sp_boost, 10.0, p_da)

    # 3. Decode each play half via Forward-Backward + FFBS + MBR optimizing spellF1
    codes = np.full(n, X, dtype=np.int8)
    sp_mask = features["sp_restart_mask"]
    dv = features["dv"]
    dtheta = features["dtheta"]

    def _post_clean(seq, s_idx, e_idx, min_poss=12, min_dead=24):
        c = cleanup_short_segments(
            seq, oob[s_idx:e_idx], vis_clean[s_idx:e_idx], sp_mask[s_idx:e_idx],
            min_dead_len=min_dead, min_inbounds_dead_len=48, min_poss_len=min_poss,
        )
        full = codes.copy()
        full[s_idx:e_idx] = c
        full = snap_transitions_learned(full, p_bha, dv, d_h, d_a, oob, s_idx, e_idx, min_seg_len=min_poss)
        # Only absorb brief <25-frame fly-by sandwiches that lack any learned boundary peak
        # (BAR-ATH's smoothed ATD ball track lowers 3-frame dtheta, so avoid aggressive max_len=48).
        full = filter_fast_sandwiches(full, b_spd, dtheta, s_idx, e_idx, max_len=24, max_entry_angle=22.0)
        return full[s_idx:e_idx]

    play_spans = [(half_blocks[0][1], half_blocks[0][2] + 1), (ko2, half_blocks[1][2] + 1)]
    codes[half_blocks[1][1]:ko2] = D

    for idx_h, (s, e) in enumerate(play_spans):
        em_h = emissions[s:e]
        pha_h = p_ha_tv[s:e]
        pad_h = p_ad_tv[s:e]
        pda_h = p_da_tv[s:e]

        alpha = forward_log_alpha(em_h, pha_h, pad_h, pda_h, temp=mbr_temp)
        samp = ffbs_sample(alpha, pha_h, pad_h, pda_h, temp=mbr_temp, n_samples=mbr_samples, seed=seed + idx_h)

        # Clean micro-flickers (<10f / <18f) on posterior samples so single-frame noise doesn't bias spell counts
        samp_spells = [
            extract_spells(cleanup_short_segments(
                samp[k], oob[s:e], vis_clean[s:e], sp_mask[s:e],
                min_dead_len=18, min_inbounds_dead_len=45, min_poss_len=10,
            ))
            for k in range(mbr_samples)
        ]

        # Build rich candidate pool: penalty scales + post-processing lengths + representative samples
        candidates = []
        for sc in (0.65, 0.85, 1.0, 1.15, 1.35):
            raw_v = viterbi_tv(em_h, np.clip(pha_h * sc, 1.0, 36.0), pad_h * sc, pda_h * sc)
            for mp in (10, 13, 17):
                candidates.append(_post_clean(raw_v, s, e, min_poss=mp, min_dead=24))
        for k in range(12):
            candidates.append(_post_clean(samp[k], s, e, min_poss=12, min_dead=22))

        best_cand = candidates[0]
        best_score = -1.0
        seen = set()
        for cand in candidates:
            sp = extract_spells(cand)
            sig = tuple(sp)
            if sig in seen:
                continue
            seen.add(sig)
            avg = sum(fast_spell_f1(ssp, sp) for ssp in samp_spells) / float(mbr_samples)
            if avg > best_score:
                best_score = avg
                best_cand = cand
        codes[s:e] = best_cand

    return LABEL_CHARS[codes], codes
