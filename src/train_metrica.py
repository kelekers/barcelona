"""Train the 2-stage Metrica-supervised possession, dead-ball, and boundary models.

All training data comes strictly from Metrica Sports open sample data (Games 1, 2, 3) with
BAR-ATH-matched ATD observation augmentation (ball smoothing, player position jitter,
tracklet dropout, live ball dropout) and Home<->Away team-swap symmetry. Never accesses
any BAR-ATH labels (neither public nor private).

Produces 6 deterministic LightGBM models + metadata in ./models/:
  * stage1_dead_players.txt  (pure collective player kinematics + PELT; zero ball leakage)
  * stage1_team_live.txt     (H vs A on Alive frames; symmetric via team-swap)
  * stage1_bound_ha.txt      (H <-> A turnover boundary within +/-2 frames)
  * stage2_dead_players.txt  (multi-lag temporal stacker over stage1_dead_players)
  * stage2_team_live.txt     (multi-lag temporal stacker over stage1 team/dead/bound)
  * stage2_bound_ha.txt      (multi-lag symmetric boundary stacker)
  * feature_meta.json        (exact column lists & training CV summary)

Usage:
    python src/train_metrica.py
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features import window_sum_1d  # noqa: E402
from src.metrica_io import A, D, H, X, load_metrica_game  # noqa: E402
from src.ml_features import compute_ml_features  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")


def augment_atd_noise(data, seed=0, smooth_sigma=2.0, ball_jitter_m=0.06,
                      player_jitter_m=0.55, player_drop=0.05, ball_drop_per_min=0.6):
    """Simulate BAR-ATH ATD tracking artifacts (smoothing, jitter, dropouts) on Metrica."""
    rng = np.random.default_rng(seed)
    out = dict(data)
    length, width = data["pitch"]
    bx = data["ball_x"].astype(np.float64).copy()
    by = data["ball_y"].astype(np.float64).copy()
    px = data["px"].astype(np.float64).copy()
    py = data["py"].astype(np.float64).copy()
    n, n_slots = px.shape

    if smooth_sigma > 0:
        vis = np.isfinite(bx)
        edges = np.diff(np.concatenate(([0], vis.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= 8:
                bx[a:b] = gaussian_filter1d(bx[a:b], sigma=smooth_sigma, mode="nearest")
                by[a:b] = gaussian_filter1d(by[a:b], sigma=smooth_sigma, mode="nearest")
    if ball_jitter_m > 0:
        bx += rng.normal(0.0, ball_jitter_m / length, n)
        by += rng.normal(0.0, ball_jitter_m / width, n)

    if ball_drop_per_min > 0:
        n_runs = int(ball_drop_per_min * n / (25 * 60))
        for _ in range(n_runs):
            s = int(rng.integers(0, n))
            ln = int(rng.exponential(35.0)) + 5
            bx[s:s + ln] = np.nan
            by[s:s + ln] = np.nan

    if player_jitter_m > 0:
        px += rng.normal(0.0, player_jitter_m / length, px.shape)
        py += rng.normal(0.0, player_jitter_m / width, py.shape)
    if player_drop > 0:
        p_burst = player_drop / 15.0
        for slot in range(n_slots):
            for s in np.flatnonzero(rng.random(n) < p_burst):
                ln = int(rng.integers(8, 25))
                px[s:s + ln, slot] = np.nan
                py[s:s + ln, slot] = np.nan

    out["ball_x"] = bx.astype(np.float32)
    out["ball_y"] = by.astype(np.float32)
    out["px"] = px.astype(np.float32)
    out["py"] = py.astype(np.float32)
    return out


def swap_game(data):
    """Return a copy of a tracking dict with Home (0) and Away (1) exchanged."""
    out = dict(data)
    out["team_map"] = 1 - np.asarray(data["team_map"])
    if "labels" in data:
        lab = data["labels"].copy()
        lab[data["labels"] == H] = A
        lab[data["labels"] == A] = H
        out["labels"] = lab
    return out


def ha_boundary_targets(labels, half_blocks, radius=2):
    """Binary indicator within +/-radius frames of every H <-> A transition."""
    n = len(labels)
    y = np.zeros(n, dtype=np.int8)
    for _, s, e in half_blocks:
        for t in range(int(s) + 1, int(e) + 1):
            p, q = int(labels[t - 1]), int(labels[t])
            if p in (H, A) and q in (H, A) and p != q:
                y[max(int(s), t - radius):min(int(e) + 1, t + radius + 1)] = 1
    return y


def _wmean(arr, b, f):
    return window_sum_1d(arr, b, f) / window_sum_1d(np.ones(len(arr), dtype=np.float64), b, f)


def _shift(arr, k):
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    if k == 0:
        out[:] = arr
    elif k > 0:
        out[:k] = arr[0]
        out[k:] = arr[:-k]
    else:
        m = -k
        out[:-m] = arr[m:]
        out[-m:] = arr[-1]
    return out


def build_stage2_features(pdead, ph, pbha):
    """Build Stage-2 multi-lag temporal features from Stage-1 probabilities.

    `Xd` uses ONLY `pdead` (ball-blind collective player trajectory) to prevent any Metrica
    ball-visibility bias from leaking into BAR-ATH dead-ball estimation.
    """
    pdead = np.clip(pdead, 1e-4, 1.0 - 1e-4)
    ph = np.clip(ph, 1e-4, 1.0 - 1e-4)
    pbha = np.clip(pbha, 1e-4, 1.0 - 1e-4)

    lt = np.log(ph) - np.log(1.0 - ph)
    ld = np.log(pdead) - np.log(1.0 - pdead)
    lb = np.log(pbha) - np.log(1.0 - pbha)

    # Pure-player Dead2 features
    d_cols = [ld]
    for k in (-60, -30, -15, -6, 6, 15, 30, 60):
        d_cols.append(_shift(ld, k))
    for w in (12, 30, 75, 125):
        d_cols.extend([_wmean(ld, w, w), _wmean(ld, w, 0), _wmean(ld, 0, w)])
    d_cols.append(_wmean(ld, 0, 50) - _wmean(ld, 50, 0))
    Xd = np.column_stack(d_cols).astype(np.float32)

    # Team2 features
    t_cols = [lt, ld, pbha]
    for k in (-35, -20, -10, -5, -2, 2, 5, 10, 20, 35):
        t_cols.append(_shift(lt, k))
    for k in (-12, -5, -2, 2, 5, 12):
        t_cols.append(_shift(pbha, k))
        t_cols.append(_shift(ld, k))
    for w in (5, 12, 25, 50):
        t_cols.extend([_wmean(lt, w, w), _wmean(lt, w, 0), _wmean(lt, 0, w),
                       _wmean(ld, w, w), _wmean(pbha, w, w)])
    t_cols.append(_wmean(lt, 0, 15) - _wmean(lt, 15, 0))
    t_cols.append(_wmean(lt, 0, 35) - _wmean(lt, 35, 0))
    t_cols.append(np.maximum.reduce([_shift(pbha, k) for k in range(-3, 4)]))
    Xt = np.column_stack(t_cols).astype(np.float32)

    # Bound2 features (symmetric under H <-> A)
    alt = np.abs(lt)
    b_cols = [pbha, lb, alt, ld]
    for k in (-20, -10, -5, -2, -1, 1, 2, 5, 10, 20):
        b_cols.append(_shift(pbha, k))
        b_cols.append(_shift(alt, k))
    for w in (6, 15, 30, 50):
        lb_w = _wmean(lt, w, 0)
        lf_w = _wmean(lt, 0, w)
        b_cols.extend([
            np.abs(lf_w - lb_w),
            np.maximum(0.0, -lb_w * lf_w),
            np.abs(lb_w),
            np.abs(lf_w),
            _wmean(pbha, w, w),
        ])
    b_cols.append(np.maximum.reduce([_shift(pbha, k) for k in range(-4, 5)]))
    b_cols.append(np.maximum.reduce([_shift(pbha, k) for k in range(-12, 13)]))
    Xb = np.column_stack(b_cols).astype(np.float32)

    return Xd, Xt, Xb


def _train_stage1_models(games, train_ids, p_cols, t_cols, b_cols):
    lgb_base = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbose": -1,
        "seed": 7,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
    }

    # 1. Stage-1 Dead Players (clean + 2 ATD-noisy variants)
    Xd = np.vstack([
        games[g][fk][games[g]["labels"] != X][:, p_cols]
        for g in train_ids for fk in ("Fc", "Fn1", "Fn2")
    ])
    yd = np.concatenate([
        (games[g]["labels"][games[g]["labels"] != X] == D).astype(np.int8)
        for g in train_ids for _ in (0, 1, 2)
    ])
    m_dead1 = lgb.train(lgb_base, lgb.Dataset(Xd, label=yd), num_boost_round=320)

    # 2. Stage-1 Team Live (clean + ATD-noisy + team-swapped)
    Xt, yt = [], []
    for g in train_ids:
        for fk, lk in (("Fc", "labels"), ("Fcs", "labels_sw"),
                       ("Fn1", "labels"), ("Fn1s", "labels_sw"),
                       ("Fn2", "labels"), ("Fn2s", "labels_sw")):
            lab = games[g][lk]
            mal = lab < 2
            Xt.append(games[g][fk][mal][:, t_cols])
            yt.append((lab[mal] == H).astype(np.int8))
    m_team1 = lgb.train(lgb_base, lgb.Dataset(np.vstack(Xt), label=np.concatenate(yt)), num_boost_round=380)

    # 3. Stage-1 Boundary H <-> A (symmetric)
    Xb, yb = [], []
    for g in train_ids:
        mal = games[g]["labels"] < 2
        yha = games[g]["y_ha"]
        for fk in ("Fc", "Fn1", "Fn2"):
            Xb.append(games[g][fk][mal][:, b_cols])
            yb.append(yha[mal])
    m_bound1 = lgb.train(
        {**lgb_base, "num_leaves": 31, "min_data_in_leaf": 100},
        lgb.Dataset(np.vstack(Xb), label=np.concatenate(yb)),
        num_boost_round=260,
    )
    return m_dead1, m_team1, m_bound1


def _predict_stage1(m_dead1, m_team1, m_bound1, F_fwd, F_rev, p_cols, t_cols, b_cols):
    p_dead = np.clip(m_dead1.predict(F_fwd[:, p_cols]), 1e-4, 1.0 - 1e-4)
    p_h_f = m_team1.predict(F_fwd[:, t_cols])
    p_h_r = m_team1.predict(F_rev[:, t_cols])
    p_h = np.clip(0.5 * (p_h_f + (1.0 - p_h_r)), 1e-4, 1.0 - 1e-4)
    p_bha = np.clip(m_bound1.predict(F_fwd[:, b_cols]), 1e-4, 1.0 - 1e-4)
    return p_dead, p_h, p_bha


def main():
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading Metrica Games 1, 2, 3 and computing clean + ATD-augmented features...")

    games = {}
    names = groups = None
    for g in (1, 2, 3):
        raw = load_metrica_game(g)
        raw_sw = swap_game(raw)
        n1 = augment_atd_noise(raw, seed=100 + g, smooth_sigma=2.0, player_jitter_m=0.55)
        n1_sw = swap_game(n1)
        n2 = augment_atd_noise(raw, seed=200 + g, smooth_sigma=2.6, player_jitter_m=0.75)
        n2_sw = swap_game(n2)

        Fc, names, groups = compute_ml_features(raw)
        Fcs, _, _ = compute_ml_features(raw_sw)
        Fn1, _, _ = compute_ml_features(n1)
        Fn1s, _, _ = compute_ml_features(n1_sw)
        Fn2, _, _ = compute_ml_features(n2)
        Fn2s, _, _ = compute_ml_features(n2_sw)
        games[g] = {
            "raw": raw, "Fc": Fc, "Fcs": Fcs, "Fn1": Fn1, "Fn1s": Fn1s, "Fn2": Fn2, "Fn2s": Fn2s,
            "labels": raw["labels"], "labels_sw": raw_sw["labels"],
            "y_ha": ha_boundary_targets(raw["labels"], raw["half_blocks"], radius=2),
        }

    ci = {k: i for i, k in enumerate(names)}
    p_cols = [ci[k] for k in groups["player"]]
    t_cols = [ci[k] for k in groups["team"]]
    b_cols = [ci[k] for k in groups["boundary"]]
    print(f"Prepared 3 Metrica games in {time.time() - t0:.1f}s "
          f"(player={len(p_cols)}, team={len(t_cols)}, bound={len(b_cols)})")

    # Generate leak-free OOF Stage-1 predictions on both clean and ATD-noisy views
    oof_s2 = {}
    cv_metrics = {}
    for val_g in (1, 2, 3):
        tr = [g for g in (1, 2, 3) if g != val_g]
        md1, mt1, mb1 = _train_stage1_models(games, tr, p_cols, t_cols, b_cols)
        lab = games[val_g]["labels"]
        mal = lab < 2

        oof_s2[val_g] = []
        for fk, fks in (("Fc", "Fcs"), ("Fn1", "Fn1s"), ("Fn2", "Fn2s")):
            pd1, ph1, pb1 = _predict_stage1(
                md1, mt1, mb1, games[val_g][fk], games[val_g][fks], p_cols, t_cols, b_cols
            )
            Xd, Xt, Xb = build_stage2_features(pd1, ph1, pb1)
            _, Xts, _ = build_stage2_features(pd1, 1.0 - ph1, pb1)
            oof_s2[val_g].append((Xd, Xt, Xts, Xb))
            if fk == "Fn1":
                cv_metrics[val_g] = {
                    "s1_auc_dead": float(roc_auc_score(lab == D, pd1)),
                    "s1_auc_team": float(roc_auc_score(lab[mal] == H, ph1[mal])),
                    "s1_ap_bha": float(average_precision_score(games[val_g]["y_ha"][mal], pb1[mal])),
                }
        print(f"Fold {val_g} Stage-1 OOF (ATD-noisy): {cv_metrics[val_g]}")

    # Validate Stage-2 OOF
    lgb_s2 = {
        "objective": "binary",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbose": -1,
        "seed": 7,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
    }

    for val_g in (1, 2, 3):
        tr = [g for g in (1, 2, 3) if g != val_g]
        Xd_tr = np.vstack([oof_s2[g][v][0][games[g]["labels"] != X] for g in tr for v in range(3)])
        yd_tr = np.concatenate([(games[g]["labels"][games[g]["labels"] != X] == D).astype(np.int8) for g in tr for _ in range(3)])
        md2 = lgb.train(lgb_s2, lgb.Dataset(Xd_tr, label=yd_tr), num_boost_round=180)

        Xt_tr, yt_tr = [], []
        for g in tr:
            lab, labs = games[g]["labels"], games[g]["labels_sw"]
            mal, mals = lab < 2, labs < 2
            for v in range(3):
                Xt_tr.append(oof_s2[g][v][1][mal]); yt_tr.append((lab[mal] == H).astype(np.int8))
                Xt_tr.append(oof_s2[g][v][2][mals]); yt_tr.append((labs[mals] == H).astype(np.int8))
        mt2 = lgb.train(lgb_s2, lgb.Dataset(np.vstack(Xt_tr), label=np.concatenate(yt_tr)), num_boost_round=200)

        Xb_tr = np.vstack([oof_s2[g][v][3][games[g]["labels"] < 2] for g in tr for v in range(3)])
        yb_tr = np.concatenate([games[g]["y_ha"][games[g]["labels"] < 2] for g in tr for _ in range(3)])
        mb2 = lgb.train(lgb_s2, lgb.Dataset(Xb_tr, label=yb_tr), num_boost_round=180)

        Xd_v, Xt_v, Xts_v, Xb_v = oof_s2[val_g][1]  # Fn1
        lab = games[val_g]["labels"]
        mal = lab < 2
        pd2 = np.clip(md2.predict(Xd_v), 1e-4, 1.0 - 1e-4)
        ph2 = np.clip(0.5 * (mt2.predict(Xt_v) + (1.0 - mt2.predict(Xts_v))), 1e-4, 1.0 - 1e-4)
        pb2 = np.clip(mb2.predict(Xb_v), 1e-4, 1.0 - 1e-4)
        cv_metrics[val_g]["s2_auc_dead"] = float(roc_auc_score(lab == D, pd2))
        cv_metrics[val_g]["s2_auc_team"] = float(roc_auc_score(lab[mal] == H, ph2[mal]))
        cv_metrics[val_g]["s2_ap_bha"] = float(average_precision_score(games[val_g]["y_ha"][mal], pb2[mal]))
        print(f"Fold {val_g} Stage-2 OOF (ATD-noisy): AUC_dead={cv_metrics[val_g]['s2_auc_dead']:.4f}, "
              f"AUC_team={cv_metrics[val_g]['s2_auc_team']:.4f}, AP_bha={cv_metrics[val_g]['s2_ap_bha']:.4f}")

    # Final training on ALL 3 Metrica games (Stage 1 on full 3 games, Stage 2 on full OOF)
    print("Training final Stage-1 and Stage-2 models on all 3 Metrica games...")
    m_dead1, m_team1, m_bound1 = _train_stage1_models(games, (1, 2, 3), p_cols, t_cols, b_cols)

    Xd_all = np.vstack([oof_s2[g][v][0][games[g]["labels"] != X] for g in (1, 2, 3) for v in range(3)])
    yd_all = np.concatenate([(games[g]["labels"][games[g]["labels"] != X] == D).astype(np.int8) for g in (1, 2, 3) for _ in range(3)])
    m_dead2 = lgb.train(lgb_s2, lgb.Dataset(Xd_all, label=yd_all), num_boost_round=180)

    Xt_all, yt_all = [], []
    for g in (1, 2, 3):
        lab, labs = games[g]["labels"], games[g]["labels_sw"]
        mal, mals = lab < 2, labs < 2
        for v in range(3):
            Xt_all.append(oof_s2[g][v][1][mal]); yt_all.append((lab[mal] == H).astype(np.int8))
            Xt_all.append(oof_s2[g][v][2][mals]); yt_all.append((labs[mals] == H).astype(np.int8))
    m_team2 = lgb.train(lgb_s2, lgb.Dataset(np.vstack(Xt_all), label=np.concatenate(yt_all)), num_boost_round=200)

    Xb_all = np.vstack([oof_s2[g][v][3][games[g]["labels"] < 2] for g in (1, 2, 3) for v in range(3)])
    yb_all = np.concatenate([games[g]["y_ha"][games[g]["labels"] < 2] for g in (1, 2, 3) for _ in range(3)])
    m_bound2 = lgb.train(lgb_s2, lgb.Dataset(Xb_all, label=yb_all), num_boost_round=180)

    m_dead1.save_model(os.path.join(MODEL_DIR, "stage1_dead_players.txt"))
    m_team1.save_model(os.path.join(MODEL_DIR, "stage1_team_live.txt"))
    m_bound1.save_model(os.path.join(MODEL_DIR, "stage1_bound_ha.txt"))
    m_dead2.save_model(os.path.join(MODEL_DIR, "stage2_dead_players.txt"))
    m_team2.save_model(os.path.join(MODEL_DIR, "stage2_team_live.txt"))
    m_bound2.save_model(os.path.join(MODEL_DIR, "stage2_bound_ha.txt"))

    meta = {
        "feature_names": names,
        "groups": groups,
        "cv_metrics": cv_metrics,
    }
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved 6 models + feature_meta.json to {MODEL_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
