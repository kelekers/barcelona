"""Stage-2 Temporal Context Stacker, Boundary Detectors, and Pure-NumPy MLP Ensemble.

Operates on top of Stage-1 DFL external model probabilities (`prob1 = [P(H), P(A), P(D)]`):
  1. `bound_ha`: symmetric H <-> A turnover boundary detector using multi-scale ball impulses,
                 proximity shifts, and symmetric Stage-1 team probability transitions.
  2. `dead2` & `bound_dl`: 100% ball-blind temporal refinement of P(D) and P(D <-> Live) boundaries.
  3. `team2`: symmetrized temporal refinement of P(H | Live).
  4. Pure-NumPy `MLP` blend in log-odds space (`w_mlp = 0.25`) for cross-family model diversity
     without adding scikit-learn as an inference requirement.
"""

import json
import os

import numpy as np

from .common import A, D, FPS, H, MODEL_DIR

BOUND_HA_FILE = "bound_ha.txt"
BOUND_DL_FILE = "bound_dl.txt"
DEAD2_FILE = "dead2.txt"
TEAM2_FILE = "team2.txt"
MLP_FILE = "stage2_mlp.npz"
META_FILE = "stage2_meta.json"


def wmean(arr, back, fwd):
    """Centered/asymmetric rolling mean over [t-back, t+fwd] of a NaN-free array."""
    a = np.asarray(arr, dtype=np.float64)
    cs = np.concatenate(([0.0], np.cumsum(a)))
    n = len(a)
    idx = np.arange(n)
    lo = np.maximum(0, idx - back)
    hi = np.minimum(n, idx + fwd + 1)
    return (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)


def ewm_bidir(arr, tau):
    """Bidirectional (forward + backward) exponential moving average with decay tau frames."""
    alpha = 1.0 - np.exp(-1.0 / float(tau))
    one_m = 1.0 - alpha
    a_l = np.asarray(arr, dtype=np.float64).tolist()
    n = len(a_l)
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    fwd = np.empty(n, dtype=np.float64)
    bwd = np.empty(n, dtype=np.float64)
    v = float(a_l[0])
    fwd[0] = v
    for i in range(1, n):
        v = alpha * a_l[i] + one_m * v
        fwd[i] = v
    v = float(a_l[-1])
    bwd[-1] = v
    for i in range(n - 2, -1, -1):
        v = alpha * a_l[i] + one_m * v
        bwd[i] = v
    return fwd, bwd


def shift(arr, k):
    """Edge-replicated time shift (positive k shifts signal earlier samples forward to t)."""
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


def logit(p):
    """Clipped log-odds log(p / (1 - p))."""
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log(1.0 - p)


def sigmoid(x):
    """Numerically clipped logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def build_bound_features(match, prob1, cues):
    """Symmetric H <-> A boundary feature matrix [n_frames, n_feat] without team bias."""
    bx = match["ball_x"].astype(np.float64)
    by = match["ball_y"].astype(np.float64)
    length, width = match["pitch"]
    fps = match.get("fps", FPS)
    n = len(bx)
    vis = cues["vis_i"]
    bxm = np.where(vis, bx, np.nan) * length
    bym = np.where(vis, by, np.nan) * width

    def _imp(span):
        dv = np.zeros(n, dtype=np.float64)
        dth = np.zeros(n, dtype=np.float64)
        if n > 2 * span:
            ok = vis[2 * span:] & vis[span:-span] & vis[:-2 * span]
            vp_x = (bxm[span:-span] - bxm[:-2 * span]) * (fps / span)
            vp_y = (bym[span:-span] - bym[:-2 * span]) * (fps / span)
            vq_x = (bxm[2 * span:] - bxm[span:-span]) * (fps / span)
            vq_y = (bym[2 * span:] - bym[span:-span]) * (fps / span)
            dv[span:-span] = np.where(ok, np.clip(np.hypot(vq_x - vp_x, vq_y - vp_y), 0.0, 40.0), 0.0)
            s1, s2 = np.hypot(vp_x, vp_y), np.hypot(vq_x, vq_y)
            oka = ok & (s1 > 1.8) & (s2 > 1.8)
            ct = np.clip((vp_x * vq_x + vp_y * vq_y) / np.maximum(s1 * s2, 1e-6), -1.0, 1.0)
            dth[span:-span] = np.where(oka, np.degrees(np.arccos(ct)), 0.0)
        return dv, dth

    dv3, th3 = _imp(3)
    dv5, th5 = _imp(5)
    dv8, th8 = _imp(8)

    dh = np.where(np.isfinite(cues["d_h"]), np.minimum(cues["d_h"], 30.0), 30.0)
    da = np.where(np.isfinite(cues["d_a"]), np.minimum(cues["d_a"], 30.0), 30.0)
    mind = np.minimum(dh, da)
    oppd = np.maximum(dh, da)
    with np.errstate(invalid="ignore"):
        dd = np.where(vis & np.isfinite(dh) & np.isfinite(da), np.clip(da - dh, -12.0, 12.0), 0.0)
    t_dd = np.tanh(dd / 2.5)
    close_h = (vis & (dh < 2.5) & (dh + 0.3 < da)).astype(np.float64)
    close_a = (vis & (da < 2.5) & (da + 0.3 < dh)).astype(np.float64)

    p_live = np.maximum(prob1[:, H] + prob1[:, A], 1e-4)
    lt = logit(np.clip(prob1[:, H] / p_live, 1e-4, 1.0 - 1e-4))
    alt = np.abs(lt)

    cols = [
        cues["b_spd"], cues["dv"], dv3, dv5, dv8, th3, th5, th8,
        mind, oppd, np.abs(dd), cues["mps_fast"], cues["p90_fast"],
        alt, shift(alt, -5), shift(alt, 5), shift(alt, -12), shift(alt, 12),
    ]
    for tau in (6, 18):
        f_l, b_l = ewm_bidir(lt, tau)
        cols.extend([np.abs(f_l - b_l), np.maximum(0.0, -f_l * b_l)])
    for w in (4, 8, 16, 32):
        lb, lf = wmean(lt, w, 0), wmean(lt, 0, w)
        cols.extend([np.abs(lf - lb), np.maximum(0.0, -lb * lf), np.abs(lb), np.abs(lf)])
    for w in (6, 12, 25, 50):
        ch_b, ca_b = wmean(close_h, w, 0), wmean(close_a, w, 0)
        ch_f, ca_f = wmean(close_h, 0, w), wmean(close_a, 0, w)
        dd_b, dd_f = wmean(t_dd, w, 0), wmean(t_dd, 0, w)
        cols.extend([
            np.abs((ch_f - ca_f) - (ch_b - ca_b)),
            np.maximum(ch_b * ca_f, ca_b * ch_f),
            np.abs(dd_f - dd_b),
            np.maximum(0.0, -dd_b * dd_f),
            np.abs(dd_b), np.abs(dd_f),
        ])
    cols.append(np.maximum.reduce([shift(dv5, k) for k in range(-4, 5)]))
    cols.append(np.maximum.reduce([shift(dv8, k) for k in range(-4, 5)]))
    cols.append(np.minimum.reduce([shift(mind, k) for k in range(-4, 5)]))
    return np.column_stack(cols).astype(np.float32)


def build_stage2_features(prob1, cues, pbha):
    """Build (Xd, Xt_fwd, Xt_rev). `Xd` is strictly ball-blind (no ball visibility inputs)."""
    p_dead = np.clip(prob1[:, D], 1e-4, 1.0 - 1e-4)
    p_live = np.maximum(prob1[:, H] + prob1[:, A], 1e-4)
    p_team = np.clip(prob1[:, H] / p_live, 1e-4, 1.0 - 1e-4)
    ld = logit(p_dead)
    lt = logit(p_team)

    dc = [ld, cues["mps_fast"], cues["p90_fast"]]
    for tau in (8, 25, 60):
        df, db = ewm_bidir(ld, tau)
        dc.extend([df, db, db - df])
    for k in (-60, -30, -15, -6, 6, 15, 30, 60):
        dc.append(shift(ld, k))
    for w in (10, 25, 60, 125):
        dc.extend([wmean(ld, w, w), wmean(ld, w, 0), wmean(ld, 0, w)])
    dc.append(wmean(ld, 0, 50) - wmean(ld, 50, 0))
    dc.append(np.abs(wmean(ld, 0, 25) - wmean(ld, 25, 0)))
    dc.append(wmean(cues["mps_fast"], 0, 40) - wmean(cues["mps_fast"], 40, 0))
    dc.append(np.abs(wmean(cues["mps_fast"], 0, 25) - wmean(cues["mps_fast"], 25, 0)))
    xd = np.column_stack(dc).astype(np.float32)

    pb_ewm_f, pb_ewm_b = ewm_bidir(pbha, 6)

    def _xt(lt_v):
        tc = [lt_v, ld, pbha, pb_ewm_f, pb_ewm_b]
        for tau in (5, 15, 35):
            tf, tb = ewm_bidir(lt_v, tau)
            tc.extend([tf, tb, tb - tf])
        for k in (-40, -25, -12, -6, -2, 2, 6, 12, 25, 40):
            tc.append(shift(lt_v, k))
        for k in (-12, -4, 4, 12):
            tc.append(shift(ld, k))
            tc.append(shift(pbha, k))
        for w in (5, 12, 25, 50):
            tc.extend([wmean(lt_v, w, w), wmean(lt_v, w, 0), wmean(lt_v, 0, w), wmean(ld, w, w)])
        tc.append(wmean(lt_v, 0, 15) - wmean(lt_v, 15, 0))
        tc.append(wmean(lt_v, 0, 35) - wmean(lt_v, 35, 0))
        tc.append(np.maximum.reduce([shift(pbha, k) for k in range(-3, 4)]))
        return np.column_stack(tc).astype(np.float32)

    return xd, _xt(lt), _xt(-lt)


def mlp_forward(x_mat, params, prefix):
    """Exact deterministic NumPy forward pass matching sklearn MLPClassifier(relu) + StandardScaler."""
    mean = params[prefix + "_mean"].astype(np.float64)
    scale = params[prefix + "_scale"].astype(np.float64)
    z = (x_mat.astype(np.float64) - mean) / scale
    n_layers = int(params[prefix + "_nlayers"])
    for i in range(n_layers):
        w = params[f"{prefix}_w{i}"].astype(np.float64)
        b = params[f"{prefix}_b{i}"].astype(np.float64)
        z = z @ w + b
        if i < n_layers - 1:
            np.maximum(z, 0.0, out=z)
    return sigmoid(z.ravel())


def _extract_mlp_params(scaler, mlp, prefix):
    out = {
        prefix + "_mean": scaler.mean_.astype(np.float64),
        prefix + "_scale": scaler.scale_.astype(np.float64),
        prefix + "_nlayers": np.array(len(mlp.coefs_), dtype=np.int32),
    }
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        out[f"{prefix}_w{i}"] = w.astype(np.float64)
        out[f"{prefix}_b{i}"] = b.astype(np.float64)
    return out


def compute_targets(truth, blocks):
    """Return (y_ha, y_dl, y_band) transition indicators inside play blocks."""
    y_ha = np.zeros(len(truth), dtype=np.int8)
    y_dl = np.zeros(len(truth), dtype=np.int8)
    y_band = np.zeros(len(truth), dtype=np.int8)
    for _, s, e in blocks:
        for t in range(int(s) + 1, int(e) + 1):
            a, b = int(truth[t - 1]), int(truth[t])
            if a != b and a <= D and b <= D:
                y_band[max(int(s), t - 10):min(int(e) + 1, t + 11)] = 1
            if a in (H, A) and b in (H, A) and a != b:
                y_ha[max(int(s), t - 2):min(int(e) + 1, t + 3)] = 1
            elif (a == D and b in (H, A)) or (b == D and a in (H, A)):
                y_dl[max(int(s), t - 3):min(int(e) + 1, t + 4)] = 1
    return y_ha, y_dl, y_band


def fit_stage2_models(train_items, nested_oof_items=None):
    """Fit Boundary + Stage-2 LightGBM + MLP ensemble from external items.

    Each item is a dict with keys: `prob`, `truth`, `blocks`, `cues`, `xb` (optional `y_ha`, `y_dl`, `y_band`).
    If `nested_oof_items` (list of (group_key, item)) is given, `pb_tr` is produced cleanly out-of-fold per group.
    """
    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    lgb_p = dict(
        objective="binary", learning_rate=0.04, num_leaves=31, min_data_in_leaf=150,
        feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1, lambda_l2=8.0,
        verbose=-1, seed=7, deterministic=True, force_row_wise=True, num_threads=6,
    )

    items = [it for _, it in nested_oof_items] if nested_oof_items is not None else list(train_items)
    for it in items:
        if "y_ha" not in it:
            y_ha, y_dl, y_band = compute_targets(it["truth"], it["blocks"])
            it["y_ha"], it["y_dl"], it["y_band"] = y_ha, y_dl, y_band

    xb_all = np.vstack([it["xb"][it["truth"] < 2] for it in items])
    yb_all = np.concatenate([it["y_ha"][it["truth"] < 2] for it in items])
    m_bha = lgb.train(lgb_p, lgb.Dataset(xb_all, label=yb_all), num_boost_round=240)

    if nested_oof_items is not None:
        groups = sorted({gk for gk, _ in nested_oof_items})
        pb_list = [None] * len(nested_oof_items)
        for gk in groups:
            xb_in = np.vstack([it["xb"][it["truth"] < 2] for g, it in nested_oof_items if g != gk])
            yb_in = np.concatenate([it["y_ha"][it["truth"] < 2] for g, it in nested_oof_items if g != gk])
            mb_in = lgb.train(lgb_p, lgb.Dataset(xb_in, label=yb_in), num_boost_round=220)
            for idx, (g, it) in enumerate(nested_oof_items):
                if g == gk:
                    pb_list[idx] = mb_in.predict(it["xb"])
    else:
        pb_list = [m_bha.predict(it["xb"]) for it in items]

    xd_tr, yd_tr, wd_tr, ydl_tr, xt_tr, yt_tr, wt_tr = [], [], [], [], [], [], []
    for it, pbha in zip(items, pb_list):
        xd, xt_f, xt_r = build_stage2_features(it["prob"], it["cues"], pbha)
        truth = it["truth"]
        gr = truth != 3
        al = truth < 2
        w_all = 1.0 + 1.5 * it["y_band"].astype(np.float32)
        xd_tr.append(xd[gr]); yd_tr.append((truth[gr] == D).astype(np.int8))
        wd_tr.append(w_all[gr]); ydl_tr.append(it["y_dl"][gr])
        xt_tr.append(xt_f[al]); yt_tr.append((truth[al] == H).astype(np.int8)); wt_tr.append(w_all[al])
        xt_tr.append(xt_r[al]); yt_tr.append((truth[al] == A).astype(np.int8)); wt_tr.append(w_all[al])

    xd_m, yd_v, wd_v, ydl_v = np.vstack(xd_tr), np.concatenate(yd_tr), np.concatenate(wd_tr), np.concatenate(ydl_tr)
    xt_m, yt_v, wt_v = np.vstack(xt_tr), np.concatenate(yt_tr), np.concatenate(wt_tr)

    m_dead2 = lgb.train(lgb_p, lgb.Dataset(xd_m, label=yd_v, weight=wd_v), num_boost_round=180)
    m_bdl = lgb.train(lgb_p, lgb.Dataset(xd_m, label=ydl_v), num_boost_round=160)
    m_team2 = lgb.train(lgb_p, lgb.Dataset(xt_m, label=yt_v, weight=wt_v), num_boost_round=200)

    sc_d = StandardScaler().fit(xd_m[::4])
    mlp_d = MLPClassifier((64, 32), alpha=2.5, max_iter=35, random_state=7, early_stopping=True)
    mlp_d.fit(sc_d.transform(xd_m[::4]), yd_v[::4], sample_weight=wd_v[::4])

    sc_t = StandardScaler().fit(xt_m[::4])
    mlp_t = MLPClassifier((64, 32), alpha=2.5, max_iter=35, random_state=7, early_stopping=True)
    mlp_t.fit(sc_t.transform(xt_m[::4]), yt_v[::4], sample_weight=wt_v[::4])

    mlp_params = {}
    mlp_params.update(_extract_mlp_params(sc_d, mlp_d, "dead"))
    mlp_params.update(_extract_mlp_params(sc_t, mlp_t, "team"))

    return {
        "m_bha": m_bha,
        "m_bdl": m_bdl,
        "m_dead2": m_dead2,
        "m_team2": m_team2,
        "mlp": mlp_params,
        "w_mlp": 0.25,
    }


def predict_stage2(bundle, match, prob1, cues):
    """Return (prob2 [n, 3], pbha [n], pbdl [n]) deterministically using NumPy + LightGBM."""
    xb = build_bound_features(match, prob1, cues)
    pbha = np.clip(bundle["m_bha"].predict(xb), 1e-4, 1.0 - 1e-4)
    xd, xt_f, xt_r = build_stage2_features(prob1, cues, pbha)

    pbdl = np.clip(bundle["m_bdl"].predict(xd), 1e-4, 1.0 - 1e-4)
    ld_lgb = logit(bundle["m_dead2"].predict(xd))
    ld_mlp = logit(mlp_forward(xd, bundle["mlp"], "dead"))

    lh_lgb = logit(0.5 * (bundle["m_team2"].predict(xt_f) + (1.0 - bundle["m_team2"].predict(xt_r))))
    p_mlp_f = mlp_forward(xt_f, bundle["mlp"], "team")
    p_mlp_r = mlp_forward(xt_r, bundle["mlp"], "team")
    lh_mlp = logit(0.5 * (p_mlp_f + (1.0 - p_mlp_r)))

    wm = float(bundle.get("w_mlp", 0.25))
    pd = sigmoid((1.0 - wm) * ld_lgb + wm * ld_mlp)
    ph = sigmoid((1.0 - wm) * lh_lgb + wm * lh_mlp)
    prob2 = np.column_stack([(1.0 - pd) * ph, (1.0 - pd) * (1.0 - ph), pd])
    return prob2, pbha, pbdl


def stage2_available(model_dir=MODEL_DIR):
    """True when Stage-2 boosters and MLP parameters exist in model_dir."""
    req = (BOUND_HA_FILE, BOUND_DL_FILE, DEAD2_FILE, TEAM2_FILE, MLP_FILE)
    return all(os.path.exists(os.path.join(model_dir, f)) for f in req)


def save_stage2(bundle, extra_meta, model_dir=MODEL_DIR):
    """Save Stage-2 boosters (with LF line endings), MLP .npz weights, and metadata."""
    os.makedirs(model_dir, exist_ok=True)
    for key, fname in (
        ("m_bha", BOUND_HA_FILE),
        ("m_bdl", BOUND_DL_FILE),
        ("m_dead2", DEAD2_FILE),
        ("m_team2", TEAM2_FILE),
    ):
        with open(os.path.join(model_dir, fname), "w", newline="\n") as handle:
            handle.write(bundle[key].model_to_string())
    np.savez(os.path.join(model_dir, MLP_FILE), **bundle["mlp"])
    meta = dict(extra_meta, w_mlp=float(bundle.get("w_mlp", 0.25)))
    with open(os.path.join(model_dir, META_FILE), "w", newline="\n") as handle:
        json.dump(meta, handle, indent=1)


def _read_booster(path):
    import lightgbm as lgb
    with open(path, "rb") as handle:
        text = handle.read().replace(b"\r\n", b"\n").decode("utf-8")
    return lgb.Booster(model_str=text)


def load_stage2(model_dir=MODEL_DIR):
    """Load (bundle, meta) for Stage-2 + Boundary + pure-NumPy MLP."""
    meta = {}
    mpath = os.path.join(model_dir, META_FILE)
    if os.path.exists(mpath):
        with open(mpath) as handle:
            meta = json.load(handle)
    with np.load(os.path.join(model_dir, MLP_FILE)) as z:
        mlp = {k: z[k] for k in z.files}
    bundle = {
        "m_bha": _read_booster(os.path.join(model_dir, BOUND_HA_FILE)),
        "m_bdl": _read_booster(os.path.join(model_dir, BOUND_DL_FILE)),
        "m_dead2": _read_booster(os.path.join(model_dir, DEAD2_FILE)),
        "m_team2": _read_booster(os.path.join(model_dir, TEAM2_FILE)),
        "mlp": mlp,
        "w_mlp": float(meta.get("w_mlp", 0.25)),
    }
    return bundle, meta
