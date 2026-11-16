# Technical Document — More than a Hack 2027 (Phase 1: Ball Possession)

## 1. Approach

We frame frame-level ball possession inference (`H` = Home / Barcelona, `A` = Away / Athletic Club, `D` = Dead Ball) as a **2-stage externally supervised spatiotemporal classification and boundary-detection problem** decoded via **Forward-Backward Posterior Sampling + Minimum Bayes Risk (MBR) Decoding** optimizing expected `spellF1` directly:

1. **Domain-Aligned Multi-Head Architecture (No `BAR-ATH` Labels Used):**
   Instead of hand-tuning emission weights or fitting a monolithic model that would overfit to Metrica's `ball = NaN` dead-ball convention, we decompose prediction into three specialized LightGBM heads (`src/train_metrica.py`, `src/ml_features.py`):
   - **`Dead_players` (`stage1_dead_players`, `stage2_dead_players`):** Trained exclusively on **ball-blind collective player kinematics and exact PELT change-point regime features** (`src/changepoints.py`), symmetrized under `H <-> A`. Because collective player speed quantiles (`mean`, `p90`, `walk`, `sprint`) match `BAR-ATH` almost identically (`median mps = 1.55` vs `1.53 m/s`), this head generalizes across tracking providers without leaking Metrica's `ball = NaN` dead-ball convention.
   - **`Team_live` (`stage1_team_live`, `stage2_team_live`):** Predicts `P(H | Alive, X)` using permutation-invariant player-ball proximity, velocity alignment, local player density, multi-scale ball impulses, and team spread/advancement. Trained with **ATD observation noise augmentation** (Gaussian ball smoothing, player position jitter, tracklet dropout, live ball dropout) and **Home/Away swap symmetry** at both training and inference time (`0.5 * (p_fwd + 1 - p_rev)`).
   - **`Bound_ha` (`stage1_bound_ha`, `stage2_bound_ha`):** Predicts `P(H <-> A transition within ±2 frames)` from team-symmetric cross-boundary shifts (`abs_ctrl_shift`, `abs_ddiff_shift`, `cross_ctrl`, `sign_flip_ddiff`) and ball impulses (`dv3`, `dv5`, `dv8`, `dtheta5`, `dtheta8`).

2. **Exact PELT Collective Regime Change-Points (`src/changepoints.py`):**
   Exact Pruned Exact Linear Time (`pelt_l2_blocks`) segmentation is run per play half on standardized 4D collective motion `[mps, p90, walk, sprint]` at fine (`penalty=28`) and medium (`penalty=65`) scales with 1-frame local refinement, capturing abrupt collective whistle stoppages and restart accelerations even for brief stoppages (`30–60` frames) without ball `NaN`s.

3. **Stage-2 Temporal Context Stacking (`src/train_metrica.py`):**
   Multi-lag (`±2..60` frames) and multi-window (`5..125` frames) temporal stackers trained on out-of-fold Stage-1 predictions boost Metrica Leave-One-Game-Out CV from `0.928–0.938` to `0.937–0.946` (`AUC_dead`), `0.986–0.990` to `0.988–0.992` (`AUC_team`), and `0.437–0.493` to `0.519–0.577` (`AP_bound_ha`).

4. **Physical & Smart-Tagging Fusion + Forward-Backward MBR Decoding (`src/learned_model.py`, `src/mbr_decoder.py`):**
   Stage-2 log-odds are fused with unlabeled `BAR-ATH` physical signals (out-of-bounds `oob`, live in-bounds ball rescue `b_spd > 3.0 & min_d < 3.5`, stationary uncontested ball, boundary-exit gaps) and Smart Tagging anchors (`GOALS`, `SET PIECES`, single-team offensive phase tie-breakers). The transition cost `p_ha(t) = clip(2.5 - 4.0 * logit(p_bha(t)), 1.2, 32.0)` is driven directly by the learned boundary detector. Exact Forward-Backward + FFBS draws `140` posterior trajectories per half, and MBR selects the candidate maximizing expected `spellF1` over `[0.3, 0.4, 0.5, 0.6, 0.7]` IoU thresholds.

5. **Compliance & Reproducibility:**
   All models, transition formulas, and hyperparameters were selected and cross-validated solely on Metrica Games 1–3 and unlabeled `BAR-ATH` diagnostics — never fitted on the 5-minute `BAR-ATH` public ground truth. Running `python src/run_pipeline.py --out evaluation/results/submission.csv` reproduces byte-identical `submission.csv` output deterministically in ~25 seconds.

---

## 2. Assets Used and Dropped

### Used
- **External Training Dataset — Metrica Sports Sample Data (`github.com/metrica-sports/sample-data`):** Sample Games 1, 2, and 3 (`145,006`, `141,156`, and `143,761` frames at `25 fps`, totaling `429,923` frames and `1,160` possession spells). Converted deterministically from Metrica's public event logs (`PASS`, `RECOVERY`, `BALL LOST`, `SET PIECE`, `SHOT`, `BALL OUT`, `FAULT RECEIVED`, `CARRY`, `CHALLENGE`) to `H/A/D/X` frame labels in `src/metrica_io.py`.
- **Tactical ATD (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`):** Primary tracking signal (`px`, `py`, cleaned `ball_x`, `ball_y`, `team_map`) across all `199,319` frames.
- **Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`):** Used strictly as soft log-odds priors and restart/celebration anchors (`GOALS`, `SET PIECES` with touchline throw-in spatial voting for `Team = N/A`, and single-team offensive phase tie-breakers).
- **Match Timing Metadata (`games.py`):** Half boundaries (`11185..90651`, `110808..190197`) and second-half kickoff (`113742`).

### Evaluated and Dropped
- **Provided ATD `ball_s` column & Broadcast Video/ATD:** `ball_s` is sparsely populated; velocity is computed from cleaned `(ball_x, ball_y)`. Broadcast ATD (34.2% invalid homography, unsynchronized cuts/replays) and video decoding were omitted in favor of the deterministic tracking + Metrica-supervised pipeline.

---

## 3. Handling Missing and Noisy Tracking Data

1. **Ball Glitch & Outlier Filtering (`src/atd_io.py`):** Removes 1–2 frame tracker spikes (`> 2.2 m/frame` with snap-back) and isolated `<= 4`-frame islands jumping `> 12 m`.
2. **ATD Smoothing & Radial Offset Alignment (`src/train_metrica.py`):** Audit of unlabeled `BAR-ATH` kinematics showed that ATD's ball track is temporally smoothed (`dv` span=3 p95 `2.65 m/s` vs `7.96 m/s` in raw Metrica) and `min_d` has a larger offset (`median 1.82 m` vs `0.29 m` in raw Metrica). Training combines clean Metrica with two ATD-augmented copies (`gaussian_filter1d` ball smoothing `sigma=2.0/2.6`, `0.55/0.75 m` player jitter, burst tracklet dropouts, and live ball dropout runs).
3. **Ball-Blind Dead-Ball Estimation + Live-Ball Rescue:** `Dead_players` sees only player kinematics and PELT regimes, avoiding Metrica's `ball = NaN` dead-ball bias. On `BAR-ATH`, visible in-bounds motion (`b_spd > 3.0 & min_d < 3.5`) provides a `-2.6` log-odds live rescue to protect slow tactical build-up pauses from false `D` predictions, while `oob` (`+5.5`) and `SET PIECES` priors catch visible dead balls.

---

## 4. Training Data

- **External Open Dataset:** Metrica Sports Sample Games 1, 2, and 3 (`data/external/metrica/`), human-annotated event logs paired with `25 fps` optical tracking, converted deterministically via `src/metrica_io.py`.
  - Game 1: `145,006` frames (`40,911 H`, `38,984 A`, `65,111 D`, `381` possessions).
  - Game 2: `141,156` frames (`42,882 H`, `39,764 A`, `58,510 D`, `364` possessions).
  - Game 3: `143,761` frames (`49,831 H`, `47,832 A`, `46,098 D`, `415` possessions).
- **`BAR-ATH` Labels Used:** **None.** Neither the 5-minute public GT nor any hand labels on `BAR-ATH` were used for training, feature selection, or hyperparameter tuning.

---

## 5. Cross-Validation & Sanity Checks

- **Metrica Leave-One-Game-Out CV (ATD-noisy held-out matches):**
  - Heuristic baseline (without Smart Tagging): mean `spellF1 = 0.3695` (`0.3698`, `0.3481`, `0.3907`).
  - 2-stage Metrica GBDT + PELT + Boundary-modulated decoding + live-ball rescue: mean `spellF1 = 0.6654` (`0.7012`, `0.6650`, `0.6301`), `macroF1 = 0.8773`.
- **Full-Match `BAR-ATH` Unlabeled Diagnostics (`158,857` graded frames):**
  - `H: 50,711 (31.9%)`, `A: 31,841 (20.0%)`, `D: 76,305 (48.0%)`, `Home alive share: 61.4%`, `297` possessions (`mean 11.1 s`).
  - Goal celebration dead-ball `D%`: `100.0%`, `100.0%`, `97.7%`, `100.0%`.
  - Non-kickoff Set Pieces alignment: `51/53` pre-restart `D`, `51/52` restarting team match within `+2s`.
- **Public 5-Minute Pipeline Sanity Slice (`BAR-ATH_possession_gt_public.csv`, `7,500` frames):**
  - `macroF1 = 0.8834` (`H 0.9016`, `A 0.8778`, `D 0.8707`, `kappa 0.8215`, `acc 0.8875`), `alive/dead F1 = 0.9686`, `team acc both-alive = 92.04%`.
  - `spellF1 = 0.6400` (`@0.5 IoU = 0.7333`, `@0.7 IoU = 0.4667`, `transF1 @±25f = 0.4444`), up from `0.5875` (`@0.5 = 0.6250`, `transF1 = 0.3243`) for the previous heuristic Viterbi submission and `0.4242` for `baseline_smooth125`.

---

## 6. Reproduction

1. Install pipeline dependencies (and scorer dependencies):
   ```bash
   pip install -r requirements-pipeline.txt
   ```
2. Place the competition data bundle in `./data` (or set `MTAH_DATA_DIR`).
3. *(Optional, pre-trained models ship in `./models/`)* Download Metrica Sample Games 1–3 into `data/external/metrica/` and retrain:
   ```bash
   python src/train_metrica.py
   ```
4. Generate the deterministic submission CSV (also supports `--mode heuristic` for the legacy physics-only baseline):
   ```bash
   python src/run_pipeline.py --out evaluation/results/submission.csv
   ```
5. Validate format and run the public sanity evaluation:
   ```bash
   python evaluation/validate_submission.py evaluation/results/submission.csv
   python evaluation/run_evaluation.py --bootstrap 500
   ```
