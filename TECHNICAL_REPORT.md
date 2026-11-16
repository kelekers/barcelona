# Technical Document — More than a Hack 2027 (Phase 1: Ball Possession)

## 1. Approach

We frame frame-level ball possession inference as a **3-state temporal sequence decoding problem** (`H` = Home / Barcelona, `A` = Away / Athletic Club, `D` = Dead Ball) solved via **time-varying, outcome-gated max-sum Viterbi decoding** on top of physically calibrated kinematic and tactical features.

Why a physics-driven feature model + time-varying Viterbi decoder rather than per-frame classification or supervised training:
1. **The ranking metric (`spellF1`) measures interval-level possession recovery**, not isolated per-frame accuracy. Per-frame nearest-player heuristics flicker rapidly during passes, aerial duels, and tracking noise, fragmenting a single match into 600–900+ micro-possessions. Time-varying Viterbi decoding enforces temporal persistence through state-transition penalties modulated by physical ball-touch impulses (`H <-> A`) while placing changepoints directly where cumulative evidence shifts, avoiding the boundary blurring of symmetric majority-vote smoothing.
2. **Compliance and zero overfitting:** Only the first 5 minutes (17 possessions) are publicly labelled, and competition rules prohibit using them for training or parameter tuning as well as any hand-labelling on the match. All physical thresholds (speeds in `m/s`, distances in metres, durations in seconds) and state penalties are derived from football kinematics and full-match unlabeled data properties rather than fitted to the 5-minute public ground truth.
3. **Speed and reproducibility:** The entire pipeline relies solely on `numpy` and `pandas` (matching `requirements.txt`), executes deterministically in ~7 seconds across all 199,319 frames, and produces byte-identical output across runs.

---

## 2. Assets Used and Dropped

### Used
- **Tactical Automatic Tracking Data (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`):** Primary signal for ball coordinates `(x, y)`, team-level player coordinates `(x, y)`, and player speeds `s` (`m/s`) across all 199,319 frames, plus team mapping (`Barcelona` -> `H`, `Atletic Club` -> `A`).
- **Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`):** Used strictly as a **soft prior and event anchor** (never copied as labels), specifically:
  - `GOALS` paired with `Kick Off (After Goal)` to identify post-goal celebration stoppages and the conceding team taking the subsequent kick-off.
  - `SET PIECES` (`Corner Kick`, `Goal Kick`, `Free Kick`, `Throw-in`) to pinpoint exact restart timestamps (`D -> H/A`) and restarting team priors.
  - Offensive phase codes (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `LONG BALL`) as weak tie-breakers (stronger during live-play ball `NaN` occlusions) whenever only one team is tagged.
- **Match Timing Metadata (`games.py`):** Half video blocks (`11185..90651`, `110808..190197`) and second-half kick-off boundary (`113742`), fixing ungraded frames to `X` and pre-kickoff second-half frames (`110808..113741`) to `D`.

### Evaluated and Dropped
- **Provided ATD `ball_s` column:** Available on only `9,909` of `113,488` visible ball frames; replaced by centered finite-difference velocity computed directly from cleaned `(ball_x, ball_y)` in metres.
- **Tactical & Broadcast Videos / Broadcast ATD (`BAR-ATH_tactical.mp4`, `BAR-ATH_broadcast*`):** Audited during EDA. Because `85%+` of missing-ball frames in live halves occur during stoppages/celebrations/set-pieces and only 17 live-play `NaN` gaps `> 1s` span a predicted state change across the full match, cross-timeline broadcast synchronization (with 34.2% invalid homography frames, cuts, and replays) or video-based ball detection was omitted in favor of a fast, self-contained, deterministic tracking + tagging pipeline.

---

## 3. Handling Missing and Noisy Tracking Data

1. **Ball Glitch & Outlier Filtering (`src/atd_io.py`):**
   - Raw consecutive ball displacements reach up to `134 m/s` due to 1–2 frame tracker snaps. We remove 1–2 frame spikes jumping `> 2.2 m/frame` (`55 m/s`) that immediately snap back, plus isolated `<= 4`-frame islands separated by `>= 25` `NaN` frames and `> 12 m` away from adjacent valid ball trajectories.
2. **Short Live-Play Interpolation vs. Long-Gap Classification (`src/features.py`):**
   - Short gaps (`<= 35` frames / `1.4 s`) are linearly interpolated for smooth proximity and velocity estimates and biased toward staying alive (`H`/`A`).
   - Long gaps (`> 35` frames) are classified using a **two-tier collective activity & reset profile** (`mean` and `90th percentile` player speed across the pitch, plus pitch-boundary proximity):
     - **Live-play occlusions:** Gaps with active collective speeds (`mean >= 1.92 m/s`, `p90 >= 2.75 m/s`) and no boundary exit stay alive (`dead_score < 0`), preventing false `D` predictions.
     - **Tier-1 stoppages:** Gaps with slow collective speed (`mean < 1.50 m/s`) or pitch-boundary exits are marked dead (`dead_score > 0`).
     - **Tier-2 jogging resets (`>= 65` frames, `1.50 <= mean < 1.92 m/s`):** Recover untagged fouls, offsides, and slow set-piece resets where players jog back into shape whenever at least two of three stoppage signatures hold: (i) no chasing sprints (`p90 < 3.1 m/s`), (ii) second-half deceleration (`mps_second_half < mps_first_half` and `< 1.60 m/s`), or (iii) post-gap stationary ball (`b_spd[b+2:b+18] < 1.5 m/s`) or out-of-bounds re-entry.
3. **Player Tracklet Slot Switching:**
   - Because the 46 track slots swap identities, no per-player history or shirt-number tracking is used. All player features are permutation-invariant per-team minimums (`d_H`, `d_A`) and collective pitch speed quantiles (`mean_ps`, `p90_ps`).

---

## 4. Training Data

**None.** No external labelled match datasets were used for supervised training, and no labels from the provided `BAR-ATH` match (including the 5-minute public ground truth) were used to train or fit parameters. All parameters represent physical football quantities and structural priors.

---

## 5. Possession Rules Implemented

1. **Alive vs. Dead Ball (`D`):**
   - **Out-of-bounds (`oob`) & touchline setups:** Visible ball positions outside `[0, 1]` (with a `0.004` tolerance for boundary jitter) and stationary balls held near sidelines/endlines strongly favour `D`.
   - **Asymmetric collective activity & uncontested slow ball:** Low collective player speed (`mean_ps < 1.30 m/s`, `p90_ps < 2.00 m/s`) provides strong `D` evidence, while higher speeds only gently favour Alive so players jogging back into shape during a Goal Kick or Corner setup do not erase `D`. Out-of-bounds evidence uses a tight `±3`-frame window so stoppage boundaries remain sharp. To detect non-set-piece stoppages promptly without waiting for full collective standstill, we combine an **uncontested slow ball** detector (`b_spd < 2.3 m/s`, `min_d > 3.5 m`, `mean_ps < 1.38 m/s`) with **deceleration inflection detection** (`b_spd < 4.5 m/s`, `min_d > 5.5 m` while collective speed drops `< 1.32 m/s` within `1 s`).
   - **Build-up pause protection:** Short (`< 4.2 s`) visible in-bounds pauses without a `SET PIECES` tag and short (`< 1.4 s`) `D` blips are bridged back to the surrounding team possession so slow build-up pauses at a defender's feet do not split possessions.
   - **Goal celebrations & Set Pieces:** Post-goal celebration tails ending at `Kick Off (After Goal)` are marked `D` up to the kick-off instant (`~83` frames after tag start). For non-kickoff `SET PIECES`, the exact acceleration onset of in-bounds ball motion (`b_spd >= 3.4 m/s` or `> 2.5 m` displacement with sharp pre-to-post speed step) marks the restart (`D -> team`), and any preceding out-of-bounds exit within 14s is bridged continuously as `D` until the restart.
   - **Throw-in team inference (`Team = N/A`):** All 26 throw-in tags carry `Team = N/A`; the restarting team is inferred by voting on players positioned near the touchline (`y < 0.035` or `y > 0.965`) and closest to the ball around the restart window.
2. **Controlled Possession vs. Passes, Deflections, and Contested Balls (`H` vs. `A`):**
   - **Ball-in-flight attenuation:** Frame-level team evidence decays exponentially with nearest-player distance (`exp(-min_d / 1.7)`) and quadratic ball speed above a dribble/flight threshold (`9.5 m/s` when dribbling within `1.6 m`, `6.0 m/s` when farther away), so passes or clearances flying past defenders do not flip possession.
   - **Causal close-control confirmation:** Sustained proximity (`d < 2.1 m`, `b_spd < 9.5 m/s`) over an asymmetric causal trailing window (`[t - 14, t + 1]`) reinforces controlled possession without pulling transitions prematurely before the receiving player reaches the ball.
   - **Contested-ball damping:** When players from both teams are close to the ball at comparable distances (`opp_d` small and `|d_A - d_H|` small), team evidence is damped so 50/50 duels stay with the preceding possessor until one team gains clear control.
   - **Outcome-gated time-varying transitions, mid-flight pass damping & boundary snapping:** Ball velocity-vector impulses (`|Δv_b| > 3.5 m/s` with a player within `2.8 m`) are validated over a post-touch outcome window (`[+5, +65]` frames: sustained own-team control, `> 12 m` clearance displacement, or dead-ball exit). Confirmed touch impulses and set-piece restarts lower the `H <-> A` transition penalty (`0.78x`), while fast passes flying past defenders (`b_spd > 7.0 m/s`, `min_d > 1.35 m`, turning angle `< 35°`, `|Δv_b| <= 4.5 m/s`) raise the turnover penalty (`1.8x`). After Viterbi decoding and micro-segment cleanup (`< 0.72 s`), a **Boundary Snap-to-Impulse** post-processor (`[-8, +16]` frames, guarded against colliding segments) shifts `H <-> A` turnovers to the local physical touch-impulse peak and `H/A -> D` exits to the exact out-of-bounds crossing frame, followed by a **high-speed sandwich filter** absorbing brief `< 2.2 s` undeflected (`< 30°`) fly-by segments (`median b_spd > 7.0 m/s`). (Asymmetric forward/backward `mps_slow` windows, a strict 5-state `ctrl`/`flight` topology prohibiting flight transitions, RTS Kalman smoothing on short gaps, and Hungarian tracklet stitching were evaluated in ablation and omitted because they reduced boundary accuracy or over-merged valid transitions.)
3. **Abstentions (`X`):**
   - Ungraded frames (`< 11185`, `90652..110807`, `>= 190198`) are labelled `X`. Inside graded play blocks, the Viterbi decoder outputs explicit `H`, `A`, or `D` on every frame.

---

## 6. Limitations

- **Single-touch clearances & fast counter-press duels:** Without pose/foot-contact detection from video, a rapid 1–2 second midfield scramble where both teams touch the ball briefly remains ambiguous in ATD alone.
- **Throw-in team attribution edge cases:** When both teams crowd the touchline without clear ball-to-thrower separation before the throw, the spatial vote can occasionally favour the wrong side at restart until the first in-bounds reception corrects it.
- **Smart Tagging errors:** Because Smart Tagging is automatic, rare missed set pieces without collective slowdown or false-positive tags rely purely on the underlying ATD kinematics.
- **Potential next steps with more time:** Synchronizing `broadcast` ATD (during `valid == 1` stretches) via player point-cloud matching to recover the 17 live-play `NaN` gaps that span state changes, and running a lightweight cropped ball detector on `BAR-ATH_tactical.mp4` specifically around ambiguous set-piece restarts and box scrambles.

---

## 7. Reproduction

1. Install minimal dependencies (Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```
2. Place the competition data bundle in `./data` (or set `MTAH_DATA_DIR`).
3. Generate the deterministic submission CSV:
   ```bash
   python src/run_pipeline.py --out evaluation/results/submission.csv
   ```
4. Validate format and inspect summary statistics:
   ```bash
   python evaluation/validate_submission.py evaluation/results/submission.csv
   ```
5. Check against the public 5-minute pipeline sanity slice:
   ```bash
   python evaluation/run_evaluation.py --bootstrap 1000
   ```
