# Technical Document — More than a Hack 2027 (branch `crossdomain`)

## 1. Approach

**Learn possession on another provider's matches, prove on a third party's matches that it
transfers, and apply it unchanged to the competition recording.**

Only 17 possessions of the competition match are public, and they may not be used to train or
tune. A method developed against them cannot be told apart from a method that memorised them.
So this pipeline never needs them: every choice that has a number attached to it was made on
labelled matches from **other** games, and the competition match contributes only its unlabelled
tracking.

The pipeline has three stages. A fourth was built and removed by validation (section 5.4).

1. **Features** (`src/features.py`). 256 per-frame features, identical for every data source:
   permutation-invariant (track slots carry no identity), in metres and seconds, mirror-invariant
   (positions enter along the home team's attack axis, inferred per half), missing-aware (ball
   features are NaN while the ball is not tracked) and bidirectional (each signal is summarised
   over windows before and after the frame). Three families:
   - *collective, ball-free*: speed distribution of all tracked players, each team's centroid
     motion along the attack axis, team shape, and restart shapes (a player on the touchline, a
     player at a corner flag, a crowded penalty area);
   - *ball and control*: distance of each team's nearest players to the ball, relative speed of
     ball and nearest player, ball speed, acceleration and turning, distance to the lines;
   - *touches*: the instants the ball's velocity changes sharply, and which team stood closer at
     the previous and at the next one. Possession follows touches — a pass in flight still
     belongs to the passer — and on clean data "team at the last strong touch" alone agrees with
     the possession label on 93% of live frames, against 81% for "nearest player".

2. **External model** (`src/model.py`, `src/train.py`). Two LightGBM heads trained on the 7
   matches of the DFL open dataset:
   - a **dead/alive head that never sees the ball**. It uses the ball-free features only. How a
     provider's ball behaves during a stoppage is the least transferable thing in tracking data:
     the DFL provider freezes it at a point, the Metrica sample drops it, a camera tracker follows
     it in a ball boy's hands. A head that cannot see the ball cannot learn any of those
     conventions;
   - a **team head** (home vs away, given the ball is alive), on all features, trained on live
     frames only.

   Training frames are corrupted to resemble automatic tracking (`src/degrade.py`): long blocks
   without a ball, interpolated short occlusions, a smoothed ball track, slowly drifting player
   positions, missing players, occasional wrong-team tracklets. Every tracking corruption is
   drawn at random, independently of the label. (One step does read the label: the DFL
   provider's frozen dead-ball position is replaced by what a camera tracker produces — no
   ball, a carried ball, a placed ball. It removes an artefact and reaches only the team head's
   windows next to a stoppage.) The corruption levels were set by matching **unlabelled**
   statistics of the competition tracking (`src/calibrate_degradation.py`). At inference the
   model is run on the match and on its team-swapped copy and averaged, so it cannot favour a
   side.

3. **Decoding** (`src/decoder.py`). A three-state Viterbi decode turns frame probabilities into
   possessions, with three penalties (change team, stop play, restart play). A boundary is only
   placed where the accumulated evidence on the far side outweighs its penalty, which is also
   where it gets placed, with no smoothing lag.

The submission path is inference only: features, two model heads, decode. It reads the tactical
ATD and nothing else.

### Why this rather than something else

- **Per-frame rules with smoothing** (the baseline) cannot represent a possession; the ranking
  metric counts possessions.
- **Hand-set physical thresholds** cannot be validated without labelled data, and the only
  labelled data on this match is off limits.
- **Training on the Metrica sample games** (the one public dataset from the organisers' own
  provider) was rejected as the training source for two measured reasons: its ball is synthetic
  (median distance to the nearest player 0.28 m, against 1.91 m in the competition tracking and
  1.42 m in the DFL data), and its ball is absent on 87–100% of dead frames, so any model trained
  on it learns "no ball = dead". Those games are used instead as the **validation set**.
- **Deep sequence models** were not attempted: ten labelled matches and a CPU.
- **Video and the broadcast recording** were not used.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players, ball, team of each slot) | the only input to the external model |
| Smart Tagging | **dropped**: tried as learned side features of an adaptation step that validation removed (5.4) |
| `games.py` half boundaries; FAQ statement that frames 110,808–113,741 are dead | graded range and one forced-dead stretch |
| ATD speed column, homography, both videos, broadcast ATD | not used |
| public 5-minute ground truth | pipeline check only (section 6); nothing fitted to it |

## 3. Handling missing and noisy tracking

- **Ball missing on 41% of graded frames.** Measured on the unlabelled tracking, the ball is
  absent in 150 blocks; 133 of them are longer than 3 s and hold 99.8% of the missing frames, and
  there is no gap between 26 and 75 frames (shorter occlusions arrive already interpolated). The
  model therefore has to label long ball-free stretches from the players alone. Ball features are
  NaN there (the trees route NaN natively) and the ball-free feature family carries the decision.
  Training reproduces the same block structure at random positions.
- **No identities, 46 slots, swaps.** Nothing is per-player. Player speeds above 12 m/s are
  treated as slot swaps and dropped.
- **Ball offset from the carrier.** In the competition tracking the nearest player is a median
  1.9 m from the ball. Training data is corrupted until its distribution brackets that.
- **Tracker glitches.** Single-frame ball teleports and isolated blips are removed
  (`src/atd_io.py`); the delivered file contained none that met the criteria.

## 4. Training data

| dataset | games | labels | role |
|---|---|---|---|
| IDSSE / DFL open dataset (Bassek et al. 2025, Scientific Data; CC-BY 4.0; mirror `huggingface.co/datasets/pysport/idsse-data`) | 7 Bundesliga / 2. Bundesliga matches, season 2022/23, official 25 Hz positions | per-frame `BallStatus` and `BallPossession` in the position files, annotated by the league's data provider | **training** |
| Metrica Sports sample data (`github.com/metrica-sports/sample-data`) | 3 anonymised matches, 25 Hz | human event logs, converted to frame labels deterministically (`src/metrica_io.py`) | **validation only** |

Labels: `D` where `BallStatus = 0`, otherwise `H`/`A` from `BallPossession`. Nobody on this team
labelled anything. **No label from the competition match was used**, public or otherwise, for
training, tuning or model selection.

## 5. Validation on other games

All numbers below are from `src/validate_external.py`, scored with the official
`evaluation/possession_metrics.py` against labels the model never saw.

### 5.1 The submitted model, on a different provider's games

Trained on the 7 DFL matches; scored on the 3 Metrica sample games, which it never saw and which
come from a different provider with a different labelling process. "light" and "heavy" are the
two corruption levels that bracket the competition tracking's unlabelled statistics.

| game | corruption | spellF1 | macroF1 | possessions predicted / true |
|---|---|---|---|---|
| Metrica 1 | light | 0.726 | 0.910 | 326 / 381 |
| Metrica 1 | heavy | 0.602 | 0.895 | 302 / 381 |
| Metrica 2 | light | 0.672 | 0.904 | 281 / 364 |
| Metrica 2 | heavy | 0.613 | 0.886 | 291 / 364 |
| Metrica 3 | light | 0.673 | 0.906 | 329 / 415 |
| Metrica 3 | heavy | 0.574 | 0.882 | 356 / 415 |
| **mean** | **light** | **0.690** | **0.907** | |
| **mean** | **heavy** | **0.596** | **0.888** | |

The dead/alive head never sees the ball, so these numbers do not depend on the Metrica
convention of dropping the ball during stoppages.

### 5.2 Decoder penalties and label timing

The three penalties were set to (change team 6, stop 8, restart 8) before any search. A grid of
54 combinations over the six runs above (before the timing step below) gives a best mean spellF1
of 0.642, the pre-set values 0.639, and the worst corner (all penalties near zero) 0.583. Every
setting with a change-team penalty of 4.5 or more lies within 0.007 of the best. The plateau is
flat, so the pre-set values were kept.

**Label timing.** On the same six runs the model's transitions sit a median of 2 to 5 frames
later than the Metrica labels, depending on the transition type. The model learned its timing
from DFL labels, and the competition's ground truth comes from the Metrica side of that
convention, so every decoded label is moved 3 frames earlier. Moving k frames earlier gives a
mean spellF1 of 0.6386 / 0.6414 / 0.6427 / 0.6432 / 0.6424 / 0.6407 / 0.6376 for k = 0 … 6: a
smooth single peak, and at k = 3 all six runs improve. The table in 5.1 includes this step.

One tracking-only measurement is consistent with it: while a player carries the ball, the
velocities of ball and carrier correlate best at a lag of -2.6 frames in the DFL data (its
player tracks trail the ball) and -0.25 frames in the competition tracking.

### 5.3 Is there a provider gap?

A development model (same design with one head, trained on 6 DFL matches) was scored on the
held-out 7th DFL match and on the Metrica games:

| corruption | held-out DFL match | Metrica mean |
|---|---|---|
| none | 0.794 | 0.778 |
| light | 0.702 | 0.733 |
| heavy | 0.633 | 0.631 |

Same provider or a different one, the score is the same, and that model's predicted transitions
sat a median of one to two frames from the Metrica labels. The DFL and Metrica possession definitions therefore agree
to within what this metric can resolve, which is the basis for expecting the DFL-trained model
to agree with the competition's ground truth too. (This one-head development model was later
replaced; see section 6.)

### 5.4 What validation removed

**Label-free adaptation (`src/adapt.py`) was built, tested and left out.** The idea: let the
external model label the competition match, train a student on the match's own features (ball
and Smart Tagging included) against those pseudo-labels, cross-fitted in time, and average the
two. Run on the six held-out Metrica runs with the procedure it would use on the competition
match:

| game | corruption | external model | + adaptation | change |
|---|---|---|---|---|
| Metrica 1 | light | 0.720 | 0.697 | -0.022 |
| Metrica 1 | heavy | 0.596 | 0.587 | -0.010 |
| Metrica 2 | light | 0.665 | 0.645 | -0.021 |
| Metrica 2 | heavy | 0.610 | 0.598 | -0.012 |
| Metrica 3 | light | 0.669 | 0.658 | -0.011 |
| Metrica 3 | heavy | 0.572 | 0.600 | +0.029 |

It lowered spellF1 on 5 of 6 (mean -0.008) while raising macroF1 slightly: the fused
probabilities are smoother, which helps per-frame accuracy and merges short possessions. The
submission therefore does not use it, and with it goes the Smart Tagging, which only entered
through the adaptation student. The code remains behind `--adapt` as a record of what was tried.
(These figures were measured before the 3-frame timing step of 5.2, hence the small difference
from the table in 5.1.)

**Removing very short predicted possessions was tested and left out.** On the held-out runs
only 33% of predicted possessions under one second and 52% of those between one and two seconds
match a true one, against 87% above eight seconds. Deleting them still costs more than it saves:
mean spellF1 falls from 0.639 to 0.634 with a one-second minimum and to 0.583 with two seconds,
because the true short possessions go with the false ones.

Touch features and dense sampling around label changes were kept because they raised spellF1 on
all 12 development runs (Metrica light 0.691 -> 0.733, heavy 0.609 -> 0.631).

## 6. Check against the public window

The public 5 minutes were used the way EVALUATION.md describes: to check that the pipeline is
correct. One such check changed the design, and it is recorded here because it is the only place
the public window influenced anything.

An earlier version used a single model that could see the ball when deciding dead/alive. It
scored well on the external games, but on the public window it labelled as live the stoppages in
which the ball stays visible. The external games could not have shown this: in the Metrica games
the ball disappears whenever play stops. The fix was structural — the dead/alive head was made
ball-blind (section 1), which removes the dependence on any provider's stoppage convention —
and no parameter was fitted to the window. On the external games the two-head model
scores about 0.04 spellF1 lower than the one-head model did (5.1 against 5.3); most of that is
plausibly what the one-head model had been gaining from the Metrica convention. The window has 17
possessions; the scorer itself reports that differences below about 0.36 on it are noise.

The submitted file on the public window, from `evaluation/run_evaluation.py`:

| | macroF1 | spellF1 | possessions (true 17) | transition F1 at +/-1 s |
|---|---|---|---|---|
| this branch | 0.917 | 0.544 | 19 | 0.558 |
| `baseline_smooth50` (published floor) | 0.621 | 0.326 | 26 | 0.082 |

Per-class F1 is 0.929 / 0.916 / 0.906 for H / A / D. These figures are reported as a pipeline
check and nothing more. Of the 19 predicted possessions, 9 match no true possession at IoU 0.5
(eight of them between 1.8 and 5.1 seconds long): on this window the model's weakness is
inventing short possessions, not missing long ones. The 3-frame timing step of 5.2 moved the
window's spellF1 from 0.556 to 0.544 and its macroF1 from 0.915 to 0.917; it was adopted on the
external evidence regardless.

## 7. Possession rules implemented

There are no hand-written possession rules. What counts as controlled possession, when a loose
ball changes hands and when play is dead are learned from the DFL provider's per-frame labels.
What the code fixes is structure:

- three states only; a possession is a maximal run of one team with the ball alive;
- possession follows touches (the touch features), so a pass in flight stays with the passer;
- dead/alive is decided from collective player behaviour alone (speeds, centroid motion,
  restart shapes); the ball does not enter that decision;
- frames outside the two halves are `X`; inside them every frame gets `H`, `A` or `D`;
- the graded stretch before the second-half kick-off is `D` (FAQ.md).

## 8. Limitations

- Possessions shorter than two seconds are the main loss. On the validation games about a
  quarter of them are recovered, against roughly nine in ten of those longer than eight seconds.
- The dead/alive head is ball-blind by design, so a quick restart during which the players never
  slow down is hard to see, and the start and end of a stoppage are placed from the players'
  behaviour rather than from the ball crossing a line.
- The dead/alive head depends on player speeds computed from positions, so it depends on the
  tracker not flickering. With frame-to-frame position jitter of 0.55 m added to a held-out
  match (every player then appears to move at 5-7 m/s) it stops detecting stoppages: alive/dead
  F1 falls from 0.95 to 0.74 and macroF1 to 0.45. The competition tracking shows no such jitter
  (median collective speed 1.47 m/s, the same as clean tracking), but a tracker that flickers
  would need its positions smoothed first.
- The Smart Tagging is not used. It cannot be validated on other games because no other game
  has it, and the one validated route to it (adaptation) did not pay.
- The corruption model is calibrated on unlabelled statistics and brackets the competition
  tracking between two levels. The validation score moves by about 0.09 between them, so the
  expected score on the competition match is correspondingly uncertain.
- Developed on an 8 GB laptop; training uses every fourth-to-eighth frame for that reason.

## 9. Reproduction

```bash
pip install -r requirements-pipeline.txt
# place the competition bundle in ./data (or set MTAH_DATA_DIR)
python src/run_pipeline.py --out evaluation/results/submission.csv
python evaluation/validate_submission.py evaluation/results/submission.csv
```

The trained external model ships in `models/`. To rebuild it and re-run the validation:

```bash
python src/download_external.py          # ~3 GB of public data about other games
python src/train.py                      # -> models/
python src/validate_external.py --settings light heavy --grid --adapt light heavy
python src/calibrate_degradation.py      # unlabelled statistics behind the corruption settings
```
