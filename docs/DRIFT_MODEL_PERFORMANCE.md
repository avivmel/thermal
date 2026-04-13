# Drift Model Performance

Evaluation of the passive thermal drift time-to-boundary model (`models/drift_time_xgb.pkl`).
Predicts how many minutes before indoor temperature passively drifts from the current value
to a comfort boundary, with HVAC off.

---

## Model Summary

- **Architecture**: Gradient Boosted Trees (sklearn)
- **Training rows** (cross-home): 274,899
- **Features**: 14 (margin, log_margin, thermal drive, time/season encoding, temps)
- **Target**: log(minutes to boundary) → exponentiated and clipped to [5, 480] min
- **Per-home homes covered**: 679
- **Per-home×mode slots covered**: 1,304
- **Residual shrinkage N**: 10

---

## Evaluation Setup

Two complementary evaluations are reported:

### 1. Cross-Home (train → test homes)

Uses the project's canonical home-level split (`split` column in `drift_episodes.parquet`).
Train homes: 679 · Val homes: 96 · Test homes: 195.
Test homes are **completely unseen** during training — this measures true generalization.
Per-home correction cannot apply to test homes (no residual history).

### 2. Within-Home (70/30 per home)

Each home's episodes are split 70/30 randomly. The same home appears in both train and test.
This allows the per-home residual correction to be applied on the 30% test portion.
Shows the best-case benefit of per-home correction for a **known** home.

---

## Results: Cross-Home Evaluation

| Model                                   |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| Direction Median                       |   83,414 |   40.7 |    5.0 |  115.0 |   81.5 |  -22.3 |
| Global Linear Rate                     |   83,414 |   36.3 |   15.5 |   89.6 |   61.6 |   +3.2 |
| Enhanced Ridge (global)                |   83,414 |   30.0 |    8.5 |   85.0 |   60.0 |  -12.3 |
| Global GBM (cross-home, test homes)    |   83,414 |   27.1 |    7.2 |   77.9 |   54.2 |   -8.8 |
| Global GBM (linear target)             |   83,414 |   28.7 |    9.6 |   79.4 |   52.5 |   +1.7 |

Per-home and home-encoded methods are omitted here when they require prior home history.
Test homes are unseen, so these results are the cold-start benchmark set.

---

## Results: Within-Home Evaluation

| Model                                   |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| Direction Median                       |   82,781 |   44.8 |    5.0 |  140.0 |   87.8 |  -26.7 |
| Global Linear Rate                     |   82,781 |   38.0 |   15.6 |   95.7 |   64.4 |   +0.7 |
| Per-Home Linear Rate                   |   82,781 |   34.0 |   16.8 |   86.4 |   58.7 |   +1.4 |
| Log-Duration Per-Home                  |   82,781 |   29.7 |    7.5 |   85.5 |   60.7 |  -12.0 |
| Enhanced Ridge + Per-Home              |   82,781 |   28.8 |    8.4 |   82.2 |   58.3 |  -13.0 |
| Global GBM (within-home)               |   82,781 |   28.1 |    7.4 |   81.5 |   55.5 |  -11.1 |
| Global GBM (linear target)             |   82,781 |   29.1 |    9.7 |   81.5 |   52.6 |   -0.2 |
| Global GBM + Home Encoding             |   82,781 |   25.5 |    7.1 |   72.6 |   51.5 |  -10.0 |
| Per-Home GBM                           |   82,781 |   23.1 |    6.3 |   64.3 |   48.6 |   -8.1 |
| Global + Per-Home Correction           |   82,781 |   25.7 |    7.3 |   72.8 |   52.1 |  -10.3 |

---

## Stratified Breakdown (Cross-Home, Global Model)

### By Direction

| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| warming_drift                          |   26,637 |   56.5 |   32.3 |  141.4 |   85.0 |  -18.0 |
| cooling_drift                          |   56,777 |   13.4 |    5.2 |   30.9 |   30.4 |   -4.5 |

### By Margin (distance from boundary at episode start)

| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| 1–2°F                                  |   57,386 |   13.1 |    5.2 |   27.4 |   32.4 |   -5.0 |
| 2–3°F                                  |   13,108 |   43.2 |   22.8 |  105.7 |   69.5 |  -13.9 |
| 3–5°F                                  |    9,315 |   71.8 |   53.9 |  160.0 |   95.8 |  -20.7 |
| >5°F                                   |    3,430 |   79.3 |   64.7 |  164.3 |  102.1 |  -22.4 |

### By True Duration

| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| ≤30 min                                |   53,420 |    9.2 |    4.9 |   17.3 |   21.3 |   +7.3 |
| 30–120 min                             |   17,855 |   32.4 |   23.9 |   67.7 |   43.9 |   -6.4 |
| >120 min                               |   12,139 |   98.1 |   83.3 |  201.6 |  124.0 |  -83.3 |

---

## Stratified Breakdown (Within-Home, Global + Per-Home)

### By Direction

| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| warming_drift                          |   27,905 |   50.8 |   27.8 |  131.0 |   79.2 |  -21.2 |
| cooling_drift                          |   54,876 |   12.9 |    4.7 |   29.9 |   29.9 |   -4.8 |

### By Margin

| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| 1–2°F                                  |   55,739 |   12.7 |    4.7 |   26.2 |   32.4 |   -5.1 |
| 2–3°F                                  |   12,379 |   41.4 |   20.8 |  108.4 |   68.5 |  -18.9 |
| 3–5°F                                  |   10,204 |   59.0 |   40.5 |  138.8 |   83.6 |  -21.4 |
| >5°F                                   |    4,244 |   70.5 |   55.1 |  154.5 |   91.6 |  -27.3 |

### By True Duration

| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |
|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| ≤30 min                                |   51,016 |    7.2 |    4.2 |   13.8 |   14.3 |   +5.1 |
| 30–120 min                             |   18,088 |   28.6 |   21.2 |   60.7 |   38.7 |   -6.9 |
| >120 min                               |   13,677 |   90.9 |   73.8 |  190.5 |  116.9 |  -72.4 |

---

## Interpretation

### Benchmark Context

The drift evaluator now mirrors the active-time benchmark ladder: simple direction medians,
linear rate models, log-duration linear models, enhanced Ridge models, global GBM variants,
per-home GBMs, and the hybrid global + per-home residual model. Cross-home evaluation only
uses methods that can operate for unseen homes without prior history; within-home evaluation
adds the per-home and home-encoded variants.

In the cold-start cross-home setup, the global linear rate baseline is 36.3 min MAE,
the enhanced Ridge model is 30.0 min MAE, and the global GBM remains best at
27.1 min MAE.

In the known-home setup, home information matters: home encoding improves the global GBM
from 28.1 to 25.5 min MAE, per-home GBM reaches 23.1 min MAE, and the deployed-style
hybrid residual correction reaches 25.7 min MAE.

### Cross-Home vs. Within-Home Gap

The cross-home MAE measures how well the model generalizes to a brand-new home with no
prior history. The within-home MAE (global) measures performance when the model has seen
other episodes from the same home. The difference between these two numbers is the
"cold-start penalty" — the cost of not having any home-specific data.

### Per-Home Correction Benefit

Comparing within-home global vs. within-home per-home corrected shows how much per-home
residuals improve predictions for **known** homes. The correction is applied in log-space
with shrinkage (N=10): homes with fewer episodes get corrections pulled toward 0.

### Margin Is the Primary Driver

Drift time scales almost linearly with margin (°F from boundary). The `log_margin` feature
captures this. Predictions for small margins (1–2°F) are hardest: a 1°F error in start_temp
due to Ecobee's integer quantization can shift the margin by 100% and the predicted time
substantially.

### Direction Asymmetry

Warming drift (heating season) tends to have longer times to boundary — homes in winter
are pre-heated to near the upper comfort bound before peak, so the boundary (lower comfort
bound) is often 4°F away. Cooling drift has smaller margins on average. Expect MAE to
differ between the two.

---

## Artifact Details

**Path**: `models/drift_time_xgb.pkl`

| Field | Value |
|-------|-------|
| `version` | 2 |
| `model_type` | `gbm_drift_time_to_boundary` |
| `backend` | `sklearn` |
| `n_train_rows` | 274,899 |
| `n_homes_with_correction` | 679 |
| `n_home_modes_with_correction` | 1,304 |
| `min_duration` | 5.0 min |
| `max_duration` | 480.0 min |

The artifact now includes `home_residuals` and `home_mode_residuals` dicts,
matching the structure of the active-time artifact. `predict_drift_time()` in
`mpc/model_interfaces.py` applies per-home correction when `home_id` is provided
and the home is in the training set.
