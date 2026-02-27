# Time-to-Target Prediction

## Task Definition

**Predict how many minutes until indoor temperature reaches the target setpoint.**

This reframes the prediction problem to sidestep integer quantization issues. Instead of predicting temperature values (which are 66% unchanged due to integer resolution), we predict duration - a continuous value that's directly useful for MPC.

| Attribute | Value |
|-----------|-------|
| Dataset | `data/setpoint_responses.parquet` |
| Input | Current state at any point in episode |
| Output | Minutes remaining until target reached |
| Metric | MAE (minutes) |

---

## Why This Framing?

### MPC Use Case

```
Question: "I need the house at 72°F by 5pm for DR event.
          It's 70°F now. When should I start?"

Time-to-target: "This home takes ~40 min for +2°F"
→ Start at 4:20pm ✓

Temp prediction: "In 15 min, temp will be 70°F (probably)"
→ Not actionable ✗
```

### Avoids Quantization Problem

- Temperature changes are integers → 66% show 0 change
- Time-to-target is continuous → no quantization floor
- Prediction error is meaningful in minutes, not masked by rounding

---

## Input Features

At prediction time (any point during episode):

| Feature | Description |
|---------|-------------|
| `current_temp` | Indoor temperature now (°F) |
| `target_setpoint` | Target to reach (°F) |
| `current_gap` | target - current (°F, signed) |
| `outdoor_temp` | Outdoor temperature (°F) |
| `thermal_drive` | outdoor - indoor (°F) |
| `elapsed_time` | Time since episode start (min) |
| `change_type` | heat_increase or cool_decrease |
| `home_id` | For per-home models |

## Target Variable

```
time_remaining = episode_duration - elapsed_time  (minutes)
```

---

## Baselines

### B1: Global Mean
Predict the mean time-to-target across all training episodes.
```
y_pred = mean(episode_durations)
```

### B2: Gap-Proportional
Predict time proportional to remaining gap.
```
y_pred = k * |current_gap|
```
Learn `k` (minutes per °F) from training data.

### B3: Gap + Thermal Drive
Account for whether outdoor temp helps or hurts.
```
y_pred = k1 * |current_gap| + k2 * thermal_drive
```

### B4: Per-Home Gap
Learn a separate `k` for each home.
```
y_pred = k[home_id] * |current_gap|
```

### B5: Per-Home Linear
Full linear model per home.
```
y_pred = model[home_id].predict(features)
```

### B6: Mode-Specific
Separate models for heating vs cooling.
```
if heat_increase:
    y_pred = k_heat * |current_gap|
else:
    y_pred = k_cool * |current_gap|
```

---

## Evaluation Strategy

### Within-Home Split
For each home, 70% of episodes to train, 30% to test. This allows per-home models to learn home-specific thermal dynamics.

### Metrics
- **MAE**: Mean absolute error in minutes
- **RMSE**: Root mean squared error
- **MAPE**: Mean absolute percentage error (relative to true duration)

### Stratification
Report results by:
1. **Change type**: heat_increase vs cool_decrease
2. **Initial gap**: How far from target at episode start
3. **Episode stage**: Early/mid/late predictions within episode

---

---

## Experimental Results

### Baseline Models (Linear Scale)

| Model | Cross-Home MAE | Within-Home MAE | Improvement |
|-------|----------------|-----------------|-------------|
| Global Mean | 51.6 min | 50.9 min | +1% |
| Gap Proportional | 47.1 min | 46.1 min | +2% |
| Mode-Specific | 47.2 min | 46.1 min | +2% |
| **Per-Home Gap** | N/A | **43.0 min** | - |
| **Per-Home + Mode** | N/A | **40.1 min** | **-14%** |

**Per-home tuning reduces error by 14%** compared to global models.

---

### Improved Models (Log-Duration + Hierarchical)

| Model | MAE | RMSE | Median | P90 | Improvement |
|-------|-----|------|--------|-----|-------------|
| Baseline (linear) | 40.1 | 146.7 | 17.0 | 83.5 | - |
| Log-Duration | 35.0 | 152.1 | 12.4 | 66.7 | **+12.6%** |
| Hierarchical | 37.0 | 151.8 | 14.0 | 68.0 | +7.5% |
| **Hierarchical+Enhanced** | **33.9** | **146.5** | **12.4** | **63.6** | **+15.5%** |

**Log-transform + enhanced features reduces MAE by 15.5%** (40.1 → 33.9 min).

### Best Model: Hierarchical + Enhanced Features

| Metric | Value |
|--------|-------|
| **MAE** | **33.9 min** |
| Median Error | 12.4 min |
| P90 Error | 63.6 min |

### Key Learned Coefficients

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `log_gap` | +0.82 | Duration scales ~linearly with gap |
| `is_heating` | -0.38 | Heating is **32% faster** than cooling |
| `system_running` | **-0.88** | Already-running HVAC is **58% faster** |
| `signed_thermal_drive` | -0.009 | Small outdoor temp effect |
| `month_cos` | -0.07 | Slight seasonal effect |

The `system_running` feature is the biggest discovery: episodes where HVAC was already active at setpoint change complete much faster.

### By Initial Gap

| Gap | Baseline MAE | Enhanced MAE | Improvement |
|-----|--------------|--------------|-------------|
| 1-2°F | 25.6 min | 22.7 min | -11% |
| 2-3°F | 39.5 min | 31.6 min | -20% |
| 3-5°F | 59.3 min | 49.1 min | -17% |
| >5°F | 87.1 min | 77.4 min | -11% |

### By Change Type

| Type | Baseline MAE | Enhanced MAE |
|------|--------------|--------------|
| Heating | 38.0 min | 31.5 min |
| Cooling | 43.7 min | 38.1 min |

### Learned Parameters

- **Global k**: 27 min/°F average
- **Per-home k range**: 6-120 min/°F (huge variation!)
- **Random effects std**: 0.30 (homes vary ±30% around global)

---

### After Filtering Extreme Episodes (≤4 hours)

Filtering episodes >4 hours (equipment failures, vacation mode) reduces MAE by ~40%.

| Model | MAE | Median | P90 | Improvement |
|-------|-----|--------|-----|-------------|
| Baseline (linear) | 22.4 min | 13.1 min | 51.9 min | - |
| Log-Duration | 20.5 min | 11.0 min | 48.9 min | -8.6% |
| Hierarchical+Enhanced | 20.2 min | 10.8 min | 48.9 min | -9.7% |

---

### Gradient Boosting Models

| Model | MAE | Median | P90 | Improvement |
|-------|-----|--------|-----|-------------|
| Global GBM (no home info) | 20.8 min | 11.0 min | 50.9 min | -7.1% |
| Global GBM + Home Encoding | 18.8 min | 9.6 min | 46.4 min | -16.1% |
| Per-Home GBM | 19.0 min | 9.2 min | 48.7 min | -15.2% |
| **Hybrid GBM** | **18.7 min** | **9.5 min** | **46.6 min** | **-16.5%** |

**Key insight**: Home target encoding (encoding home_id as mean log-duration with shrinkage) is more effective than training separate per-home models.

---

## Key Findings

1. **Filter anomalies** - Episodes >4 hours are equipment failures/vacation mode; filtering reduces MAE by 40%

2. **Log-transform is essential** - handles 20x home variation naturally

3. **System already running is highly predictive** - if HVAC was active at setpoint change, episode completes 50% faster

4. **Heating faster than cooling** - 28-32% faster

5. **Hierarchical partial pooling works** - homes vary ±30% around global mean

6. **Home target encoding > per-home models** - global GBM + encoding beats separate per-home GBMs

7. **Hybrid is best** - global model + per-home residual correction achieves 18.7 min MAE

8. **Cold-start still hard** - system not running: ~30 min MAE vs ~16 min when running

---

## Scripts

```bash
# Original baselines
python scripts/run_time_to_target.py

# Improved models (log-duration, hierarchical, enhanced features)
python scripts/run_improved_baselines.py

# Gradient boosting models
python scripts/run_xgboost_baselines.py
```
