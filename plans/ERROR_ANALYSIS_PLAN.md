# Error Analysis & Improvement Plan

## Summary

Analysis of model errors revealed that **9.2% of episodes (>2 hours) account for 55.4% of total error**. The model systematically underpredicts long episodes and struggles when HVAC wasn't running at episode start.

---

## Error Analysis Results

### Overall Performance
- MAE: 31.5 min
- Median AE: 11.4 min
- Bias: -16.3 min (underpredicts)
- P90 error: 62.0 min
- P95 error: 114.5 min
- P99 error: 344.1 min

### Error Percentile Distribution
| Percentile | Error |
|------------|-------|
| P50 | 11.4 min |
| P75 | 26.2 min |
| P90 | 62.0 min |
| P95 | 114.5 min |
| P99 | 344.1 min |

---

## Key Findings

### 1. Long Episodes Dominate Errors

| Duration | MAE | Bias | Count | % of Data |
|----------|-----|------|-------|-----------|
| <15 min | 9.8 | +9.4 | 10,427 | 21.7% |
| 15-30 min | 11.0 | +8.9 | 15,392 | 32.1% |
| 30-60 min | 15.1 | -0.4 | 10,959 | 22.8% |
| 60-120 min | 34.8 | -27.6 | 6,844 | 14.3% |
| 120-240 min | 93.3 | -89.3 | 2,814 | 5.9% |
| **>240 min** | **293.3** | **-291.1** | **1,521** | **3.2%** |

**Episodes >2 hours (9.2% of data) account for 55.4% of total absolute error.**

Worst predictions (actual durations in minutes):
- 7655, 6675, 5820, 5575, 4770, 4365, 4055, 3765, 3380, 3445...

These are clearly anomalous (100+ hours = equipment failure, vacation mode, or data issues).

### 2. System NOT Running = 3.6x Worse

| System State | MAE | Bias | Count |
|--------------|-----|------|-------|
| Running | 19.8 min | -8.5 | 36,984 |
| **Not running** | **70.9 min** | **-42.6** | **11,043** |

When HVAC wasn't active at episode start, predictions are much worse. This suggests:
- Cold-start dynamics differ from steady-state
- May need equipment warm-up time
- Should model this interaction explicitly

### 3. Large Errors (P95+) Are Systematic

2,402 episodes with >P95 error. Breakdown:

**By Duration:**
| Duration | % of Large Errors | % of All Data |
|----------|-------------------|---------------|
| >240 min | 59.5% | 3.2% |
| 120-240 min | 36.2% | 5.9% |
| 60-120 min | 0.7% | 14.3% |
| <60 min | 3.6% | 76.6% |

**By System Running:**
| State | % of Large Errors | % of All Data |
|-------|-------------------|---------------|
| Not running | **64.4%** | 23.0% |
| Running | 35.6% | 77.0% |

**By Change Type:**
| Type | % of Large Errors | % of All Data |
|------|-------------------|---------------|
| heat_increase | 55.0% | 64.2% |
| cool_decrease | **45.0%** | **35.8%** |

Cooling is overrepresented in large errors.

### 4. Calibration Issues

| Predicted Duration | Actual Duration | Bias |
|--------------------|-----------------|------|
| <20 min | 20.6 min | -4.5 |
| 20-40 min | 35.2 min | -5.9 |
| 40-60 min | 61.2 min | -12.5 |
| 60-100 min | 109.1 min | -34.8 |
| 100-200 min | 238.9 min | **-108.9** |
| >200 min | 487.6 min | **-216.5** |

Model is well-calibrated for short predictions but severely underpredicts long episodes.

### 5. Under vs Over Prediction

| Direction | Count | % | Mean Error |
|-----------|-------|---|------------|
| Underpredicted (pred < actual) | 21,013 | 43.8% | -54.7 min |
| Overpredicted (pred > actual) | 27,014 | 56.2% | +13.5 min |

Underpredictions are less frequent but much more severe.

### 6. Per-Home Variation

- Home MAE range: 3.5 - 1275.8 min
- Home MAE median: 24.5 min
- Home MAE P75: 41.7 min

**Hard vs Easy Homes:**

| Feature | Hard Homes (top 10) | Easy Homes (top 10) |
|---------|---------------------|---------------------|
| Avg duration | 291.5 min | 18.4 min |
| Avg gap | 2.9°F | 1.6°F |
| System running | 60% | 100% |
| Outdoor temp | 53.8°F | 64.2°F |

Hard homes have longer episodes, larger gaps, and system often not running.

### 7. Seasonal Pattern

| Month | MAE | Bias |
|-------|-----|------|
| Jan | 24.3 | -9.8 |
| Feb | 24.2 | -10.2 |
| Mar | 26.9 | -11.3 |
| Apr | 34.6 | -19.5 |
| May | 42.7 | -26.1 |
| Jun | 42.0 | -27.5 |
| Jul | 36.9 | -22.2 |
| Aug | 33.9 | -19.5 |
| Sep | 41.0 | -27.5 |
| Oct | 41.3 | -26.9 |
| Nov | 28.6 | -10.6 |
| Dec | 25.4 | -9.1 |

Winter months (heating) are more predictable than shoulder/summer months (cooling).

### 8. Outdoor Temperature Effect

| Outdoor Temp | MAE | Bias |
|--------------|-----|------|
| <32°F | 26.8 | -7.8 |
| 32-50°F | 26.2 | -9.1 |
| 50-70°F | 39.1 | -25.9 |
| 70-85°F | 30.1 | -16.2 |
| >85°F | 35.3 | -23.2 |

Mild weather (50-70°F) is hardest - HVAC cycling behavior is less predictable.

---

## Actionable Improvements

### ✅ Implemented: Filter Extreme Episodes
- Filter episodes >4 hours (240 min) during training and evaluation
- These are likely anomalies and hurt model learning
- **Actual impact: -40% MAE reduction (33.9 → 20.2 min)**

#### Results After Filtering

| Model | Before Filter | After Filter | Improvement |
|-------|--------------|--------------|-------------|
| Baseline (linear) | 40.1 min | 22.4 min | -44% |
| Log-Duration | 35.0 min | 20.5 min | -41% |
| Hierarchical | 37.0 min | 21.9 min | -41% |
| **Hierarchical+Enhanced** | **33.9 min** | **20.2 min** | **-40%** |

**Best model: 20.2 min MAE, 10.8 min median, 48.9 min P90**

### 🔲 TODO: Interaction Features

Add interaction between `system_running` and other features:
```python
# When system NOT running, dynamics are different
"system_running × log_gap"      # Cold start needs more time per degree
"system_running × is_heating"   # Heating cold-start may differ from cooling
```

### 🔲 TODO: Separate Cooling Model

Cooling is systematically harder (45% of large errors vs 36% baseline). Options:
1. Separate model for cooling episodes
2. Add cooling-specific features (humidity interaction, outdoor temp × cooling)
3. Increase regularization for cooling predictions

### 🔲 TODO: Robust Loss / Quantile Regression

Current model minimizes MSE on log-duration, which is sensitive to outliers. Options:
1. Use Huber loss instead of MSE
2. Train quantile regression for P50/P80/P90
3. Winsorize targets before training

### 🔲 TODO: Prediction Capping

For MPC use, cap predictions at reasonable maximum:
```python
pred_duration = min(pred_duration, 240)  # Cap at 4 hours
```

This prevents absurd predictions from affecting MPC planning.

### 🔲 TODO: Two-Stage Model

Stage 1: Classify episode as "normal" (<2 hours) vs "extended" (>2 hours)
Stage 2: Predict duration with stage-specific model

This handles the bimodal distribution better.

### 🔲 TODO: Confidence Intervals

Add prediction intervals for MPC:
- P50: best estimate
- P80: conservative estimate for planning
- P90: safety margin

---

## Priority Order

1. **Filter extremes** (immediate, large impact) ✅
2. **System_running interaction** (medium effort, likely helpful)
3. **Quantile regression** (medium effort, useful for MPC)
4. **Prediction capping** (trivial, safety net)
5. **Separate cooling model** (more effort, unclear benefit)
6. **Two-stage model** (significant effort, unclear benefit)

---

## Scripts

- Error analysis: `scripts/analyze_errors.py`
- Improved baselines: `scripts/run_improved_baselines.py`
- Original baselines: `scripts/run_time_to_target.py`
