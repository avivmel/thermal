# Deep Learning for HVAC Time-to-Target Prediction

## Problem Statement

We're building a smart thermostat system for residential demand response. The key planning question:

**"If I change the setpoint now, how long until the home reaches the target temperature?"**

Accurate prediction enables optimal scheduling of HVAC pre-conditioning before peak electricity hours.

---

## Current Approach & Results

### Best Model: Hybrid Gradient Boosting

Global GBM with per-home residual correction (target encoding + shrinkage).

| Metric | Value |
|--------|-------|
| MAE | 18.7 min |
| Median Error | 9.5 min |
| P90 Error | 46.6 min |
| Episodes | 46K test |

### Model Comparison

| Model | MAE | Median | P90 |
|-------|-----|--------|-----|
| Per-Home Linear | 22.4 min | 13.1 min | 51.9 min |
| Hierarchical Linear | 20.2 min | 10.8 min | 48.9 min |
| Global GBM (no home) | 20.8 min | 11.0 min | 50.9 min |
| GBM + Home Encoding | 18.8 min | 9.6 min | 46.4 min |
| **Hybrid GBM** | **18.7 min** | **9.5 min** | **46.6 min** |

### Learned Feature Importance (Linear Model)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `log_gap` | +0.72 | Larger gap → longer duration |
| `is_heating` | -0.33 | Heating 28% faster than cooling |
| `system_running` | -0.69 | Already-on HVAC = 50% faster |
| `thermal_drive` | -0.009 | Small outdoor temp effect |
| `month_cos` | -0.05 | Slight seasonality |

### What We've Tried

1. **Linear per-home models** - 22.4 min MAE
2. **Log-transform** - Handles 20x home variation, +8.6% improvement
3. **Hierarchical/mixed effects** - Partial pooling across homes
4. **Enhanced features** - Runtime, seasonality, humidity
5. **Filtering anomalies** - Removed >4h episodes (40% MAE reduction)
6. **Gradient boosting** - Global + home target encoding achieves 18.7 min MAE

---

## Dataset

**Source:** Ecobee Donate Your Data - 971 US homes, full year 2017

### Episode Structure

Each episode = setpoint change → temperature reaches target

| Attribute | Value |
|-----------|-------|
| Total episodes | 158K |
| Episodes ≤4h (filtered) | 145K |
| Train/Test split | Within-home (70/30 episodes per home) |
| Resolution | 5-minute intervals |

### Features Available at Prediction Time (episode start)

| Feature | Description |
|---------|-------------|
| `initial_gap` | Target setpoint - current temp (°F) |
| `outdoor_temp` | Outdoor temperature (°F) |
| `indoor_humidity` | Indoor humidity (%) |
| `outdoor_humidity` | Outdoor humidity (%) |
| `heat_runtime_start` | Heating runtime in current 5-min interval (sec) |
| `cool_runtime_start` | Cooling runtime in current 5-min interval (sec) |
| `hour`, `month` | Timestamp features |
| `state` | US state (CA, TX, IL, NY) |
| `home_id` | Anonymized home identifier |
| `change_type` | `heat_increase` or `cool_decrease` |

### Sequence Data Available

For each episode, we have the full 5-minute resolution trajectory:
- Indoor temperature (quantized to integers)
- Outdoor temperature
- HVAC runtime (heating/cooling/fan seconds per 5-min)
- Humidity

This could enable:
- Online prediction updates after first 10-15 minutes
- Learning from temperature trajectory shape
- Detecting anomalous episodes early

---

## Key Challenges

### 1. Extreme Home-to-Home Variation

The rate parameter `k` (minutes per °F) varies **20x** across homes:

| Statistic | k (min/°F) |
|-----------|------------|
| Fastest home | 6 |
| Median | 21 |
| Slowest home | 120 |

This reflects real physical differences:
- HVAC system power (BTU capacity)
- Home size and thermal mass
- Insulation quality
- Climate zone

**Question:** How should we model this? Home embeddings? Meta-learning? Hierarchical priors?

### 2. Limited Data Per Home

| Statistic | Value |
|-----------|-------|
| Episodes per home | 50-200 |
| Median | ~100 |

Not enough for complex per-home models, but enough to estimate home-specific parameters.

**Question:** How to balance global knowledge with per-home adaptation?

### 3. Cold Start for New Homes

At deployment, we'll encounter homes with zero history. Need to:
1. Start with reasonable global prediction
2. Rapidly adapt as episodes accumulate
3. Transfer knowledge from similar homes

**Question:** Few-shot learning? Online adaptation? Home clustering?

### 4. Heavy-Tailed Duration Distribution

| Duration | % of Episodes |
|----------|---------------|
| <30 min | 54% |
| 30-60 min | 23% |
| 1-2 hours | 14% |
| 2-4 hours | 6% |
| >4 hours | 3% (filtered as anomalies) |

The distribution is highly skewed. Current model underpredicts long episodes.

**Question:** Robust losses? Mixture models? Separate short/long predictors?

### 5. Two Distinct Regimes

**System already running** (77% of episodes):
- MAE: 19.8 min
- Predictable, HVAC at steady state

**Cold start** (23% of episodes):
- MAE: 70.9 min (3.6x worse!)
- Equipment warm-up, uncertain dynamics

**Question:** Separate models? Regime detection? Different architectures?

### 6. Sequence vs Point Prediction

Current model predicts at episode start only. But we could:
1. Update prediction after observing first 10-15 min
2. Model the full temperature trajectory
3. Use sequence models (LSTM, Transformer) on historical episodes

**Question:** What's the right architecture for sequence-aware prediction?

---

## Specific Questions for Deep Learning Approach

### Architecture

1. **Home embeddings vs hierarchical priors** - Should we learn a dense embedding per home, or use a Bayesian approach with learned priors?

2. **Sequence encoder for episodes** - Should we encode the first N minutes of an episode with LSTM/Transformer and fuse with static features?

3. **Multi-task learning** - Should we jointly predict:
   - Time to target (main task)
   - Temperature trajectory (auxiliary)
   - Episode type (normal vs anomalous)

4. **Attention over historical episodes** - Can we attend over a home's past episodes to inform the current prediction?

### Training

5. **Loss function** - Given heavy tails, should we use:
   - Huber loss
   - Quantile regression (predict P50, P80, P90)
   - Heteroscedastic loss (predict mean + variance)

6. **Data augmentation** - Any sensible augmentations for time series episodes?

7. **Curriculum learning** - Train on easy (short, system running) episodes first?

### Transfer & Adaptation

8. **Meta-learning** - MAML-style adaptation to new homes with few episodes?

9. **Home similarity** - Can we learn a home embedding space where similar homes cluster?

10. **Online learning** - How to update model as new episodes arrive without catastrophic forgetting?

---

## Evaluation Protocol

### Within-Home Split (current)
- For each home: 70% episodes to train, 30% to test
- Tests: How well can we predict for a home given its history?

### Cross-Home Split (important for deployment)
- Train on 70% of homes, test on 30% held-out homes
- Tests: How well do we generalize to new homes?

### Metrics
- **MAE** (minutes) - primary
- **Median AE** - robust to outliers
- **P90 AE** - worst-case for planning
- **MAPE** - relative error

### Stratification
Report results by:
- Initial gap (1-2°F, 2-3°F, 3-5°F, >5°F)
- Change type (heating vs cooling)
- System running (yes/no)

---

## What Success Looks Like

| Metric | Current (Hybrid GBM) | Target | Rationale |
|--------|----------------------|--------|-----------|
| MAE | 18.7 min | <15 min | Tighter MPC planning |
| Median | 9.5 min | <8 min | Most predictions very accurate |
| P90 | 46.6 min | <35 min | Reduce worst-case errors |
| Cold-start MAE | ~30 min | <20 min | Critical for deployment |

Beyond accuracy:
- Calibrated uncertainty estimates for MPC
- Fast adaptation to new homes (<20 episodes)
- Interpretable failure modes

---

## Data Access

```python
import polars as pl

# Load episode data
df = pl.read_parquet("data/setpoint_responses.parquet")

# Each row is one 5-min timestep within an episode
# Group by episode_id to get episode-level data
episodes = df.group_by("episode_id").agg([
    pl.col("home_id").first(),
    pl.col("initial_gap").first(),
    pl.col("change_type").first(),
    pl.col("target_setpoint").first(),
    pl.col("Indoor_AverageTemperature").first().alias("start_temp"),
    pl.col("Outdoor_Temperature").first().alias("outdoor_temp"),
    pl.len().alias("n_timesteps"),  # Duration = n_timesteps * 5 minutes
])

# Full trajectory for an episode
ep = df.filter(pl.col("episode_id") == "some_id").sort("timestep_idx")
temps = ep["Indoor_AverageTemperature"].to_numpy()
runtimes = ep["HeatingEquipmentStage1_RunTime"].to_numpy()
```

---

## References

- Linear baselines: `scripts/run_improved_baselines.py`
- GBM baselines: `scripts/run_xgboost_baselines.py`
- Error analysis: `plans/ERROR_ANALYSIS_PLAN.md`
- Dataset docs: `docs/SETPOINT_RESPONSES.md`
