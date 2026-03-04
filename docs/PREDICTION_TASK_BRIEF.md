# Thermal Response Time Prediction: Task Brief

## Problem Context

We're building a smart thermostat system for **residential demand response (DR)**. During peak electricity hours (5-8pm), we want to pre-heat or pre-cool homes so HVAC can be turned off during the peak without sacrificing comfort.

**The key planning question**: *"If I change the setpoint now, how long until the home reaches the target temperature?"*

Accurate prediction of this "time-to-target" enables optimal scheduling of HVAC pre-conditioning.

---

## Prediction Task

**Input**: State at the moment a setpoint changes
- Current indoor temperature (°F)
- Target setpoint (°F)
- Initial gap: |target - current| (°F)
- Outdoor temperature (°F)
- Change type: heating or cooling
- Home identifier (for per-home models)

**Output**: Time in minutes until indoor temperature reaches the target setpoint

**Example**:
```
Input:  current=68°F, target=72°F, gap=4°F, outdoor=45°F, type=heating, home=X
Output: 85 minutes
```

---

## Dataset

**Source**: Ecobee Donate Your Data (smart thermostat data from ~1,000 US homes, full year 2017)

**Setpoint Response Episodes**: 158,604 episodes where:
- Setpoint changed by >1°F
- Indoor temperature subsequently reached the new setpoint
- No interruptions (setpoint didn't change again mid-episode)

| Statistic | Value |
|-----------|-------|
| Total episodes | 158,604 |
| Heat increase episodes | 102K (64%) |
| Cool decrease episodes | 57K (36%) |
| Mean duration | 60 min |
| Median duration | 30 min |
| Duration range | 5 min - 8+ hours |

**Episode duration by initial gap**:
| Gap | Median Duration | Mean Duration |
|-----|-----------------|---------------|
| 1-2°F | 20 min | 39 min |
| 2-3°F | 30 min | 56 min |
| 3-5°F | 40 min | 79 min |
| 5-10°F | 60 min | 110 min |

---

## Current Baseline Results

We've established baselines using simple models. Evaluation uses **within-home episode splits** (70% train / 30% test per home).

| Model | MAE | Description |
|-------|-----|-------------|
| Global Mean | 50.9 min | Predict average duration |
| Gap Proportional | 46.1 min | `time = k × gap`, k=27 min/°F |
| Mode-Specific | 46.1 min | Separate k for heating vs cooling |
| **Per-Home Gap** | **43.0 min** | Learn k per home |
| **Per-Home + Mode** | **40.5 min** | Learn k per home AND mode |

**Best model achieves**:
- MAE: 40.5 minutes
- Median error: 17 minutes
- MAPE: 82%

---

## Key Challenge: Extreme Home-to-Home Variation

The learned rate parameter `k` (minutes per °F of gap) varies enormously across homes:

| Statistic | k (min/°F) |
|-----------|------------|
| Global average | 27 |
| Minimum (fastest home) | 6 |
| Maximum (slowest home) | 120 |

**A 20x difference** between fastest and slowest homes! This reflects real physical differences:
- HVAC system power (BTU capacity)
- Home size and thermal mass
- Insulation quality
- Climate zone

This suggests **per-home modeling is essential**, but we need approaches that can:
1. Learn quickly from limited episodes per home (~100-200 episodes/home)
2. Transfer knowledge across similar homes
3. Adapt to new homes with few observations

---

## Additional Data Available

Beyond the core features, we have:

| Variable | Description |
|----------|-------------|
| `Outdoor_Humidity` | Outdoor humidity (%) |
| `Indoor_Humidity` | Indoor humidity (%) |
| `HeatingEquipmentStage1_RunTime` | Heating runtime per 5-min interval |
| `CoolingEquipmentStage1_RunTime` | Cooling runtime per 5-min interval |
| `Fan_RunTime` | Fan runtime per 5-min interval |
| `state` | US state (CA, TX, IL, NY) |
| Timestamp | Full datetime for seasonality |

**Note on runtime**: We have historical runtime data, but at prediction time (episode start) we don't know future HVAC behavior - only current state.

---

## Data Quirk: Integer Temperature Resolution

The Ecobee thermostats report temperature as **whole degrees Fahrenheit only**. This means:
- Temperature trajectories are stair-steps, not smooth curves
- 66% of 5-minute intervals show exactly 0°F change
- Fine-grained temperature prediction is noisy

This is why we reframed from "predict temperature in 15 min" to "predict time to reach target" - it sidesteps the quantization issue.

---

## What We're Looking For

Ideas to improve beyond the 40.5 min MAE baseline:

1. **Better per-home modeling**
   - How to learn home-specific dynamics from ~100-200 episodes?
   - Meta-learning / few-shot approaches?
   - Home embeddings?

2. **Feature engineering**
   - Thermal drive (outdoor - indoor) as predictor of rate?
   - Time-of-day / seasonal effects?
   - Derived features from historical episodes?

3. **Transfer learning**
   - Can we cluster similar homes?
   - Learn a shared representation + home-specific adaptation?

4. **Sequence modeling**
   - Use trajectory from first few minutes to update prediction?
   - Online adaptation during episode?

5. **Uncertainty quantification**
   - Prediction intervals for MPC planning?
   - When to trust the prediction vs fall back to conservative estimates?

6. **Architecture choices**
   - Simple models (linear, trees) vs neural networks?
   - Trade-offs given data size (~100-200 episodes per home)?

---

## Evaluation Protocol

**Within-home split**: For each home, 70% of episodes to train, 30% to test. This measures how well we can predict for a home given its history.

**Cross-home split**: Train on 70% of homes, test on 30% held-out homes. This measures generalization to new homes (harder, but important for deployment).

**Metrics**:
- MAE (minutes) - primary metric
- MAPE (%) - relative error
- Median absolute error - robustness to outliers
- P90 error - worst-case planning

**Stratification**: Report results by:
- Initial gap size (1-2°F, 2-3°F, 3-5°F, >5°F)
- Change type (heating vs cooling)

---

## Quick Start

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

episodes = episodes.with_columns([
    (pl.col("n_timesteps") * 5).alias("duration_min")
])
```

---

## Contact

Questions about the data or task? [Your contact info here]
