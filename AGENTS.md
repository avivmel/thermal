# Thermal MPC Project

ML-based thermostat control system for residential HVAC demand response.

## Instructions

Before starting work on this project, read the relevant documentation files in `docs/` to understand the current state of the project, experimental results, and design decisions.

## Project Goal

Develop a smart thermostat system that reduces peak electricity demand while maintaining occupant comfort, using neural network-based thermal prediction combined with Model Predictive Control (MPC).

## Problem Context

- Residential HVAC systems draw 3-5 kW during operation, creating significant peak demand
- Current DR programs use rule-based control (uniform setpoint adjustments) that ignore building-specific thermal dynamics
- This project replaces rule-based approaches with learned thermal models + optimization

## Architecture

### 1. Thermal Prediction Model
- **Input**: Indoor temp, outdoor temp, setpoints, time features (NOT runtime - thermostat doesn't know future HVAC state)
- **Output**: Predicted indoor temperature 30 minutes ahead (single point, not trajectory)
- **Key constraint**: Must generalize to new homes without per-home fine-tuning
- **Model progression**: Baselines -> RNN -> LSTM -> Transformer

### 2. Model Predictive Controller (MPC)
- Optimizes HVAC schedule over receding horizon
- Objective: Minimize electricity cost under time-of-use pricing (peak: 5-8pm)
- Constraints: Keep indoor temp within user-specified comfort bounds

## Current Status: Time-to-Target Prediction

**Best model achieves 18.7 min MAE** (median 9.5 min) - highly usable for MPC planning.

### Linear Models

| Model | MAE | Median | P90 | Improvement |
|-------|-----|--------|-----|-------------|
| Per-Home + Mode (linear) | 22.4 min | 13.1 min | 51.9 min | baseline |
| Log-Duration Per-Home | 20.5 min | 11.0 min | 48.9 min | -8.6% |
| Hierarchical + Enhanced | 20.2 min | 10.8 min | 48.9 min | -9.7% |

### Gradient Boosting Models

| Model | MAE | Median | P90 | Improvement |
|-------|-----|--------|-----|-------------|
| Global GBM (no home info) | 20.8 min | 11.0 min | 50.9 min | -7.1% |
| Global GBM + Home Encoding | 18.8 min | 9.6 min | 46.4 min | -16.1% |
| Per-Home GBM | 19.0 min | 9.2 min | 48.7 min | -15.2% |
| **Hybrid GBM** | **18.7 min** | **9.5 min** | **46.6 min** | **-16.5%** |

*Note: Filtered to episodes ≤4 hours. Anomalous long episodes (>4h) were 9% of data but caused 55% of errors.*

### Key Discoveries

1. **Filter anomalies** - Episodes >4 hours are equipment failures/vacation mode; filtering reduced MAE by 40%
2. **Log-transform is essential** - homes vary 20x in thermal response (k: 6-120 min/°F)
3. **`system_running` is highly predictive** - if HVAC already active, episodes complete 50% faster
4. **Heating 28% faster than cooling** - consistent across models (coef: -0.33)
5. **Home target encoding works** - encoding home_id as mean log-duration (with shrinkage) is very effective
6. **Hybrid approach is best** - global GBM + per-home residual correction beats per-home models
7. **Cold-start still hard** - system not running: ~30 min MAE vs ~16 min when running

### Learned Coefficients (Hierarchical Linear Model)

| Feature | Coefficient | Meaning |
|---------|-------------|---------|
| `log_gap` | +0.72 | Duration scales ~linearly with gap |
| `is_heating` | -0.33 | Heating 28% faster |
| `system_running` | -0.69 | Already-on = 50% faster |
| `signed_thermal_drive` | -0.009 | Small outdoor temp effect |
| `month_cos` | -0.05 | Slight seasonal effect |

---

## Key Findings

### Critical: Temperature Data is Quantized to Integers

**The Ecobee sensors report whole degrees Fahrenheit only.** This fundamentally limits prediction granularity.

| Observation | Value |
|-------------|-------|
| Homes with integer-only temps | 98% |
| 15-min windows with exactly 0°F change | 66% |
| Unique delta values (should be continuous) | 73 |

**Temperature trajectories are stair-steps:**
```
Episode: 69→69→69→69→69→70  (sits at 69°F for 25min, jumps to 70°F)
Episode: 77→77→77→77→76→76→76→76→75→75→75→75→74
```

**Why persistence baseline wins for temperature prediction:** With integer quantization, most 15-min windows don't cross a degree boundary. Predicting "no change" is literally correct 66% of the time.

**Solution:** Reframe as time-to-target prediction (continuous minutes, not quantized degrees).

### Temperature Prediction Baselines (for reference)

**30-min horizon, arbitrary timesteps:**
- Persistence: 0.434°F MAE - very strong due to quantization
- Linear models can't beat it (bias-variance tradeoff)
- See `docs/BASELINE_RESULTS.md`

**15-min horizon, setpoint response episodes:**
- Persistence: 0.433°F MAE - still wins on MAE
- Per-home thermal: 0.475 MAE but best RMSE (-17%)
- See `docs/SETPOINT_BASELINES.md`

### Time-to-Target Prediction (recommended approach)

Predict minutes until temperature reaches setpoint. Avoids quantization issues entirely.

See `docs/TIME_TO_TARGET.md` for full results.

### Data Split Strategy
- **Home split (NOT temporal)**: 70% train / 10% val / 20% test
- **Stratified by state** to ensure climate diversity in each split
- **All homes use full year** - no time holdout
- **Rationale**: Goal is new-home generalization, not time generalization

## Dataset: Ecobee Donate Your Data (DYD)

Curated subset from LBNL containing smart thermostat data from ~1,000 single-family homes.

| Attribute | Value |
|-----------|-------|
| Time Period | January 1 - December 31, 2017 |
| Resolution | 5-minute intervals |
| Homes | 971 with valid state assignments |
| Raw Size | ~31.66 GB (NetCDF files) |
| Raw Location | `ecobee_processed_dataset/extracted/clean_data/` |

### Prepared Datasets

**Flat Dataset** (arbitrary timesteps)
| Attribute | Value |
|-----------|-------|
| File | `data/thermal_dataset.csv` |
| Size | 8.4 GB |
| Rows | 101M |
| Columns | home_id, timestamp, state, split, Indoor_AverageTemperature, Outdoor_Temperature, Indoor_HeatSetpoint, Indoor_CoolSetpoint |

**Setpoint Response Dataset** (goal-oriented episodes)
| Attribute | Value |
|-----------|-------|
| File | `data/setpoint_responses.parquet` |
| Size | 72 MB |
| Rows | 1.9M |
| Episodes | 159K (102K heat_increase, 57K cool_decrease) |
| Use case | Predict temp trajectory after setpoint change |

See `docs/SETPOINT_RESPONSES.md` for details.

### Geographic Distribution
| State | Total | Train | Val | Test |
|-------|-------|-------|-----|------|
| CA | 346 | 242 | 35 | 69 |
| TX | 244 | 171 | 24 | 49 |
| IL | 248 | 174 | 25 | 49 |
| NY | 133 | 93 | 13 | 27 |

### Key Variables
| Variable | Unit | Description |
|----------|------|-------------|
| `Indoor_AverageTemperature` | °F | Target variable |
| `Outdoor_Temperature` | °F | Outdoor air temperature |
| `Indoor_HeatSetpoint` | °F | Heating setpoint (HVAC heats when indoor < this) |
| `Indoor_CoolSetpoint` | °F | Cooling setpoint (HVAC cools when indoor > this) |

### Missing Value Indicators
- `0.0` for float variables
- `-9999` for HVAC_Mode
- Empty string for Schedule/Event

## Next Steps

1. **Two-stage early update** - After 10-15 min, use observed progress to refine prediction
2. **Quantile regression** - Add P50/P80/P90 predictions for MPC uncertainty bounds
3. **Cross-home evaluation** - Test generalization to unseen homes (cold start)
4. **Neural network models** - Try embeddings + MLP for nonlinear transfer

---

## CityLearn DR Simulation Results

Tested demand response strategies in CityLearn simulation environment. See `docs/CITYLEARN_DR_RESULTS.md` for full details.

### Key Finding

**Simple time-based RBC fails; feedback control with time-varying gains succeeds.**

| Controller | PAR | Peak Load | Energy | Result |
|------------|-----|-----------|--------|--------|
| Baseline | 3.26 | 495 kW | 327 MWh | - |
| Open-loop RBC | 1.54 | 684 kW | 956 MWh | FAILED (3x energy) |
| **Feedback + PeakShave** | **2.81** | **255 kW** | **196 MWh** | **SUCCESS** |

### Why Open-Loop Failed

Action = fraction of *nominal power*, NOT *heating needed*. Setting `action=0.7` runs heat pump at 70% max power (~20 kW) regardless of actual need (~5 kW), causing massive overheating.

### Why Feedback Worked

```python
error = setpoint - current_temp
action = Kp * max(0, error) + bias
# Kp varies: high during pre-heat (14-16), low during peak (17-20)
```

Proportional control only heats when needed, time-varying gains shift load from peak to pre-peak hours.

### Implications for MPC

- **Feedback control is the baseline to beat** (14% PAR reduction, 48% peak reduction)
- **MPC should improve** by optimizing schedule using thermal predictions
- **CityLearn validated** as simulation platform for DR testing

## Tech Stack

- PyTorch for model training
- Polars for fast data loading
- xarray/netCDF4 for raw data
- scikit-learn for baselines
- statsmodels for mixed effects models

## Project Structure

```
thermal/
├── CLAUDE.md                           # This file
├── data/
│   ├── thermal_dataset.csv             # Flat dataset (8.4 GB)
│   └── setpoint_responses.parquet      # Setpoint response episodes (72 MB)
├── scripts/
│   ├── prepare_dataset.py              # Creates thermal_dataset.csv
│   ├── extract_setpoint_responses.py   # Creates setpoint_responses.parquet
│   ├── run_baselines.py                # Flat dataset baselines
│   ├── run_setpoint_baselines.py       # Setpoint response baselines
│   ├── run_time_to_target.py           # Time-to-target baselines (linear)
│   ├── run_improved_baselines.py       # Log-duration + hierarchical models ★
│   ├── run_xgboost_baselines.py        # Gradient boosting models ★
│   └── analyze_errors.py               # Error analysis script ★
├── docs/
│   ├── BASELINE_MODELS.md              # Detailed model descriptions
│   ├── BASELINE_RESULTS.md             # Results and analysis
│   ├── DATA_ANALYSIS.md                # Temperature change distributions
│   ├── SETPOINT_RESPONSES.md           # Setpoint response dataset docs
│   ├── SETPOINT_BASELINES.md           # Setpoint response baseline docs
│   ├── TIME_TO_TARGET.md               # Time-to-target prediction docs
│   ├── PREDICTION_TASK_BRIEF.md        # Task brief for external collaborators
│   ├── DEEP_LEARNING_BRIEF.md          # Deep learning consultation prompt ★
│   ├── citylearn_datasets.md           # CityLearn simulation datasets
│   ├── CITYLEARN_DR_EXPERIMENT.md      # DR experiment plan ★
│   └── CITYLEARN_DR_RESULTS.md         # DR experiment results ★
├── plans/
│   ├── BASELINE_PLAN.md                # Baseline experiment plan
│   ├── DATA_PLAN.md                    # Data preparation plan
│   ├── RNN_PLAN.md                     # RNN experiment plan
│   └── ERROR_ANALYSIS_PLAN.md          # Error analysis & next improvements ★
├── ecobee_processed_dataset/
│   ├── extracted/clean_data/           # Raw NetCDF files (~2.6 GB each)
│   │   ├── Jan_clean.nc
│   │   └── ...
│   └── DATASET_DOCUMENTATION.md
└── CityLearn/                          # Simulation environment
```

## References

- Dataset DOI: 10.25584/ecobee/1854924
- CityLearn: arXiv:2012.10504
