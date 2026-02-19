# Thermal MPC Project

ML-based thermostat control system for residential HVAC demand response.

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

## Key Findings

### Baseline Results (30-min horizon)
- **Persistence baseline: 0.434°F MAE** - very strong, linear models can't beat it
- Must evaluate by HVAC mode separately:
  - Passive (73%): 0.413 MAE - temp barely changes
  - Heating (20%): 0.481 MAE - temp rises ~0.36°F avg
  - Cooling (7%): 0.522 MAE - temp falls ~0.29°F avg
- Linear models correct bias but don't improve MAE (bias-variance tradeoff)
- See `docs/BASELINE_RESULTS.md` for full analysis

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

### Prepared Dataset
| Attribute | Value |
|-----------|-------|
| File | `data/thermal_dataset.csv` |
| Size | 8.4 GB |
| Rows | 101M |
| Columns | home_id, timestamp, state, split, Indoor_AverageTemperature, Outdoor_Temperature, Indoor_HeatSetpoint, Indoor_CoolSetpoint |

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

## Tech Stack

- PyTorch for model training
- Polars for fast data loading
- xarray/netCDF4 for raw data
- scikit-learn for baselines

## Project Structure

```
thermal/
├── CLAUDE.md                           # This file
├── data/
│   └── thermal_dataset.csv             # Prepared dataset (8.4 GB)
├── scripts/
│   ├── prepare_dataset.py              # Creates thermal_dataset.csv
│   └── run_baselines.py                # Runs all baseline models
├── docs/
│   ├── BASELINE_MODELS.md              # Detailed model descriptions
│   ├── BASELINE_RESULTS.md             # Results and analysis
│   └── DATA_ANALYSIS.md                # Temperature change distributions
├── plans/
│   ├── BASELINE_PLAN.md                # Baseline experiment plan
│   ├── DATA_PLAN.md                    # Data preparation plan
│   └── RNN_PLAN.md                     # RNN experiment plan
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
