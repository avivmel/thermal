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
- **Input**: Indoor temp, outdoor temp, HVAC runtime, setpoints, time of day
- **Output**: Predicted indoor temperature trajectory over multi-hour horizon
- **Key constraint**: Must generalize to new homes without per-home fine-tuning (achieved via context window of historical data)
- **Model progression**: RNN (baseline) -> LSTM -> Transformer

### 2. Model Predictive Controller (MPC)
- Optimizes HVAC schedule over receding horizon
- Objective: Minimize electricity cost under time-of-use pricing (peak: 5-8pm)
- Constraints: Keep indoor temp within user-specified comfort bounds
- Approaches to evaluate: Exhaustive enumeration, beam search, SQP solvers

## Dataset: Ecobee Donate Your Data (DYD)

Curated subset from LBNL containing smart thermostat data from 1,000 single-family homes.

| Attribute | Value |
|-----------|-------|
| Time Period | January 1 - December 31, 2017 |
| Resolution | 5-minute intervals |
| Homes | 990 with valid state assignments |
| Size | ~31.66 GB (NetCDF files) |
| Location | `ecobee_processed_dataset/clean_data/` |

### Geographic Distribution
- California (CA): 346 homes - Warm-Marine climate
- Texas (TX): 244 homes - Hot-Humid climate
- Illinois (IL): 248 homes - Cold-Humid climate
- New York (NY): 133 homes - Mixed-Humid climate

### Key Variables for Thermal Modeling
| Variable | Unit | Description |
|----------|------|-------------|
| `Indoor_AverageTemperature` | F | Weighted average of all sensors |
| `Outdoor_Temperature` | F | Outdoor air temperature |
| `Indoor_CoolSetpoint` | F | Cooling setpoint |
| `Indoor_HeatSetpoint` | F | Heating setpoint |
| `HeatingEquipmentStage1_RunTime` | sec | Heating runtime (0-300 per 5-min interval) |
| `CoolingEquipmentStage1_RunTime` | sec | Cooling runtime (0-300 per 5-min interval) |

### Data Loading Example
```python
import xarray as xr

ds = xr.open_dataset('ecobee_processed_dataset/clean_data/Jan_clean.nc')
home_id = ds.id.values[0]
temp = ds['Indoor_AverageTemperature'].sel(id=home_id)
```

### Missing Value Indicators
- `0.0` for float variables
- `-9999` for HVAC_Mode
- Empty string for Schedule/Event

## Evaluation

- **Thermal model**: Mean Absolute Error (MAE) at multiple time horizons
- **Control system**: CityLearn simulation environment
  - Constraint adherence (temperature within comfort bounds)
  - Total electricity cost
  - Peak-to-average ratio (PAR) reduction

## Tech Stack

- PyTorch for model training
- xarray/netCDF4 for data loading
- CityLearn for simulation

## Project Structure

```
thermal/
├── CLAUDE.md                    # This file
├── ecobee_processed_dataset/
│   ├── DATASET_DOCUMENTATION.md # Full dataset documentation
│   ├── Metadata file_Ecobee.json
│   └── clean_data/
│       ├── Jan_clean.nc         # Monthly NetCDF files (~2.6 GB each)
│       ├── Feb_clean.nc
│       └── ...
```

## References

- Dataset DOI: 10.25584/ecobee/1854924
- CityLearn: arXiv:2012.10504
