# Data Plan: Ecobee Thermal Prediction Dataset

## Overview

This document outlines the data preparation strategy for training thermal prediction models on the Ecobee DYD dataset.

**Dataset Summary:**
- 990 homes across 4 U.S. states (CA: 346, TX: 244, IL: 248, NY: 133)
- Full year 2017, 5-minute resolution (~105K timesteps/home)
- ~31.66 GB total (monthly NetCDF files)

---

## 1. Available Features

### 1.1 Time Series Features (per timestep)

| Feature | Description | Use Case |
|---------|-------------|----------|
| `Indoor_AverageTemperature` | Weighted avg of all sensors (F) | **Target variable** |
| `Outdoor_Temperature` | Outdoor air temp (F) | Primary driver |
| `Indoor_HeatSetpoint` | Heating setpoint (F) | Control signal |
| `Indoor_CoolSetpoint` | Cooling setpoint (F) | Control signal |
| `HeatingEquipmentStage1_RunTime` | Heating runtime (0-300 sec) | HVAC state |
| `CoolingEquipmentStage1_RunTime` | Cooling runtime (0-300 sec) | HVAC state |
| `HeatPumpsStage1_RunTime` | Heat pump runtime (0-300 sec) | HVAC state |
| `Fan_RunTime` | Fan runtime (0-300 sec) | HVAC activity indicator |
| `Indoor_Humidity` | Indoor RH (%) | Secondary driver |
| `Outdoor_Humidity` | Outdoor RH (%) | Secondary driver |
| `Schedule` | Home/Away/Sleep | Occupancy proxy |
| `Thermostat_DetectedMotion` | Motion at thermostat | Occupancy proxy |
| `RemoteSensorN_Temperature` | Room-level temps (1-5) | Spatial variation |

### 1.2 Derived Time Features

| Feature | Derivation |
|---------|------------|
| `hour_of_day` | Extract from timestamp (0-23) |
| `day_of_week` | Extract from timestamp (0-6) |
| `month` | Extract from timestamp (1-12) |
| `is_weekend` | Saturday/Sunday flag |
| `hour_sin`, `hour_cos` | Cyclical encoding |

### 1.3 Derivable Home-Level Features

These are **static per home** but must be computed from the time series:

| Feature | Derivation Method | Variation Observed |
|---------|-------------------|-------------------|
| `state` | From `State` column | 4 categories |
| `climate_zone` | Map from state (2A, 3C, 4A, 5A) | 4 categories |
| `hvac_type` | Which runtime columns have data | Furnace, HeatPump, AC, Multi-stage |
| `num_remote_sensors` | Count sensors with >100 valid readings | 0-5 (most homes: 1 or 3) |
| `typical_heat_setpoint` | Mean of `Indoor_HeatSetpoint` | 62-72F range |
| `typical_cool_setpoint` | Mean of `Indoor_CoolSetpoint` | 75-80F range |
| `thermal_mass_proxy` | Temp decay rate when HVAC off | High variance across homes |
| `occupancy_rate` | % of time with motion detected | ~20% average |
| `schedule_complexity` | Entropy of schedule distribution | Varies by household |

**Key Finding:** Thermal drift rate varies significantly across homes (-0.017 to +0.097 F/5min), suggesting building characteristics substantially affect dynamics.

---

## 2. Data Quality Notes

| Variable | Issue | Handling |
|----------|-------|----------|
| HVAC Runtime columns | 70-99% NaN (equipment not present) | Use NaN = 0 (no runtime) |
| `Indoor_AverageTemperature` | ~1.2% NaN | Linear interpolation (already applied) |
| `Outdoor_Temperature` | ~1.0% NaN | Linear interpolation |
| Missing indicator | `0.0` for floats, `-9999` for HVAC_Mode | Filter/impute |

---

## 3. Train / Validation / Test Split Strategy

### 3.1 Recommended: Temporal + Home Holdout Split

```
                    Train Homes (80%)              Val/Test Homes (20%)
                    ─────────────────              ────────────────────
Jan-Sep 2017       TRAIN                          HELD OUT
                   (learn thermal dynamics)

Oct-Nov 2017       VALIDATION                     VALIDATION
                   (in-distribution homes,        (OOD homes,
                    OOD time)                      OOD time)

Dec 2017           TEST                           TEST
                   (in-distribution homes,        (OOD homes,
                    OOD time)                      OOD time)
```

**Rationale:**
- **Temporal split**: MPC operates on future predictions; must test on unseen future
- **Home holdout**: Key requirement is generalization to new homes without fine-tuning
- **Stratified by state**: Ensure each split has proportional climate zone representation

### 3.2 Split Sizes

| Split | Homes | Months | Purpose |
|-------|-------|--------|---------|
| Train | 792 | Jan-Sep | Learn thermal dynamics |
| Val (in-dist) | 792 | Oct-Nov | Hyperparameter tuning |
| Val (OOD) | 198 | Oct-Nov | Assess generalization |
| Test (in-dist) | 792 | Dec | Final in-distribution eval |
| Test (OOD) | 198 | Dec | **Primary metric**: new-home generalization |

### 3.3 Stratification

Ensure home splits maintain state distribution:
```
CA: 277 train, 69 test
TX: 195 train, 49 test
IL: 198 train, 50 test
NY: 106 train, 27 test
```

---

## 4. Experimental Axes

### 4.1 Input Feature Experiments

| Experiment | Features | Hypothesis |
|------------|----------|------------|
| **Baseline** | Indoor temp, outdoor temp, HVAC runtime, time features | Minimum viable |
| **+ Setpoints** | Add heat/cool setpoints | Captures control intent |
| **+ Humidity** | Add indoor/outdoor humidity | Secondary thermal driver |
| **+ Schedule** | Add Home/Away/Sleep | Occupancy proxy |
| **+ Motion** | Add motion detection | Direct occupancy signal |
| **+ Remote Sensors** | Add room-level temps | Spatial thermal distribution |

### 4.2 Home Context Experiments

**Core Question:** Does providing home-level context improve generalization?

| Experiment | Context Provided | Expected Outcome |
|------------|------------------|------------------|
| **No context** | Just time series | Baseline; must learn implicitly |
| **State only** | Climate zone embedding | Regional patterns |
| **HVAC type** | Equipment embedding | System-specific dynamics |
| **Full derived** | All derivable features | Best generalization (hypothesis) |
| **Learned context** | Context window of past data | Model learns to infer home characteristics |

**Learned Context Approach (from CLAUDE.md):**
> "Key constraint: Must generalize to new homes without per-home fine-tuning (achieved via context window of historical data)"

This means providing N hours of past data as context, letting the model implicitly infer home characteristics from observed dynamics.

### 4.3 Context Window Experiments

| Window Size | Timesteps | Compute | Information |
|-------------|-----------|---------|-------------|
| 1 hour | 12 | Low | Minimal context |
| 6 hours | 72 | Medium | Captures diurnal patterns |
| 24 hours | 288 | High | Full day cycle |
| 7 days | 2016 | Very high | Weekly patterns |

---

## 5. Data Pipeline Design

### 5.1 Preprocessing Steps

1. **Load monthly NetCDF files** via xarray
2. **Compute derived home features** (HVAC type, sensor count, etc.)
3. **Handle missing values:**
   - NaN runtime = 0 (equipment not running/present)
   - Interpolate small gaps in temperature
   - Flag homes with >10% missing indoor temp
4. **Normalize:**
   - Temperatures: StandardScaler per-feature (fit on train)
   - Runtime: Scale to [0, 1] (divide by 300)
   - Categorical: Embeddings or one-hot
5. **Create sequences:**
   - Context window: past N timesteps
   - Prediction horizon: 1-24 steps ahead (5 min - 2 hours)

### 5.2 Output Format

```python
# Per-sample structure
{
    "home_id": str,
    "timestamp": datetime,

    # Context (past)
    "context_temps": [N, features],      # Past N timesteps
    "context_hvac": [N, hvac_features],

    # Home-level (static)
    "home_features": {
        "state": str,
        "hvac_type": List[str],
        "num_sensors": int,
        ...
    },

    # Target (future)
    "target_temps": [H],  # Next H timesteps of indoor temp
}
```

---

## 6. Evaluation Metrics

### 6.1 Primary: Temperature Prediction

| Metric | Horizon | Purpose |
|--------|---------|---------|
| MAE @ 15min | 3 steps | Short-term accuracy |
| MAE @ 1hr | 12 steps | MPC-relevant horizon |
| MAE @ 2hr | 24 steps | Extended prediction |
| MAE @ 4hr | 48 steps | DR event horizon |

### 6.2 Secondary: Generalization

| Metric | Split | Purpose |
|--------|-------|---------|
| OOD Home Gap | Test OOD - Test In-Dist | Quantify generalization penalty |
| Per-State MAE | By climate zone | Climate-specific performance |
| Per-HVAC-Type MAE | By equipment | System-specific performance |

### 6.3 Downstream: MPC Performance

Evaluate in CityLearn simulation:
- Temperature constraint violations
- Peak demand reduction
- Energy cost savings

---

## 7. Implementation Checklist

- [ ] Write data loader for NetCDF files
- [ ] Implement home feature derivation
- [ ] Create train/val/test home splits (stratified by state)
- [ ] Implement sequence windowing
- [ ] Build PyTorch Dataset class
- [ ] Add normalization/denormalization utilities
- [ ] Create data quality report script
- [ ] Set up experiment configs for feature ablations

---

## 8. File Organization

```
thermal/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ecobee_dataset.py      # PyTorch Dataset
│   │   ├── preprocessing.py        # Normalization, imputation
│   │   ├── home_features.py        # Derive static features
│   │   ├── splits.py               # Train/val/test split logic
│   │   └── quality.py              # Data quality checks
│   └── ...
├── configs/
│   ├── data/
│   │   ├── baseline.yaml
│   │   ├── full_features.yaml
│   │   └── context_ablation.yaml
│   └── ...
└── scripts/
    ├── prepare_splits.py           # One-time split generation
    └── data_quality_report.py      # Generate quality stats
```

---

## 9. Next Steps

1. **Implement data loader** with configurable feature sets
2. **Generate home splits** and save to disk
3. **Compute and cache derived home features**
4. **Build baseline RNN** on minimal features
5. **Run feature ablation experiments**
6. **Evaluate context window approaches**
