# Experiment Plan: RNN Thermal Prediction

## Problem

Predict indoor temperature 30 minutes (6 timesteps) into the future.

---

## Dataset

**Source:** Ecobee Donate Your Data (LBNL subset)

| Attribute | Value |
|-----------|-------|
| Homes | 990 |
| Time period | Jan - Dec 2017 |
| Resolution | 5-minute intervals |
| Location | `ecobee_processed_dataset/clean_data/` |

### Variables Used

| Variable | Unit | Description |
|----------|------|-------------|
| `Indoor_AverageTemperature` | °F | Target variable |
| `Outdoor_Temperature` | °F | Outdoor air temp |
| `Indoor_HeatSetpoint` | °F | Heating setpoint |
| `Indoor_CoolSetpoint` | °F | Cooling setpoint |

---

## Data Split

**Goal:** Test generalization to unseen homes (not unseen time periods).

### Home Split (stratified by state)

| Split | Homes | % | Purpose |
|-------|-------|---|---------|
| Train | 693 | 70% | Model training |
| Val | 99 | 10% | Hyperparameter tuning |
| Test | 198 | 20% | Final evaluation |

### Time Split

**All homes use full year (Jan-Dec).**

This ensures:
- Model sees all seasonal patterns during training
- We isolate home generalization (not conflated with time generalization)
- Val/test homes are completely unseen during training

---

## Model Inputs

### Context (Past)
Configurable history length `C` timesteps.

| Feature | Shape | Description |
|---------|-------|-------------|
| `indoor_temp` | [C] | Past indoor temperatures |
| `outdoor_temp` | [C] | Past outdoor temperatures |
| `heat_setpoint` | [C] | Past heating setpoints |
| `cool_setpoint` | [C] | Past cooling setpoints |
| `hour_sin` | [C] | sin(2π × hour/24) |
| `hour_cos` | [C] | cos(2π × hour/24) |

**Context tensor:** `[batch, C, 6]`

### Future (Known at prediction time)

| Feature | Shape | Description |
|---------|-------|-------------|
| `outdoor_temp_forecast` | [1] | Weather forecast at t+6 |
| `heat_setpoint_future` | [1] | Planned heating setpoint at t+6 |
| `cool_setpoint_future` | [1] | Planned cooling setpoint at t+6 |
| `hour_sin_future` | [1] | Time encoding at t+6 |
| `hour_cos_future` | [1] | Time encoding at t+6 |

**Future tensor:** `[batch, 5]`

---

## Model Output

| Output | Shape | Description |
|--------|-------|-------------|
| `indoor_temp_pred` | [1] | Predicted indoor temp at t+6 (30 min ahead) |

---

## Architecture

```
Context [B, C, 6] ──→ GRU ──→ h_final [B, hidden]
                                    │
                                    ├── concat ──→ MLP ──→ prediction [B, 1]
                                    │
Future [B, 5] ──────────────────────┘
```

---

## Training Objective

```python
loss = MSE(y_pred, y_true)
```

Target: `Indoor_AverageTemperature` at t+6

---

## Experiments

### Context Length (Primary)

| Exp | C | Duration |
|-----|---|----------|
| C1 | 6 | 30 min |
| C2 | 12 | 1 hr |
| C3 | 24 | 2 hr |
| C4 | 72 | 6 hr |

### Architecture

| Exp | Hidden | Layers |
|-----|--------|--------|
| A1 | 64 | 2 |
| A2 | 128 | 2 |

---

## Evaluation

**Metric:** MAE in °F on test homes (198 homes, full year)

**Target:** < 1.0°F (must beat baselines)
