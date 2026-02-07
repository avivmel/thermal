# Baseline Plan: Simple Thermal Prediction

## Problem

Predict indoor temperature 30 minutes (6 timesteps) ahead.

---

## Dataset

**Source:** Ecobee Donate Your Data (LBNL subset)

| Attribute | Value |
|-----------|-------|
| Homes | 990 |
| Time period | Jan - Dec 2017 |
| Resolution | 5-minute intervals |
| Location | `ecobee_processed_dataset/clean_data/` |

### Geographic Distribution

| State | Homes | Climate |
|-------|-------|---------|
| CA | 346 | Warm-Marine |
| TX | 244 | Hot-Humid |
| IL | 248 | Cold-Humid |
| NY | 133 | Mixed-Humid |

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

### Stratification by State

| State | Train | Val | Test |
|-------|-------|-----|------|
| CA | 242 | 35 | 69 |
| TX | 171 | 24 | 49 |
| IL | 174 | 25 | 49 |
| NY | 93 | 13 | 27 |

### Sampling

For each home, sample random windows throughout the year:
- Input: C timesteps of context
- Target: Indoor temp at t+6 (30 min ahead)

---

## Baseline Models

### 1. Persistence (Naive)

Predict that temperature stays the same.

```python
y_pred = indoor_temp[t]
```

**Inputs:** Current indoor temp only

---

### 2. Linear Trend

Extrapolate recent temperature trend.

```python
slope = (indoor_temp[t] - indoor_temp[t-6]) / 6
y_pred = indoor_temp[t] + slope * 6
```

**Inputs:** Past 6 indoor temps

---

### 3. Linear Regression

Fit coefficients on training data.

```python
y_pred = w0 + w1*indoor_temp[t] + w2*outdoor_temp[t] + w3*heat_sp[t] + w4*cool_sp[t]
```

**Inputs:**
- Current indoor temp
- Current outdoor temp
- Current setpoints

**Training:** Least squares on train homes

---

### 4. Linear Regression + Trend

Add recent temperature change as feature.

```python
delta_temp = indoor_temp[t] - indoor_temp[t-6]

y_pred = w0 + w1*indoor_temp[t] + w2*outdoor_temp[t] + w3*heat_sp[t] + w4*cool_sp[t] + w5*delta_temp
```

**Inputs:**
- Current indoor temp
- Current outdoor temp
- Current setpoints
- 30-min temperature change

---

### 5. Thermal Decay Model (Physics-Inspired)

Temperature drifts toward a target based on HVAC state.

```python
if indoor_temp[t] < heat_sp[t]:
    target = heat_sp[t]
    tau = tau_heat
elif indoor_temp[t] > cool_sp[t]:
    target = cool_sp[t]
    tau = tau_cool
else:
    target = outdoor_temp_forecast
    tau = tau_passive

y_pred = target + (indoor_temp[t] - target) * exp(-30 / tau)
```

**Inputs:**
- Current indoor temp
- Outdoor temp forecast
- Setpoints

**Learned params:** tau_heat, tau_cool, tau_passive (fit on train homes)

---

## Experiments

| ID | Model | Features |
|----|-------|----------|
| B1 | Persistence | indoor_temp |
| B2 | Linear trend | indoor_temp history |
| B3 | Linear regression | indoor, outdoor, setpoints |
| B4 | LinReg + trend | above + delta_temp |
| B5 | Thermal decay | indoor, outdoor forecast, setpoints |

---

## Evaluation

**Metric:** MAE in °F on test homes (198 homes, full year)

**Target:** Establish baseline that RNN must beat
