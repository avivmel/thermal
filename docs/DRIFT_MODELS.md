# Drift Models: Complete Reference

This document covers every aspect of the passive thermal drift prediction system — what it is, why it exists, the data it is trained on, how it is engineered and trained, how predictions are made, how performance is evaluated, and how the model plugs into the MPC controller.

---

## 1. What "Drift" Means

**Passive thermal drift** is what happens when the HVAC system is off. With no active heating or cooling, the indoor temperature moves naturally toward the outdoor temperature, driven by heat flowing through the building envelope (walls, roof, windows).

This is distinct from **active** HVAC time, where the system is running and driving temperature toward a setpoint. Drift is purely passive — the building is its own thermal mass, and the rate of drift depends on how well-insulated it is and how large the outdoor–indoor temperature difference is.

### Physics

Newton's law of cooling: the rate of temperature change is proportional to the temperature difference between inside and outside.

```
dT_indoor/dt ≈ -k × (T_indoor - T_outdoor)
```

Where `k` is a building-specific thermal conductivity constant (inverse of thermal resistance). Homes in the Ecobee dataset span a 20× range in `k` (6–120 min/°F), which is why a global model needs to be carefully designed.

### Why Drift Matters for MPC

The MPC controller's core strategy during peak pricing hours is to **coast** — turn off HVAC entirely and let the home drift. The home was pre-heated (winter) or pre-cooled (summer) before the peak window, so it has a thermal buffer. The drift model answers the question:

> "Given where the indoor temperature is right now, how many minutes before it drifts to the comfort boundary?"

If the answer is "longer than the peak window," the controller coasts. If not, it defers HVAC until the boundary is actually reached.

---

## 2. Drift Data: `drift_episodes.parquet`

**Location:** `data/drift_episodes.parquet` (59 MB)

This dataset contains **passive drift episodes** extracted from the Ecobee dataset. Each episode begins at a timestep when the HVAC is off and records how long it takes for the indoor temperature to passively reach a boundary temperature.

### Schema

| Column | Description |
|--------|-------------|
| `home_id` | Identifier for the home |
| `timestamp` | Start timestamp of the episode |
| `start_temp` | Indoor temperature at the start of the drift episode (°F) |
| `boundary_temp` | The temperature boundary the home is drifting toward (°F) |
| `Outdoor_Temperature` | Outdoor temperature at episode start (°F) |
| `time_to_boundary_min` | Minutes until indoor temperature first crossed `boundary_temp` |
| `drift_direction` | `"warming_drift"` (indoor rising toward boundary) or `"cooling_drift"` (indoor falling toward boundary) |
| `crossed_boundary` | Boolean — `True` if the home actually reached the boundary |
| `timestep_idx` | Index within the episode (0 = start, 1 = 5 min later, etc.) |

### Episode Types

- **`warming_drift`**: Used in heating mode. The home is cooling off (HVAC off in winter) and drifts downward toward the lower comfort bound. Wait — this is a bit counterintuitive: in heating mode, when the HVAC turns off, the home drifts *down* because it's cold outside. So `warming_drift` actually means the outdoor air is warmer than the indoor boundary, which might happen in mild weather. More precisely, the direction naming reflects the direction of outdoor forcing, not the HVAC mode. In practice:
  - **Heating season**: HVAC off → indoor temp falls → approaching `comfort_lower_f`
  - **Cooling season**: HVAC off → indoor temp rises → approaching `comfort_upper_f`

---

## 3. Training Data Pipeline

### Step 1: Sample Selection (`create_drift_samples`)

**Source:** `scripts/run_xgboost_baselines.py`, `create_drift_samples()` (line 707)

From the full `drift_episodes.parquet`, only episodes that satisfy all of the following are kept:

| Filter | Value | Reason |
|--------|-------|--------|
| `timestep_idx == 0` | Start of each episode only | One sample per episode; avoids data leakage from mid-episode states |
| `crossed_boundary == True` | Must have reached the boundary | Ensures the label is observed, not censored |
| `time_to_boundary_min` ∈ [5, 480] | Between 5 minutes and 8 hours | 5 min is one timestep (lower floor); 8 hours filters out anomalies (stuck systems, vacation mode) |
| `drift_direction` matches | `warming_drift` or `cooling_drift` processed separately | Keeps direction semantics clean |

After filtering, up to **80,000 samples per direction** (warming + cooling) are drawn at random (seed 42) to keep training tractable while balancing the two modes.

Total training data: up to **160,000 rows** (80K warming + 80K cooling).

### Step 2: Feature Engineering

For each selected episode start, the following features are computed:

**Core features:**

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `margin` | heating: `start_temp - boundary_temp`; cooling: `boundary_temp - start_temp` | How far (°F) the home is from the comfort boundary. The primary predictor of drift time. |
| `log_margin` | `log(margin + 0.1)` | Log-transformed margin. Since drift time scales roughly linearly with margin (Newton's law), the log transform captures this relationship and compresses the range. |
| `is_heating` | `1` if heating direction, `0` if cooling | Mode flag; heating and cooling homes may have asymmetric dynamics |
| `signed_thermal_drive` | heating: `start_temp - outdoor_temp`; cooling: `outdoor_temp - start_temp` | Positive = outdoor is fighting the drift (slowing it down). Negative = outdoor is accelerating drift toward the boundary. |
| `outdoor_temp` | raw °F | Absolute outdoor temperature; correlated with season and drive direction |
| `start_temp` | indoor temp at episode start (°F) | Absolute starting temperature |
| `boundary_temp` | comfort limit (°F) | The target boundary temperature |

**Time features (cyclical encoding):**

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `hour_sin` | `sin(2π × hour / 24)` | Time of day (smooth circular) |
| `hour_cos` | `cos(2π × hour / 24)` | Time of day (smooth circular) |
| `month_sin` | `sin(2π × month / 12)` | Season (smooth circular) |
| `month_cos` | `cos(2π × month / 12)` | Season (smooth circular) |
| `hour` | integer 0–23 | Raw hour (alongside cyclical, for the model to use directly) |
| `month` | integer 1–12 | Raw month |
| `day_of_week` | integer 0 (Mon)–6 (Sun) | Weekday vs. weekend behavior |

**Total: 14 features** (`DRIFT_FEATURES` constant in `run_xgboost_baselines.py`, lines 320–335).

**Label:**
```
log_duration = log(time_to_boundary_min)
```

The model predicts log-minutes, not raw minutes. This is essential because drift times span a wide range and a log transform makes the residuals more normally distributed.

### Note on `signed_thermal_drive` Sign Convention

The sign convention for drift differs from active-time features:

- **Active time**: `signed_thermal_drive = outdoor - indoor` for heating (outdoor cold helps HVAC)
- **Drift time**: `signed_thermal_drive = indoor - outdoor` for heating (outdoor cold *accelerates* passive cooling drift, which is bad for us — it means the home drifts faster toward the boundary)

A positive `signed_thermal_drive` in the drift model means outdoor is working against us (driving the home toward the boundary faster).

---

## 4. Model Architecture: GBM (Gradient Boosted Trees)

**Function:** `fit_drift_xgb_artifact()` — `scripts/run_xgboost_baselines.py` line 768

The model is a **gradient boosted regression tree**, using whichever backend is available (LightGBM → XGBoost → scikit-learn HistGradientBoosting). The training uses the same hyperparameters as the active-time model:

| Hyperparameter | Value |
|----------------|-------|
| `max_depth` | 6 |
| `n_estimators` | 200 |
| `learning_rate` | 0.1 |
| `min_samples_leaf` | 10 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.1 (L1) |
| `reg_lambda` | 1.0 (L2) |

**Target:** `log(time_to_boundary_min)` — log-transformed duration in minutes.

**Loss:** Mean squared error on log-minutes.

### Why Log-Space?

Homes vary 20× in thermal response speed (thermal mass, insulation quality, climate zone). Training on raw minutes causes the model to minimize high-error predictions for slow homes (outliers dominate). In log-space, a 2× prediction error is treated the same whether the true duration is 10 minutes or 200 minutes — which matches the controller's actual needs.

### No Per-Home Correction

Unlike the active-time model, **the drift model has no per-home residual correction**. The artifact does not store `home_residuals` or `home_mode_residuals`. The global GBM must generalize to all homes without home-specific tuning.

This is a notable difference from the active-time model (which uses a hybrid global + per-home residual approach). The reasoning is that drift is driven primarily by physics (margin, thermal drive) that the global model can capture, whereas active HVAC time depends heavily on system-specific factors (compressor capacity, duct losses) that are harder to capture globally.

---

## 5. Model Artifact

**Saved to:** `models/drift_time_xgb.pkl`

The artifact is a Python `dict` serialized with `pickle`. Fields:

| Key | Type | Value |
|-----|------|-------|
| `version` | `int` | `1` |
| `model_type` | `str` | `"gbm_drift_time_to_boundary"` (validated on load) |
| `backend` | `str` | `"lightgbm"`, `"xgboost"`, or `"sklearn"` |
| `model` | GBM estimator | The trained model object |
| `feature_cols` | `list[str]` | The 14 feature names in training order |
| `predict_log` | `bool` | `True` — model outputs log(minutes) |
| `min_duration` | `float` | `5.0` — minimum clamp value |
| `max_duration` | `float` | `480.0` — maximum clamp value (8 hours) |
| `n_train_rows` | `int` | Number of training samples |

**Loading:**

```python
from mpc.model_interfaces import XGBFirstPassagePredictor

predictor = XGBFirstPassagePredictor.from_model_files(
    drift_model_path="models/drift_time_xgb.pkl"
)
```

The loader (`_GBMArtifact.load`) validates that `model_type == "gbm_drift_time_to_boundary"` and raises `ValueError` if a wrong artifact is passed.

---

## 6. Prediction Interface

**Class:** `XGBFirstPassagePredictor` in `mpc/model_interfaces.py`

**Method:** `predict_drift_time()`

```python
def predict_drift_time(
    self,
    current_temp: float,     # Current indoor temperature (°F)
    boundary_temp: float,    # Comfort boundary the home must not cross (°F)
    outdoor_temp: float,     # Representative outdoor temperature over the forecast interval
    timestamp: pd.Timestamp, # Current time (for time features)
    home_id: str | None,     # Ignored — drift model has no per-home correction
    direction: Direction,    # "heating" or "cooling"
) -> float:                  # Predicted minutes before indoor reaches boundary_temp
```

### Prediction Steps (line 167–188)

1. **Compute margin**: `margin = max(current - boundary, 0)` for heating; `max(boundary - current, 0)` for cooling.
2. **Early return**: If `margin <= 1e-6`, the home is already at or past the boundary → return `0.0`.
3. **Build feature row**: Calls `_drift_feature_row()` (line 248) with 14 features.
4. **GBM inference**: `model.predict(X)` returns raw log-minutes.
5. **Exponentiate and clip**: `exp(log_pred)`, clamped to `[MIN_DRIFT_MINUTES=5, MAX_DRIFT_MINUTES=480]`.

### Output Bounds

| Bound | Value | Reason |
|-------|-------|--------|
| `MIN_DRIFT_MINUTES` | `5.0` | One 5-minute thermostat control step; predicting shorter would be meaningless |
| `MAX_DRIFT_MINUTES` | `480.0` | 8 hours — matches the training data filter. Predictions longer than this are clamped rather than extrapolating into anomaly territory |

---

## 7. Integration into the MPC Controller

**File:** `mpc/peak_mpc.py`, `PeakAwareMPCController.decide()` (line 144)

The drift model is called exclusively during the **peak window** phase. The decision loop is:

```
During peak window:
    1. Compute minutes_to_peak_end
    2. Get representative outdoor temp over [now, peak_end] from forecast
    3. Call predict_drift_time(current_temp, drift_boundary, outdoor_temp, ...)
    4. If margin == 0 (already at boundary):   → PEAK_MAINTAIN
    5. If drift_minutes >= minutes_to_peak_end + safety_margin: → PEAK_COAST (safe)
    6. Otherwise:                                → PEAK_COAST (defer — wait for boundary)
```

### Mode-Specific Drift Boundaries

| Mode | Storage Target | Drift Boundary | Direction |
|------|---------------|----------------|-----------|
| `heating` | `comfort_upper_f` | `comfort_lower_f` | `"heating"` |
| `cooling` | `comfort_lower_f` | `comfort_upper_f` | `"cooling"` |

In heating mode the home was pre-heated to the upper bound; drift means falling toward the lower bound. In cooling mode the home was pre-cooled to the lower bound; drift means rising toward the upper bound.

### Drift Safety Margin

`MPCConfig.drift_safety_margin_minutes` (default: `10.0`)

The coast condition is:

```python
drift_minutes >= minutes_to_peak_end + drift_safety_margin_minutes
```

This 10-minute buffer compensates for prediction uncertainty. If the model predicts 70 minutes of drift and the peak has 60 minutes left, the controller does **not** coast (`70 < 60 + 10 = 70` is False, barely). This prevents cutting it too close when the model is imprecise.

### Deferred HVAC Logic

When the drift prediction says the home will reach the boundary before peak ends (short drift), the controller still does **not** immediately activate HVAC. It issues `peak_coast` with a "defer" reason and waits for the home to actually reach the boundary (caught by the `_at_or_past_boundary` check on a future step). This keeps peak runtime low even when predictions are imprecise.

### Forecast Integration

`representative_outdoor_temp()` collapses the forecast over `[now, peak_end]` into a single value by sampling at hourly intervals and taking the mean. This single value is passed to `predict_drift_time()` as `outdoor_temp`. If no forecast is provided, the current sensor reading is used as a fallback.

---

## 8. Performance Evaluation

### Training Setup

The drift model trains on up to 160,000 samples (80K warming + 80K cooling) from `drift_episodes.parquet`. The training uses a random 70/30 split at the episode level.

The drift model **does not** have a separate reported evaluation in `run_xgboost_baselines.py` — the `main()` function trains and saves the drift artifact after the active-time model evaluation tables, but no test-set metrics are printed for the drift model. This is a known gap: the evaluation framework (`compute_metrics`, `print_results`) was built for active-time models and has not been applied to drift predictions.

### What Is Known About Drift Prediction Difficulty

Several factors make drift prediction hard or easy:

**Easier cases:**
- Large margin (home is far from the boundary) → more time, easier to predict directionally
- Strong thermal drive in one direction → drift rate is predictable
- Well-insulated home with slow dynamics → drift is slow and steady

**Harder cases:**
- Small margin (home is close to the boundary) → even small errors in predicted drift rate matter
- Mild outdoor temperature (small thermal drive) → drift is slow and noisy; quantization artifacts dominate
- Integer-quantized temperature sensors (Ecobee reports whole °F only): drift episodes where the boundary is 1°F away produce highly uncertain time estimates due to quantization noise

### Relationship to Active-Time Model Performance

For context, the **active-time** model (which uses the same GBM architecture and log-transform approach) achieves:

| Model | MAE | Median AE | P90 |
|-------|-----|-----------|-----|
| Global GBM (no home info) | 20.8 min | 11.0 min | 50.9 min |
| Global GBM + Home Encoding | 18.8 min | 9.6 min | 46.4 min |
| **Hybrid GBM (active-time)** | **18.7 min** | **9.5 min** | **46.6 min** |

The drift model uses the same architecture but with 14 features (vs. 15 for active-time), no per-home correction, and a wider duration range (5–480 min vs. 5–240 min). Expect drift MAE to be broadly comparable but with higher variance for short-margin episodes.

### MPC-Level Evaluation (Indirect)

The drift model has been validated indirectly through MPC controller tests (`tests/test_peak_mpc.py`). Key test scenarios:

| Test | Setup | Expected Outcome |
|------|-------|-----------------|
| Long drift (240 min predicted), 2h peak remaining | `drift_minutes=240`, peak ends in 120 min, margin=10 min | `peak_coast` (safe) |
| Short drift (10 min predicted), peak has 2h left | `drift_minutes=10`, peak ends in 120 min | `peak_coast` (defer) |
| At boundary during peak | `indoor_temp == comfort_lower_f` | `peak_maintain` (boundary check overrides drift) |
| Long drift predicted but home is at boundary | `drift_minutes=240`, `indoor_temp == comfort_lower_f` | `peak_maintain` (boundary wins) |
| Forecast temperature passed correctly | Forecast series provided | `outdoor_temp` in drift call equals mean of forecast over peak window |

These tests use a `FakePredictor` with configurable `drift_minutes` — they verify the controller's decision logic, not the drift model's accuracy.

---

## 9. Key Design Decisions

### Why No Per-Home Residual for Drift?

The active-time model uses a two-stage approach (global GBM + per-home mean residual correction) because homes vary 20× in HVAC system power and duct efficiency — factors that are hard to capture globally.

Passive drift is governed more purely by physics: insulation quality, thermal mass, and outdoor driving force. The features `signed_thermal_drive`, `outdoor_temp`, and `margin` directly encode the physics. The GBM can learn the thermal response curve without needing a per-home offset.

This also makes the drift model more suitable for cold-start (new home with no history) — no per-home term means no cold-start penalty.

### Why 8-Hour Max Duration?

The active-time model caps at 4 hours (240 min) because setpoint-response episodes longer than 4 hours are typically equipment failures or vacation mode. Drift is a slower process — a well-insulated home might take 6–8 hours to drift from 72°F to 68°F in mild weather. The 8-hour (480 min) cap reflects this, while still filtering extreme outliers.

### Why Train Only on `crossed_boundary == True`?

Including episodes that didn't cross the boundary would introduce **censored labels** — we'd know the drift took at least X minutes, but not how much longer it would have gone. Standard regression on censored data underestimates true duration. By filtering to confirmed crossings only, every label is exact.

This does introduce selection bias: homes with very strong insulation (that rarely drift far enough to cross a comfort boundary) are underrepresented. The practical implication is that the model may slightly underestimate drift time for very well-insulated homes.

### Why `timestep_idx == 0` Only?

Including later timesteps from the same episode would:
1. Inflate the training set with highly correlated samples
2. Introduce data leakage (a model that sees `timestep_idx=5` knows the episode lasted at least 25 minutes)

Using only episode starts gives independent samples with clean labels.

---

## 10. File Map

| File | Role |
|------|------|
| `data/drift_episodes.parquet` | Raw drift episode data (59 MB) |
| `scripts/run_xgboost_baselines.py` | Training: `create_drift_samples()`, `fit_drift_xgb_artifact()`, `save_drift_xgb_artifact()` |
| `models/drift_time_xgb.pkl` | Serialized trained artifact |
| `mpc/model_interfaces.py` | `_drift_feature_row()`, `predict_drift_time()`, output clamping constants |
| `mpc/peak_mpc.py` | Controller that calls `predict_drift_time()` during peak windows |
| `tests/test_peak_mpc.py` | Integration tests using `FakePredictor` |
| `docs/MPC_EXPLAINED.md` | Full MPC system reference (drift covered in §7) |
| `docs/MPC_CONTROLLER_SRS.md` | Formal specification (drift in §6, FR-6) |

---

## 11. How to Retrain

```bash
python scripts/run_xgboost_baselines.py \
    --model-output models/active_time_xgb.pkl \
    --drift-model-output models/drift_time_xgb.pkl
```

This runs all five active-time model variants, saves the best (hybrid) active-time model, then trains and saves the drift model. Training takes a few minutes with LightGBM, longer with sklearn.

To skip saving the active-time model and only retrain drift:

```bash
python scripts/run_xgboost_baselines.py \
    --model-output "" \
    --drift-model-output models/drift_time_xgb.pkl
```

---

## 12. Known Gaps and Future Work

1. **No held-out evaluation metrics**: The drift model is trained and saved without reporting test-set MAE, RMSE, or median error. Adding a 70/30 split evaluation for drift (analogous to the active-time tables in `main()`) would make performance visible.

2. **No per-home correction**: If homes show systematic drift biases (e.g., one home consistently drifts 30% slower than the global model predicts), adding a per-home mean residual correction (identical to the active-time approach) could reduce MPC-level comfort violations.

3. **Quantile predictions**: The MPC uses a fixed `drift_safety_margin_minutes=10` to handle prediction uncertainty. Replacing this with a learned P80 or P90 drift time would give a principled, per-episode safety margin. Quantile regression (e.g., `objective="quantile"` in LightGBM) could provide this directly.

4. **Censored data**: Using survival analysis or censored regression on episodes that didn't cross the boundary (currently filtered out) could recover useful signal and reduce selection bias toward poorly insulated homes.

5. **Dynamic forecast temperature**: The current controller passes a single mean outdoor temperature to `predict_drift_time()`. If the peak window is long (3 hours) with large forecast variation, using the temperature at the *end* of the window (worst case) rather than the mean might give more conservative and reliable coasting decisions.
