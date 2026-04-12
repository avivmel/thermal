# Peak-Aware MPC Controller: Complete Technical Reference

This document explains every part of the MPC system — what it is, why it exists, how every piece
works, and how the pieces fit together.

---

## 1. What the MPC Does

The goal is to **minimize HVAC electricity consumption during peak pricing hours** (e.g., 5–8 PM)
while keeping the indoor temperature inside a user-defined comfort band.

The strategy is:

1. **Before peak:** Pre-condition the home — heat it up toward the top of the comfort band (winter)
   or cool it down toward the bottom (summer). This stores "thermal energy" in the building mass.
2. **During peak:** Let the home coast passively. The pre-stored thermal buffer gives the building
   time to drift before it hits the uncomfortable boundary. HVAC stays off as long as possible.
3. **If coasting fails:** When drift predictions say the home will hit the comfort limit before peak
   ends, allow minimal HVAC to maintain the boundary — nothing more.

The controller never precomputes a day-long schedule. It re-decides every control step (default: 5
minutes) based on current sensor data and fresh predictions.

---

## 2. File Map

```
mpc/
├── peak_mpc.py          # Controller logic — config, state, command types, decision algorithm
└── model_interfaces.py  # Predictor interface + XGBoost implementation

tests/
└── test_peak_mpc.py     # Unit tests using a fake predictor

docs/
└── MPC_CONTROLLER_SRS.md  # Software requirements specification (formal spec)
```

---

## 3. Data Types (`peak_mpc.py`)

### `PeakWindow`

```python
@dataclass(frozen=True)
class PeakWindow:
    start: pd.Timestamp
    end: pd.Timestamp
```

Represents one peak pricing interval. `contains(timestamp)` returns `True` if
`start <= timestamp < end` (end-exclusive).

Validation: `end` must be strictly after `start`. Multiple windows must not overlap — checked at
config construction time.

---

### `MPCConfig`

All configuration lives here. Immutable (`frozen=True`).

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `peak_windows` | `list[PeakWindow]` | required | One or more peak intervals. Zero windows = always normal. |
| `comfort_lower_f` | `float` | required | Lower comfort bound (°F). No setpoint will go below this. |
| `comfort_upper_f` | `float` | required | Upper comfort bound (°F). No setpoint will go above this. |
| `normal_heat_setpoint_f` | `float \| None` | `None` | Setpoint during normal operation in heating mode. |
| `normal_cool_setpoint_f` | `float \| None` | `None` | Setpoint during normal operation in cooling mode. |
| `control_step_minutes` | `int` | `5` | How often `decide()` is called. Not enforced internally — caller's responsibility. |
| `horizon_minutes` | `int` | `360` | How far ahead to look for an upcoming peak window (6 hours). |
| `precondition_margin_minutes` | `float` | `10.0` | Safety buffer added to predicted active time before deciding to start preconditioning. |
| `drift_safety_margin_minutes` | `float` | `10.0` | Extra cushion added to `minutes_to_peak_end` when checking if coasting is safe. |
| `min_precondition_gap_f` | `float` | `0.5` | If the temperature gap to the storage target is smaller than this, skip preconditioning. |
| `max_precondition_lead_minutes` | `float` | `240.0` | If the peak is farther away than this, return normal (no pre-conditioning 4+ hours early). |

**Validation** (raises `ValueError`):
- `comfort_lower_f >= comfort_upper_f`
- `control_step_minutes <= 0`
- `horizon_minutes < 0`
- Any margin/limit is negative
- Two peak windows overlap

---

### `ThermostatState`

A snapshot of sensor data at one control step. All values are read-only.

```python
@dataclass(frozen=True)
class ThermostatState:
    timestamp:      pd.Timestamp
    indoor_temp_f:  float        # Current indoor temperature (°F)
    outdoor_temp_f: float        # Current outdoor temperature (°F)
    system_running: bool         # Is HVAC actively running right now?
    mode:           "heating" | "cooling"
    home_id:        str | None   # Optional — used for per-home residual correction
```

`mode` determines the thermal storage direction:
- `"heating"` → pre-heat toward `comfort_upper_f`, coast down toward `comfort_lower_f`
- `"cooling"` → pre-cool toward `comfort_lower_f`, coast up toward `comfort_upper_f`

---

### `ThermostatCommand`

The return value of `decide()`. Tells the thermostat what setpoints to use and why.

```python
@dataclass(frozen=True)
class ThermostatCommand:
    heat_setpoint_f: float | None   # New heating setpoint. None means "don't change"
    cool_setpoint_f: float | None   # New cooling setpoint. None means "don't change"
    phase:           Phase          # One of: normal, precondition, peak_coast, peak_maintain
    reason:          str            # Human-readable explanation of this decision
```

In heating mode, `cool_setpoint_f` is always `None`, and vice versa. Only the active-mode setpoint
is changed.

---

### Phase Semantics

| Phase | When | What setpoint is issued |
|-------|------|------------------------|
| `normal` | No peak nearby, or too early to precondition | `normal_heat/cool_setpoint_f` — normal user preference |
| `precondition` | Pre-peak, and it's time to start storing thermal energy | Heating: `comfort_upper_f` (heat to top of band); Cooling: `comfort_lower_f` (cool to bottom) |
| `peak_coast` | Inside peak window, home predicted to coast safely | Heating: `comfort_lower_f` (HVAC won't call unless temp drops to lower bound); Cooling: `comfort_upper_f` |
| `peak_maintain` | Inside peak, home at/near the uncomfortable boundary | Same setpoints as `peak_coast` — but the reason is that we've reached the limit, not that we're safely coasting |

`peak_coast` and `peak_maintain` set the same setpoints — the difference is the `phase` label and
the `reason` string, which enables evaluation to distinguish between "coasting successfully" and
"had to maintain the boundary."

---

## 4. Weather Forecast Handling

The predictor needs an outdoor temperature value for its predictions. The controller accepts a
forecast object so it can use a representative temperature over the relevant future interval, rather
than just the current reading.

### `ForecastInput` types

```python
ForecastInput = Union[WeatherForecast, Callable[[pd.Timestamp], float], pd.Series, None]
```

Three ways to provide a forecast:
- **`WeatherForecast` protocol** — any object with `outdoor_temp_at(timestamp) -> float`
- **callable** — wrapped in `_CallableWeatherForecast` internally
- **`pd.Series`** — wrapped in `SeriesWeatherForecast` (see below)
- **`None`** — falls back to `state.outdoor_temp_f` at every prediction

### `SeriesWeatherForecast`

A thin wrapper around a `pd.Series` indexed by `pd.Timestamp`.

```python
class SeriesWeatherForecast:
    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float
```

**Lookup behavior (non-interpolating mode):** Uses `searchsorted` to find the last known value at
or before `timestamp`. This is nearest-prior (step) interpolation — the value doesn't change
between forecast points.

**Interpolating mode:** `interpolate=True` reindexes the series to include the target timestamp and
uses pandas `interpolate(method="time")` for linear interpolation between points.

If the timestamp is before the first forecast point, a `KeyError` is raised. The controller catches
this and falls back to `state.outdoor_temp_f`.

### `representative_outdoor_temp()`

```python
def representative_outdoor_temp(
    forecast: ForecastInput,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fallback: float,
) -> tuple[float, str]
```

Collapses a forecast interval `[start, end]` into a single representative value for the predictor.

**How it works:**
1. Generates sample timestamps at hourly intervals within `[start, end]`, always including both
   endpoints (`_sample_times()`).
2. Queries the forecast at each sample point.
3. Returns the **mean** of all finite values.
4. If the forecast is `None`, lookup fails, or no finite values are found, falls back to `fallback`
   (current outdoor temp).

Also returns a `reason` string explaining which path was taken, which ends up in
`ThermostatCommand.reason`.

---

## 5. The Decision Algorithm (`PeakAwareMPCController.decide()`)

Called every control step. Stateless — same inputs always produce the same output.

```
decide(state, forecast=None):
```

### Step 1 — Validate input

```python
_validate_plausible_temp(state.indoor_temp_f, "indoor_temp_f")
```

Rejects temperatures outside `[-50, 150]°F` — a sanity check against sensor errors.

---

### Step 2 — Find the relevant peak window

```python
window = _active_or_next_peak_window(state.timestamp, config.peak_windows, config.horizon_minutes)
```

**Logic of `_active_or_next_peak_window()`:**
- Iterates through peak windows (sorted by start time).
- If `state.timestamp` is inside a window (`start <= now < end`), returns it immediately.
- Otherwise, returns the first window whose `start` is within `horizon_minutes` of now.
- Returns `None` if no window is active or upcoming within the horizon.

If `window is None` → **return `normal_command("no peak window in horizon")`**.

---

### Step 3 — Determine mode targets

```python
storage_target, drift_boundary, direction = _mode_targets(state.mode, config)
```

| Mode | Storage target | Drift boundary | Direction |
|------|---------------|----------------|-----------|
| `heating` | `comfort_upper_f` | `comfort_lower_f` | `"heating"` |
| `cooling` | `comfort_lower_f` | `comfort_upper_f` | `"cooling"` |

- **Storage target**: where we pre-condition to (the "charged" side of the comfort band)
- **Drift boundary**: the uncomfortable side — the limit we cannot cross

---

### Step 4 — Branch on whether we're inside or before the peak

---

#### Branch A: Inside the peak window (`window.contains(state.timestamp)`)

**Goal:** Avoid HVAC runtime. Only allow it if the home will definitely hit the comfort limit
before peak ends.

```
minutes_to_peak_end = window.end - now
forecast_temp = representative_outdoor_temp(forecast, now, window.end, fallback)
drift_minutes = predictor.predict_drift_time(
    current_temp, drift_boundary, forecast_temp, timestamp, home_id, direction
)
```

**Check 1 — Boundary already reached?**

```python
if _at_or_past_boundary(state.indoor_temp_f, drift_boundary, direction):
    return peak_maintain_command(...)
```

`_at_or_past_boundary` in heating mode: `current_temp <= drift_boundary`.
In cooling mode: `current_temp >= drift_boundary`.

This check runs **first** — if the home is already at the limit, there's nothing to coast.

**Check 2 — Can we coast safely?**

```python
if drift_minutes >= minutes_to_peak_end + config.drift_safety_margin_minutes:
    return peak_coast_command("predicted drift stays within comfort through peak")
```

The `drift_safety_margin_minutes` (default 10) adds a buffer. If the model predicts 70 minutes of
drift time and peak ends in 50 minutes, we coast only if `70 >= 50 + 10 = 60`. This prevents
cutting it too close.

**Default (short drift, not yet at boundary):**

```python
return peak_coast_command("defer HVAC until closer to comfort boundary")
```

Even when coasting is predicted to fail, the controller still waits. It defers HVAC until the home
actually reaches the boundary (caught by Check 1 on a future step). This avoids triggering HVAC
prematurely based on an imprecise prediction.

---

#### Branch B: Before the peak window

**Goal:** Decide whether to start preconditioning.

**Check 1 — Is peak too far away?**

```python
if minutes_to_peak_start > config.max_precondition_lead_minutes:
    return normal_command("peak window too far away")
```

Default `max_precondition_lead_minutes = 240` (4 hours). No preconditioning starts more than 4
hours before peak.

**Predict how long active HVAC will be needed:**

```python
forecast_temp = representative_outdoor_temp(forecast, now, window.start, fallback)
active_minutes = predictor.predict_active_time(
    current_temp, storage_target, forecast_temp, timestamp, system_running, home_id, direction
)
```

This asks: "if HVAC turns on right now, how many minutes until the home reaches the storage
target?"

**Check 2 — Already near the storage target?**

```python
gap_to_storage_target = _directional_gap(state.indoor_temp_f, storage_target, direction)
if gap_to_storage_target < config.min_precondition_gap_f:
    return normal_command("already near storage target")
```

`_directional_gap` is the unsigned gap in the right direction:
- Heating: `max(storage_target - current_temp, 0)`
- Cooling: `max(current_temp - storage_target, 0)`

If the gap is < `min_precondition_gap_f` (default 0.5°F), the home is already charged. No need to
run HVAC.

**Check 3 — Is it time to start?**

```python
latest_start_minutes = active_minutes + config.precondition_margin_minutes
if minutes_to_peak_start <= latest_start_minutes:
    return precondition_command(...)
```

`latest_start_minutes` = predicted HVAC runtime + margin (default 10 min). The controller waits
until the remaining time before peak is ≤ this. At that point, if HVAC doesn't start now, the home
won't finish preconditioning before peak.

**Example:**
- Peak starts at 17:00. It's 15:50 (70 min away).
- Predicted active time = 55 min. Latest start = 55 + 10 = 65 min.
- `70 > 65` → return `normal` (not yet).
- At 16:00 (60 min away): `60 <= 65` → return `precondition`.

**Default:**

```python
return normal_command("not time to precondition yet")
```

---

## 6. Predictor Interface (`model_interfaces.py`)

The controller depends on an abstract interface — not on any concrete model. This enables unit
testing with fake predictors and future model swaps.

### `FirstPassagePredictor` (abstract base)

```python
class FirstPassagePredictor:
    def predict_active_time(
        self,
        current_temp: float,    # Current indoor temperature (°F)
        target_temp: float,     # Target temperature to reach (storage target)
        outdoor_temp: float,    # Representative outdoor temp over the interval
        timestamp: pd.Timestamp,
        system_running: bool,   # Is HVAC already on? (50% faster if yes)
        home_id: str | None,    # For per-home correction
        direction: Direction,   # "heating" | "cooling"
    ) -> float:
        ...  # Returns minutes of active HVAC needed

    def predict_drift_time(
        self,
        current_temp: float,    # Current indoor temperature (°F)
        boundary_temp: float,   # Comfort boundary (the limit we can't cross)
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        home_id: str | None,
        direction: Direction,
    ) -> float:
        ...  # Returns minutes before passive drift reaches boundary_temp
```

### Output Bounds (Constants)

```python
MIN_ACTIVE_MINUTES = 5.0    # Never predict < 5 min active time
MAX_ACTIVE_MINUTES = 240.0  # Cap at 4 hours
MIN_DRIFT_MINUTES  = 5.0    # Never predict < 5 min drift time
MAX_DRIFT_MINUTES  = 480.0  # Cap at 8 hours
```

---

## 7. XGBoost Predictor (`XGBFirstPassagePredictor`)

The production implementation. Uses two trained XGBoost models loaded from pickle files.

### Model Artifacts

```python
DEFAULT_ACTIVE_MODEL_PATH = Path("models/active_time_xgb.pkl")
DEFAULT_DRIFT_MODEL_PATH  = Path("models/drift_time_xgb.pkl")
```

Each `.pkl` file is a dict containing:
- `"model"` — the trained XGBoost regressor
- `"model_type"` — string tag, validated on load (`"hybrid_gbm_active_time_to_target"` / `"gbm_drift_time_to_boundary"`)
- `"feature_cols"` — ordered list of feature names the model was trained on
- `"n_train_rows"` — how many rows were used for training (metadata)
- `"min_duration"` / `"max_duration"` — clamping bounds from training data
- `"home_mode_residuals"` — dict `{(home_id, mode_int): residual}` for per-home+mode correction
- `"home_residuals"` — dict `{home_id: residual}` for per-home correction (fallback)

### Prediction Flow

Both models predict in **log-minutes** space (the model outputs `log(minutes)`). This is because
home response times vary 20× across homes, and the log transform makes the distribution much more
normal for regression.

**Active time prediction (`predict_active_time`):**

1. Compute `gap = max(target - current, 0)` (heating) or `max(current - target, 0)` (cooling).
2. If `gap <= 1e-6`, return `0.0` immediately (already at target).
3. Build feature row (see Feature Engineering below).
4. Call `model.predict(X)` → raw log-minutes.
5. Apply per-home residual correction if `home_id` is provided:
   - Try `(home_id, mode_int)` key in `home_mode_residuals` first.
   - Fall back to `home_id` key in `home_residuals`.
   - Add the residual to the log prediction.
6. `exp(log_pred)` → clip to `[min_duration, max_duration]`.

**Drift time prediction (`predict_drift_time`):**

1. Compute `margin = max(current - boundary, 0)` (heating) or `max(boundary - current, 0)` (cooling).
2. If `margin <= 1e-6`, return `0.0` immediately (already at boundary).
3. Build feature row.
4. Call `model.predict(X)` → log-minutes.
5. No per-home correction for drift (drift model is global only).
6. `exp(log_pred)` → clip to `[min_duration, max_duration]`.

### Feature Engineering

**Active time features** (15 features):

| Feature | How computed | Why |
|---------|-------------|-----|
| `log_gap` | `log(gap + 0.1)` | Log-gap is the primary predictor; gap = \|target - current\| |
| `abs_gap` | `gap` | Linear gap as a complement |
| `is_heating` | `1.0` if heating, `0.0` if cooling | Mode affects speed (heating 28% faster) |
| `system_running` | `1.0` if HVAC already on | Already-on = ~50% faster completion |
| `signed_thermal_drive` | `(outdoor - indoor)` if heating; `-(outdoor - indoor)` if cooling | Positive = outdoor helps HVAC; negative = outdoor fights it |
| `outdoor_temp` | raw °F | Absolute outdoor temperature |
| `start_temp` | indoor temp (°F) | Starting point affects efficiency |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour of day | Time-of-day effects (solar gain, etc.) |
| `month_sin`, `month_cos` | Cyclical encoding of month | Seasonal effects |
| `hour` | raw integer (0–23) | Direct hour (alongside cyclical) |
| `month` | raw integer (1–12) | Direct month |
| `day_of_week` | integer (0=Monday) | Weekday vs. weekend patterns |
| `indoor_humidity` | hardcoded `50.0` | Placeholder (Ecobee data lacks humidity) |
| `outdoor_humidity` | hardcoded `50.0` | Placeholder |

**Drift time features** (14 features):

| Feature | How computed | Why |
|---------|-------------|-----|
| `margin` | `\|current - boundary\|` in the right direction | How far from the boundary we are |
| `log_margin` | `log(margin + 0.1)` | Log-transformed margin |
| `is_heating` | mode flag | |
| `signed_thermal_drive` | `-`(outdoor - indoor) if heating; `+`(outdoor - indoor) if cooling | Positive = outdoor accelerates drift toward boundary |
| `outdoor_temp` | raw °F | |
| `start_temp` | indoor temp | |
| `boundary_temp` | the comfort boundary (°F) | Absolute value of the limit |
| `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | cyclical time features | |
| `hour`, `month`, `day_of_week` | raw time features | |

**Note on `signed_thermal_drive` difference:**
- For **active time**: positive drive means outdoor temp helps HVAC (e.g., cold outside helps
  cooling), so `signed_thermal_drive = outdoor - indoor` for heating, `-(outdoor - indoor)` for
  cooling.
- For **drift time**: positive drive means outdoor temp accelerates passive drift toward the
  boundary (bad for us), so the sign is flipped: `-(outdoor - indoor)` for heating,
  `+(outdoor - indoor)` for cooling.

---

## 8. Per-Home Residual Correction

The active time model uses a **hybrid approach**: a global XGBoost model trained on all homes,
plus per-home (and per-home-per-mode) residual corrections learned from the training set.

This handles the fact that homes vary 20× in thermal response speed. A global model predicts the
average; the residual captures each home's idiosyncratic offset.

**Residual application:**
```python
log_pred_global = model.predict(X)[0]  # Global prediction
log_pred_corrected = log_pred_global + home_mode_residuals[(home_id, mode)]
minutes = exp(log_pred_corrected)
```

In log space, adding a residual is multiplicative in real space: a residual of `+0.3` means the
home takes ~35% longer than average.

**Cold-start (new home, no `home_id` or unknown `home_id`):** No residual is applied. The global
model gives a reasonable estimate, but with higher uncertainty (~30 min MAE vs ~16 min when
`home_id` is known).

---

## 9. Helper Functions

### `_minutes_between(start, end) -> float`
Computes `(end - start).total_seconds() / 60`.

### `_directional_gap(current_temp, target_temp, direction) -> float`
Unsigned gap in the direction of travel:
- Heating: `max(target - current, 0)` — how far below the target we are
- Cooling: `max(current - target, 0)` — how far above the target we are

### `_at_or_past_boundary(current_temp, boundary_temp, direction) -> bool`
- Heating: `current <= boundary` (too cold)
- Cooling: `current >= boundary` (too hot)

### `_clip_optional(value, config) -> float | None`
Clips a setpoint to `[comfort_lower_f, comfort_upper_f]`. If value is `None`, returns `None`.
Ensures `normal_heat/cool_setpoint_f` can't accidentally escape the comfort band.

### `_sample_times(start, end) -> list[pd.Timestamp]`
Generates hourly timestamps between `start` and `end`, always including both endpoints. Used by
`representative_outdoor_temp` to sample the forecast.

---

## 10. Command Construction

Each phase has a private method that constructs the right command for the current mode:

```python
def _normal_command(state, reason):
    # Heating: heat_sp = normal_heat_setpoint_f (clipped); cool_sp = None
    # Cooling: cool_sp = normal_cool_setpoint_f (clipped); heat_sp = None

def _precondition_command(state, reason):
    # Heating: heat_sp = comfort_upper_f (pre-heat to top of band)
    # Cooling: cool_sp = comfort_lower_f (pre-cool to bottom of band)

def _peak_coast_command(state, reason):
    # Heating: heat_sp = comfort_lower_f (HVAC won't call unless temp drops to lower bound)
    # Cooling: cool_sp = comfort_upper_f (HVAC won't call unless temp rises to upper bound)

def _peak_maintain_command(state, reason):
    # Same setpoints as peak_coast — the difference is the phase label
    # Heating: heat_sp = comfort_lower_f
    # Cooling: cool_sp = comfort_upper_f
```

**Why `peak_coast` and `peak_maintain` produce the same setpoints:**
Setting the setpoint to the comfort boundary means the thermostat will not call for HVAC until the
indoor temperature actually hits the boundary. Both phases achieve this. The distinction is purely
semantic — `peak_coast` means "we're comfortably inside the band and expect to coast"; `peak_maintain`
means "we've hit the limit and are holding it."

---

## 11. Key Design Decisions

### No open-loop power commands
The CityLearn DR experiment (documented in `CITYLEARN_DR_RESULTS.md`) proved that setting
`action = 0.7` (70% of max HVAC power) causes massive overheating because it ignores whether
heating is actually needed. This controller exclusively uses **thermostat setpoints** — the HVAC
turns on/off based on whether indoor temp is above/below the setpoint. This is how real thermostats
work.

### Stateless receding-horizon decisions
No schedule is precomputed. Every call to `decide()` independently computes the right action for
the current moment. This means:
- The controller gracefully handles sensor glitches, forecast errors, and prediction uncertainty
- No drift between planned and actual behavior
- Easy to test: same inputs always produce same outputs

### First-passage framing (not trajectory prediction)
Instead of predicting a temperature trajectory 30 steps ahead, the models answer two simpler
questions:
1. "How many minutes until HVAC gets us to temperature X?" (active time)
2. "How many minutes until the home passively drifts to temperature Y?" (drift time)

These single-number answers are all the controller needs to make its two decisions (start
preconditioning? can we coast?). This avoids compounding trajectory prediction errors.

### Log-space prediction
Home response times span 6–120 min/°F thermal resistance — a 20× range. Training regression
models on raw minutes leads to high-error models for fast homes (outlier-dominated). Log-transforming
the target compresses this range and makes residuals more normally distributed, dramatically
improving predictions for fast-responding homes.

### Deferred HVAC during peak
When the model predicts coasting will fail (drift time < peak remaining + margin), the controller
does **not** immediately turn on HVAC. Instead it issues `peak_coast` with a "defer" reason and
waits for the home to actually reach the boundary. This keeps the primary metric (peak runtime
minutes) low even when predictions are imperfect, at the cost of a small comfort violation
tolerance.

---

## 12. Integration with ThermalGym

ThermalGym (documented in `THERMALGYM.md`) is an EnergyPlus-backed simulation environment. It
provides the simulation loop in which the MPC controller runs.

**Connection points:**
- `ThermalEnv.step(action)` takes `{"heat_setpoint": float, "cool_setpoint": float}` — exactly
  what `ThermostatCommand` provides.
- `ThermalEnv` observation dict provides `indoor_temp`, `outdoor_temp`, `hvac_mode`, and
  `timestamp` — the inputs to `ThermostatState`.
- `env.history` DataFrame captures the full episode for computing peak runtime metrics.

**Typical evaluation loop:**

```python
env = thermalgym.ThermalEnv(building="medium_cold_heatpump")
obs = env.reset(date="2017-01-15")

config = MPCConfig(
    peak_windows=[(pd.Timestamp("2017-01-15 17:00"), pd.Timestamp("2017-01-15 20:00"))],
    comfort_lower_f=68.0,
    comfort_upper_f=72.0,
    normal_heat_setpoint_f=70.0,
)
predictor = XGBFirstPassagePredictor.from_model_files()
controller = PeakAwareMPCController(config, predictor)

while not env.done:
    state = ThermostatState(
        timestamp=obs["timestamp"],
        indoor_temp_f=obs["indoor_temp"],
        outdoor_temp_f=obs["outdoor_temp"],
        system_running=obs["hvac_mode"] != "off",
        mode=obs["hvac_mode"] if obs["hvac_mode"] != "off" else "heating",
    )
    cmd = controller.decide(state)
    action = {
        "heat_setpoint": cmd.heat_setpoint_f or 70.0,
        "cool_setpoint": cmd.cool_setpoint_f or 76.0,
    }
    obs = env.step(action)

df = env.history
peak_runtime = df[(df["hour"] >= 17) & (df["hour"] < 20) & (df["hvac_power_kw"] > 0)]
print(f"Peak runtime: {len(peak_runtime) * 5} minutes")
```

---

## 13. Evaluation Metrics

From the SRS, the primary metric is:

```
peak_runtime_minutes = sum(minutes where HVAC is actively calling during peak windows)
```

Secondary diagnostics:
- **Peak runtime reduction** vs. normal thermostat baseline
- **Comfort violation minutes** — time spent outside `[comfort_lower_f, comfort_upper_f]`
- **Max comfort violation** — largest single deviation in °F
- **Pre-peak runtime** — confirms runtime is shifted, not eliminated
- **Total daily runtime** — catches inefficient over-preconditioning

A good outcome: peak runtime drops significantly while total daily runtime stays within ~25% of
baseline and comfort violations remain near zero.

---

## 14. Configuration Examples

### Winter heating, 5–8 PM peak

```python
from mpc.peak_mpc import MPCConfig, PeakWindow
import pandas as pd

# Repeat the peak window for each day
peak_windows = [
    PeakWindow(
        start=pd.Timestamp("2017-01-15 17:00"),
        end=pd.Timestamp("2017-01-15 20:00"),
    )
]

config = MPCConfig(
    peak_windows=peak_windows,
    comfort_lower_f=68.0,
    comfort_upper_f=72.0,
    normal_heat_setpoint_f=70.0,
    normal_cool_setpoint_f=None,
    precondition_margin_minutes=10.0,
    drift_safety_margin_minutes=10.0,
)
```

### Summer cooling, two daily peaks

```python
config = MPCConfig(
    peak_windows=[
        PeakWindow(pd.Timestamp("2017-07-15 14:00"), pd.Timestamp("2017-07-15 16:00")),
        PeakWindow(pd.Timestamp("2017-07-15 19:00"), pd.Timestamp("2017-07-15 21:00")),
    ],
    comfort_lower_f=72.0,
    comfort_upper_f=76.0,
    normal_cool_setpoint_f=74.0,
)
```

### More aggressive preconditioning

```python
config = MPCConfig(
    peak_windows=[...],
    comfort_lower_f=68.0,
    comfort_upper_f=72.0,
    precondition_margin_minutes=20.0,   # Start 20 min earlier than strictly needed
    drift_safety_margin_minutes=15.0,   # Require 15 min of extra drift margin before coasting
    max_precondition_lead_minutes=300.0, # Allow preconditioning up to 5 hours ahead
)
```

---

## 15. Quick Reference: Decision Tree

```
decide(state):
│
├── indoor_temp outside [-50, 150]°F → ValueError
│
├── No peak window within horizon_minutes
│   └── return NORMAL
│
├── Peak window found
│   │
│   ├── We're INSIDE the peak window
│   │   ├── at/past comfort boundary (too cold/hot)
│   │   │   └── return PEAK_MAINTAIN
│   │   ├── drift_minutes >= peak_remaining + safety_margin
│   │   │   └── return PEAK_COAST (safe to coast)
│   │   └── drift too short, not yet at boundary
│   │       └── return PEAK_COAST (defer — wait for boundary)
│   │
│   └── We're BEFORE the peak window
│       ├── minutes_to_peak > max_precondition_lead
│       │   └── return NORMAL (too far away)
│       ├── gap to storage target < min_precondition_gap
│       │   └── return NORMAL (already charged)
│       ├── minutes_to_peak <= active_minutes + precondition_margin
│       │   └── return PRECONDITION (start charging now)
│       └── otherwise
│           └── return NORMAL (not yet time)
```
