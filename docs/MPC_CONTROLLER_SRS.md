# SRS: Peak-Aware First-Passage MPC Controller

## 1. Purpose

Specify a simple MPC controller that intelligently pre-heats or pre-cools a home before a configurable set of peak windows, then minimizes thermostat runtime during those peak windows while keeping indoor temperature inside user comfort bounds.

The controller should use the first-passage prediction interface in `mpc/model_interfaces.py`:

- `predict_active_time(...)`: minutes of active HVAC needed to reach a target temperature.
- `predict_drift_time(...)`: minutes before passive temperature drift reaches a comfort boundary.

This is not a general nonlinear optimizer. It is a receding-horizon, first-passage controller that makes a small number of decisions from predicted active and drift times.

## 2. Scope

### In Scope

- Support any configurable number of peak windows per day.
- Support heating and cooling modes.
- Precondition the home toward the favorable side of the comfort band before peak:
  - Heating season: pre-heat toward the upper comfort bound.
  - Cooling season: pre-cool toward the lower comfort bound.
- During peak, avoid HVAC runtime unless the home is predicted to hit the opposite comfort boundary before the peak window ends.
- Recompute the decision every control step, using the latest indoor temperature, weather forecast, timestamp, system-running state, and optional `home_id`.
- Keep the implementation small: one controller file plus tests or a demo script. No more than three implementation-related files.

### Out of Scope

- Training new prediction models.
- Full cost optimization over arbitrary tariffs.
- Whole-home load forecasting outside HVAC runtime.
- Multi-zone control.
- Occupancy inference.
- Recovery optimization after peak beyond returning to the normal user setpoint.
- CityLearn-specific controller integration in the first implementation, unless added as an optional adapter.

## 3. Key Metric

Primary metric:

```
peak_runtime_minutes = sum(minutes where thermostat/HVAC is actively calling during configured peak windows)
```

The controller should minimize peak runtime subject to comfort constraints. Energy cost, PAR, and total daily HVAC runtime are secondary diagnostics, not the primary objective.

Recommended evaluation metrics:

- Peak runtime minutes per home-day.
- Peak runtime reduction versus normal thermostat baseline.
- Comfort violation minutes.
- Maximum comfort violation magnitude in degrees F.
- Pre-peak runtime minutes, to verify runtime is shifted rather than simply eliminated.
- Total daily runtime, to catch inefficient over-preconditioning.

## 4. Inputs

### Controller Configuration

```python
PeakWindow = tuple[pd.Timestamp, pd.Timestamp]

MPCConfig:
    peak_windows: list[PeakWindow]
    control_step_minutes: int = 5
    horizon_minutes: int = 360
    comfort_lower_f: float
    comfort_upper_f: float
    normal_heat_setpoint_f: float | None
    normal_cool_setpoint_f: float | None
    precondition_margin_minutes: float = 10
    drift_safety_margin_minutes: float = 10
    min_precondition_gap_f: float = 0.5
    max_precondition_lead_minutes: float = 240
```

`peak_windows` must accept zero, one, or many windows. Windows may be non-contiguous. The first implementation should assume windows do not overlap; validation should raise a clear error if they do.

### Runtime State

```python
ThermostatState:
    timestamp: pd.Timestamp
    indoor_temp_f: float
    outdoor_temp_f: float
    system_running: bool
    home_id: str | None
    mode: Literal["heating", "cooling"]
```

The `mode` chooses the thermal storage direction:

- `heating`: pre-heat to `comfort_upper_f`, coast down to `comfort_lower_f`.
- `cooling`: pre-cool to `comfort_lower_f`, coast up to `comfort_upper_f`.

### Weather Forecast

The caller will provide an outdoor temperature forecast covering the controller horizon. The first implementation should keep this interface simple:

```python
WeatherForecast:
    def outdoor_temp_at(timestamp: pd.Timestamp) -> float
```

Acceptable MVP representations:

- A callable: `Callable[[pd.Timestamp], float]`.
- A small wrapper around a `pd.Series` indexed by timestamp.

Forecast lookup should return the nearest available forecast value at or before the requested timestamp, or interpolate if the wrapper implements interpolation. If no forecast value is available for the requested time, the controller may fall back to `state.outdoor_temp_f` and include that fallback in the command reason or debug metadata.

Because `FirstPassagePredictor` accepts one `outdoor_temp` value per prediction, the controller should collapse a forecast interval into one representative value. The MVP can use the mean forecasted temperature over the interval, or nearest-prior lookup at the interval start if only point lookup is implemented.

## 5. Outputs

The controller returns a thermostat-level command rather than a raw equipment-power command:

```python
ThermostatCommand:
    heat_setpoint_f: float | None
    cool_setpoint_f: float | None
    phase: Literal["normal", "precondition", "peak_coast", "peak_maintain"]
    reason: str
```

For heating:

- `normal`: use `normal_heat_setpoint_f`.
- `precondition`: set heat setpoint to `comfort_upper_f`.
- `peak_coast`: set heat setpoint to `comfort_lower_f` to avoid calls.
- `peak_maintain`: set heat setpoint to `comfort_lower_f`; if already below/near lower bound, allow normal thermostat protection.

For cooling:

- `normal`: use `normal_cool_setpoint_f`.
- `precondition`: set cool setpoint to `comfort_lower_f`.
- `peak_coast`: set cool setpoint to `comfort_upper_f` to avoid calls.
- `peak_maintain`: set cool setpoint to `comfort_upper_f`; if already above/near upper bound, allow normal thermostat protection.

The controller should not emit open-loop fixed equipment actions like `action=0.7`; prior CityLearn experiments showed fixed fraction-of-nominal-power actions can massively overheat or overcool. All runtime decisions should be feedback-gated by current temperature and comfort bounds.

## 6. Main Algorithm

The controller runs every control step. It finds the relevant peak window, predicts whether passive drift can cover the peak, and starts preconditioning only when active time plus a safety margin says it is necessary.

```text
function decide(state, config, predictor, forecast):
    validate state.indoor_temp is within a plausible range
    window = active_or_next_peak_window(state.timestamp, config.peak_windows, config.horizon_minutes)

    if no window:
        return normal_command("no peak window in horizon")

    if state.mode == heating:
        storage_target = config.comfort_upper_f
        drift_boundary = config.comfort_lower_f
        normal_setpoint = config.normal_heat_setpoint_f
        direction = "heating"
    else:
        storage_target = config.comfort_lower_f
        drift_boundary = config.comfort_upper_f
        normal_setpoint = config.normal_cool_setpoint_f
        direction = "cooling"

    if state.timestamp is inside window:
        minutes_to_peak_end = window.end - state.timestamp
        forecast_temp = representative_outdoor_temp(
            forecast,
            start=state.timestamp,
            end=window.end,
            fallback=state.outdoor_temp_f,
        )
        drift_minutes = predictor.predict_drift_time(
            current_temp=state.indoor_temp_f,
            boundary_temp=drift_boundary,
            outdoor_temp=forecast_temp,
            timestamp=state.timestamp,
            home_id=state.home_id,
            direction=direction,
        )

        if drift_minutes >= minutes_to_peak_end + config.drift_safety_margin_minutes:
            return peak_coast_command("predicted drift stays within comfort through peak")

        if at_or_past_boundary(state.indoor_temp_f, drift_boundary, direction):
            return peak_maintain_command("comfort boundary reached during peak")

        return peak_coast_command("defer HVAC until closer to comfort boundary")

    minutes_to_peak_start = window.start - state.timestamp

    if minutes_to_peak_start > config.max_precondition_lead_minutes:
        return normal_command("peak window too far away")

    predicted_start_time = state.timestamp
    forecast_temp = representative_outdoor_temp(
        forecast,
        start=state.timestamp,
        end=window.start,
        fallback=state.outdoor_temp_f,
    )
    active_minutes = predictor.predict_active_time(
        current_temp=state.indoor_temp_f,
        target_temp=storage_target,
        outdoor_temp=forecast_temp,
        timestamp=predicted_start_time,
        system_running=state.system_running,
        home_id=state.home_id,
        direction=direction,
    )

    gap_to_storage_target = directional_gap(state.indoor_temp_f, storage_target, direction)
    latest_start_minutes = active_minutes + config.precondition_margin_minutes

    if gap_to_storage_target < config.min_precondition_gap_f:
        return normal_command("already near storage target")

    if minutes_to_peak_start <= latest_start_minutes:
        return precondition_command("active-time prediction says preconditioning should start")

    return normal_command("not time to precondition yet")
```

### Notes on Peak Runtime Minimization

The peak-period rule intentionally does not try to reach a target temperature during the peak. It only asks whether drift will violate comfort before the peak ends. If the model predicts the home can coast, the thermostat setpoint is moved to the permissive comfort boundary so the equipment is unlikely to run.

If the model predicts comfort cannot last until peak end, the first implementation should still avoid aggressive peak runtime. It should defer calls until the comfort boundary is reached, then maintain the boundary. This keeps the primary metric low while enforcing comfort.

## 7. Functional Requirements

### FR-1 Configurable Peak Windows

The controller shall accept a list of peak windows. The list may contain any number of windows, including zero.

Acceptance criteria:

- With zero peak windows, the controller always returns `normal`.
- With multiple windows, the controller uses the currently active window if inside one; otherwise it uses the next window within the horizon.
- Overlapping windows raise `ValueError` during config validation.

### FR-2 Heating and Cooling Support

The controller shall support `mode="heating"` and `mode="cooling"`.

Acceptance criteria:

- Heating preconditions toward `comfort_upper_f` and coasts toward `comfort_lower_f`.
- Cooling preconditions toward `comfort_lower_f` and coasts toward `comfort_upper_f`.

### FR-3 Predictor Integration

The controller shall depend on the `FirstPassagePredictor` interface, not on concrete XGBoost artifacts.

Acceptance criteria:

- Unit tests can pass a fake predictor implementing `predict_active_time` and `predict_drift_time`.
- The controller can use `XGBFirstPassagePredictor.from_model_files()` without code changes.

### FR-4 Weather Forecast Integration

The controller shall use caller-provided forecasted outdoor temperature for first-passage predictions.

Acceptance criteria:

- The controller accepts a simple forecast provider, either as a callable or wrapper object.
- Active-time prediction uses representative forecasted outdoor temperature between the decision timestamp and the next peak start.
- Drift-time prediction during peak uses representative forecasted outdoor temperature between the decision timestamp and the peak end.
- If forecast data is missing, the controller falls back to `state.outdoor_temp_f` and exposes the fallback in the command reason or debug metadata.

### FR-5 Receding-Horizon Decision

The controller shall make a stateless or minimally stateful decision from the current state every control step.

Acceptance criteria:

- The same input state and config produce the same command.
- No day-long schedule needs to be precomputed.

### FR-6 Peak Runtime Protection

During a peak window, the controller shall prefer `peak_coast` whenever predicted passive drift remains inside comfort through the peak end.

Acceptance criteria:

- If `predict_drift_time(...) >= minutes_to_peak_end + drift_safety_margin_minutes`, return `peak_coast`.
- If the home is at or beyond the comfort boundary, return `peak_maintain`.

### FR-7 Preconditioning Start Time

Before a peak window, the controller shall start preconditioning only when the predicted active time plus a configurable margin reaches the time remaining before peak start.

Acceptance criteria:

- If `minutes_to_peak_start <= active_minutes + precondition_margin_minutes`, return `precondition`.
- If the current temperature is already near the storage target, return `normal`.
- If the peak is farther than `max_precondition_lead_minutes`, return `normal`.

### FR-8 Comfort Bounds

The controller shall never intentionally command a setpoint outside `[comfort_lower_f, comfort_upper_f]`.

Acceptance criteria:

- Heating setpoints are clipped or validated within bounds.
- Cooling setpoints are clipped or validated within bounds.
- Invalid config where `comfort_lower_f >= comfort_upper_f` raises `ValueError`.

### FR-9 Explainability

Every command shall include a short `phase` and `reason`.

Acceptance criteria:

- Logs or tests can identify whether the command came from normal, precondition, peak coast, or peak maintain logic.

## 8. Non-Functional Requirements

- Implementation footprint: target one new module, `mpc/peak_mpc.py`; optional tests in one test file. Do not exceed three implementation-related files for the first version.
- Dependencies: use only the existing project stack (`pandas`, `numpy`, standard library) unless a later implementation needs an adapter.
- Runtime: one decision should complete in milliseconds excluding predictor model latency.
- Determinism: no random sampling in the controller.
- Units: all controller temperatures are degrees F; all durations are minutes.
- Simplicity: avoid generic optimization libraries for the first implementation.

## 9. Proposed File Layout

MVP:

```text
mpc/
├── model_interfaces.py   # existing predictor API
└── peak_mpc.py           # new controller config, state, command, algorithm
```

Optional verification file:

```text
tests/test_peak_mpc.py
```

No additional model files, training scripts, or old plan documents are required for the first implementation.

## 10. Test Plan

Use a fake predictor with configurable active and drift durations.

Required unit tests:

- No peak windows returns `normal`.
- Future peak outside horizon returns `normal`.
- Future peak inside horizon but before latest start returns `normal`.
- Future peak at/after latest start returns `precondition`.
- Heating precondition command uses the upper comfort bound.
- Cooling precondition command uses the lower comfort bound.
- In-peak drift long enough returns `peak_coast`.
- In-peak drift too short but not at boundary returns `peak_coast` with a boundary-deferral reason.
- In-peak at boundary returns `peak_maintain`.
- Forecast provider value is passed into active-time prediction.
- Forecast provider value is passed into drift-time prediction.
- Missing forecast falls back to current outdoor temperature.
- Multiple peak windows choose the active window over a later window.
- Multiple peak windows choose the next future window when not inside a peak.
- Overlapping windows raise `ValueError`.
- Invalid comfort bounds raise `ValueError`.

Recommended simulation checks:

- Compare against normal thermostat baseline on peak runtime minutes.
- Confirm comfort violation minutes remain near zero.
- Confirm total runtime does not grow unreasonably, e.g. more than 25% versus baseline without an explicit aggressive setting.

## 11. Open Questions

- Should the first implementation expose only thermostat setpoints, or also a CityLearn action adapter for `heating_device`/`cooling_device`?
- Should `mode` be supplied directly by the caller, or inferred from outdoor temperature and current heat/cool setpoints?
- What comfort band should be used in the first experiment, e.g. `[68, 72]` F for heating and `[72, 76]` F for cooling, or user-specific bounds from data?
- Should preconditioning be capped by a maximum added pre-peak runtime budget per peak event?
- Should the forecast wrapper interpolate between hourly forecast points, or use nearest-prior values to match thermostat control intervals?
