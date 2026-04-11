# Plan: Rule-Based Preheat MPC with First-Passage Models

## Context

The dynamic-programming MPC is useful as a control baseline, but it is heavier than needed for the first practical preheating controller. It also makes it easy to hide important behavior inside the objective and transition approximation.

This plan defines a simpler MPC-style controller that combines:

- the interpretability of the existing `PreHeat` rule-based policy
- the active heating first-passage model for estimating when preheating must start
- the drift first-passage model for estimating how much peak-window runtime preheating can avoid
- explicit comfort and thermostat bounds

The controller should not optimize over every possible setpoint sequence. Instead, it should compute a small set of candidate preheat schedules, reject unsafe ones, and choose the schedule with the best explicit peak-window runtime/kWh proxy.

---

## Goal

Implement a heating-season controller that:

1. keeps the home inside allowed comfort bounds
2. preheats before a high-cost or peak-demand window only when needed
3. uses the learned first-passage models to choose the preheat start time and preheat setpoint
4. executes a simple, explainable thermostat plan:

```text
normal comfort setpoint -> preheat setpoint -> peak setback setpoint -> recovery setpoint
```

The key difference from full DP is that this controller is a **model-informed rule policy**, not a full value-function optimizer.

---

## Intended Behavior

### Before Peak

Use the active heating model to answer:

```text
How many minutes are needed to raise the house from current temperature T
to the desired preheat target T_preheat before peak_start?
```

Then start preheating only when:

```text
time_to_peak <= predicted_heat_time + safety_margin
```

This avoids arbitrary fixed windows like "preheat 2 hours before peak" and replaces them with a home- and weather-specific start time.

### During Peak

Use a lower heating setpoint such as:

```text
peak_heat_setpoint = 66°F
```

The controller should try to coast through as much of the peak as possible using thermal storage from preheating. It does **not** require a preheat target to survive the entire peak window. If the home reaches the lower comfort boundary during the peak, the thermostat may still call for heat; the objective is to reduce peak-window runtime/kWh, not to make peak heating impossible.

### After Peak

Return to the normal heating setpoint:

```text
normal_heat_setpoint = 68°F
```

Optionally use a recovery ramp instead of an immediate jump if the post-peak recovery creates a new heating burst.

---

## Model Questions

The controller should use the existing first-passage semantics directly.

### Active Heating Model

Use for preheat timing:

```python
tau_heat = predict_heat_time(
    current_temp=T_now,
    target_temp=T_preheat,
    outdoor_temp=outdoor_forecast_at_start,
    timestamp=now,
    system_running=hvac_running,
    home_id=home_id,
)
```

This answers:

```text
If I start heating now, how long until I reach the preheat target?
```

### Drift Model

Use for peak runtime shifting:

```python
tau_drift = predict_drift_time(
    current_temp=T_preheat,
    boundary_temp=peak_lower_comfort,
    outdoor_temp=outdoor_forecast_during_peak,
    timestamp=peak_start,
    home_id=home_id,
)
```

This answers:

```text
If I enter the peak at T_preheat and set heat to the peak setback,
how long can the home delay peak-window heating before crossing the lower comfort boundary?
```

---

## Candidate Plan Generation

Generate a small list of candidate preheat targets:

```text
T_preheat in {68.5, 69.0, 69.5, 70.0, 70.5, 71.0, 72.0}
```

For each candidate:

1. Estimate active heating time from current temperature to `T_preheat`.
2. Compute required start time:

```text
preheat_start = peak_start - tau_heat - safety_margin
```

3. Estimate how much peak-window heating remains after the home coasts:

```text
residual_peak_runtime = max(
    0,
    peak_duration + drift_safety_margin - tau_drift(T_preheat -> peak_lower_comfort)
)
```

4. Score each candidate with a runtime proxy:

```text
score =
    peak_runtime_weight * residual_peak_runtime
  + preheat_runtime_weight * predicted_preheat_runtime
```

The residual peak runtime term is the primary objective because the policy is meant to reduce peak-window kWh. The preheat runtime term keeps the controller from using excessive preheat when the peak benefit is small.

Use a conservative default heat-time safety margin because the active first-passage model can underpredict how early preheating must start. In the initial EnergyPlus check, a 10-minute margin selected the right high target but started too late; a 30-minute margin shifted more runtime out of the peak while staying under the 72°F cap.

5. Reject candidates that:

- exceed the maximum allowed heat setpoint
- would make indoor temperature exceed the upper allowed comfort bound
- require preheating before the earliest allowed preheat time
- create excessive pre-peak energy use according to the runtime proxy

6. Choose the feasible candidate with the lowest score, breaking ties toward the lower preheat target.

This gives a simple and defensible policy:

```text
Use preheating when the active-heating and drift models predict that it reduces peak-window runtime/kWh enough to justify the pre-peak runtime.
```

---

## Runtime Control Logic

At each 5-minute control step:

```text
if now < preheat_start:
    heat_setpoint = normal_heat_setpoint

elif preheat_start <= now < peak_start:
    heat_setpoint = selected_preheat_target

elif peak_start <= now < peak_end:
    if drift_model says comfort violation before peak_end:
        heat_setpoint = peak_lower_comfort
    else:
        heat_setpoint = peak_setback

else:
    heat_setpoint = normal_heat_setpoint
```

Keep the cooling setpoint fixed high for heating-season experiments:

```python
{"heat_setpoint": heat_setpoint, "cool_setpoint": 76.0}
```

---

## Receding-Horizon Update

This is still an MPC-style controller because it replans using the latest observation.

At each timestep:

1. observe current indoor temperature, outdoor temperature, HVAC mode, and time
2. update the forecast slice to the peak window
3. recompute candidate preheat targets and start times
4. execute only the current thermostat command
5. repeat at the next timestep

This gives feedback correction without solving a full DP.

---

## Configuration

Add a config object such as:

```python
@dataclass
class RulePreheatMPCConfig:
    timestep_minutes: int = 5
    normal_heat_setpoint: float = 68.0
    peak_heat_setpoint: float = 66.0
    cool_setpoint: float = 76.0
    peak_start_hour: int = 17
    peak_end_hour: int = 20
    candidate_preheat_targets: tuple[float, ...] = (68.5, 69.0, 69.5, 70.0, 70.5, 71.0, 72.0)
    earliest_preheat_hour: int = 14
    latest_preheat_start_margin_min: float = 10.0
    drift_safety_margin_min: float = 15.0
    heat_time_safety_margin_min: float = 30.0
    max_allowed_indoor_temp: float = 72.0
    preheat_runtime_weight: float = 0.25
    peak_runtime_weight: float = 1.0
```

The config should be explicit and serializable so the resulting schedules can be reproduced.

---

## Outputs for Debugging

The controller should expose the selected plan:

```python
{
    "selected_preheat_target": 70.0,
    "predicted_heat_time_min": 55.0,
    "preheat_start": "2017-01-15T16:05:00",
    "predicted_peak_coast_time_min": 205.0,
    "predicted_residual_peak_runtime_min": 0.0,
    "score": 13.75,
    "peak_duration_min": 180.0,
    "replan_reason": "lowest_peak_runtime_score",
}
```

This is important because the rule-based version is meant to be explainable. If it does not preheat, the controller should be able to explain why:

```text
No preheat selected because current temperature is already high enough to coast through peak.
```

or:

```text
No feasible preheat target is within thermostat, comfort, and timing bounds.
```

---

## Evaluation

Compare against:

- `Baseline(heat_setpoint=68)`
- fixed `PreHeat`
- current dynamic-programming MPC
- this rule-based preheat MPC

Metrics:

- total kWh
- peak-window kWh
- peak kW
- peak-to-average ratio
- comfort violations against the true comfort schedule, not against temporary preheat commands
- max indoor temperature
- selected preheat target
- selected preheat start time

The comfort metric must use the occupant comfort band, not the commanded thermostat preheat setpoint. A preheat command of `71°F` should not make `69°F` count as a comfort violation.

---

## Implementation Sketch

Recommended files:

| Action | File |
|--------|------|
| Create | `rule_mpc/rule_preheat.py` |
| Create | `rule_mpc/__init__.py` |
| Create | `scripts/run_rule_preheat_mpc_demo.py` |
| Create | `scripts/plot_rule_preheat_mpc.py` |

Core classes:

```python
@dataclass
class RulePreheatPlan:
    preheat_start: pd.Timestamp | None
    preheat_target: float
    peak_setback: float
    predicted_heat_time_min: float
    predicted_coast_time_min: float
    predicted_residual_peak_runtime_min: float
    score: float
    feasible: bool
    reason: str


class RulePreheatMPCController:
    def __call__(self, obs: dict) -> dict:
        plan = self.plan(obs)
        return {
            "heat_setpoint": plan.current_heat_setpoint,
            "cool_setpoint": self.config.cool_setpoint,
        }
```

---

## Risks

### Risk 1: Drift model only predicts boundary time from the current state

The scoring check asks about drifting from `T_preheat` at `peak_start`, which may be a hypothetical future state. For v1 this is acceptable, but the result should be treated as approximate.

### Risk 2: Preheating may not reduce measured peak kW

If the simulator reports brief equipment spikes even under a low setpoint, the controller may reduce peak-window kWh without changing max kW.

### Risk 3: Comfort metric confusion

Do not evaluate comfort against the commanded preheat setpoint. Evaluate against the actual occupant lower comfort boundary.

---

## Success Criteria

The v1 rule-based preheat MPC is successful if:

- it chooses no preheat on mild days
- it chooses earlier preheat on colder days
- it raises the preheat target when that lowers the predicted peak-window runtime/kWh score
- it reduces peak-window kWh versus baseline
- it has fewer comfort violations than fixed `PreHeat`
- its selected preheat start/target are explainable from model predictions and candidate scores
