# Plan: Dynamic Programming MPC with Active Heating and Drift Models

## Context

The repo now has two control-relevant first-passage-time models:

- **Active heating model**: predict minutes until indoor temperature reaches a commanded heating target
- **Drift model**: predict minutes until indoor temperature reaches the heating comfort boundary while HVAC stays off

That is a good fit for finite-horizon dynamic programming, but there is one important modeling constraint:

- the models predict **time to a threshold**
- they do **not** directly predict `next_temp = f(current_temp, action, outdoor, time)`

So the MPC plan should not start from a generic linear state-space formulation. It should start from the models we actually have and build the DP around their first-passage-time semantics.

For v1, keep scope narrow:

- single home
- heating season only
- one-day horizon at 5-minute control intervals
- deterministic outdoor temperature forecast over the horizon
- thermostat setpoint control, not direct power control

---

## Goal

Implement an MPC controller that chooses a heating schedule over the remainder of the day to:

1. keep the home within the comfort band
2. reduce concentrated heating bursts and overall peak demand
3. minimize an **additive squared heating-effort objective**

Recommended stage cost:

```text
stage_cost_t = lambda_energy * energy_t^2
             + lambda_comfort * comfort_violation_t^2
```

Why `energy_t^2` is the right starting point:

- it is additive across time, so Bellman recursion applies directly
- it penalizes concentrated heating bursts more than spread-out heating
- it encourages flatter heating schedules even without an explicit price or cost term

---

## Key Design Choice

Use the active and drift models as **travel-time oracles**, then precompute a discrete transition table for DP.

Do not try to build MPC by pretending the models are one-step linear dynamics. That would throw away the main advantage of the current setup.

Recommended interpretation:

- under **heating**, ask: "How long would it take to reach temperature `T'` from `T` if I command target `T'`?"
- under **drift**, ask: "How long would it take to drift from `T` down to the relevant lower comfort boundary with HVAC off?"

Then convert those travel times into reachable next-temperature bins over one control interval.

---

## DP Formulation

### State

Use a discrete state:

```text
s_t = (t, indoor_temp_bin, hvac_state_flag)
```

where:

- `t` is the 5-minute time index
- `indoor_temp_bin` is a discrete indoor temperature, for example 60°F to 76°F in 0.5°F bins
- `hvac_state_flag` is whether the system is already running at the start of the interval

Keep exogenous signals outside the state and treat them as known over the horizon:

- outdoor forecast
- comfort boundary / occupied setpoint schedule

The `hvac_state_flag` is worth keeping because the active model already uses `system_running`, and the error gap between cold-start and already-running episodes is large.

### Action

For heating-only v1, use thermostat setpoints directly as the action.

The simulator only accepts thermostat commands:

```python
{"heat_setpoint": float, "cool_setpoint": float}
```

So the controller should optimize only `heat_setpoint` and hold `cool_setpoint` fixed high
(for example `76°F`) throughout the heating-season experiment.

Recommended action set:

```text
A_t = {
  heat_to_lower_comfort_setpoint,
  heat_to_68,
  heat_to_69,
  heat_to_70,
  ...
  heat_to_74
}
```

The exact grid should align with allowed thermostat setpoints and the comfort schedule.

This keeps the controller thermostat-realistic:

- every DP action is a thermostat setpoint that can be sent directly to the simulator
- active actions are higher commanded heating setpoints
- the passive action is the current minimum allowed heating setpoint
- HVAC power remains implicit

The passive action is not a separate "off" command. It means:

- command the current lower comfort setpoint
- if indoor temperature is above that level, the thermostat remains off and the home drifts naturally
- when indoor temperature gets close enough, the simulator's thermostat logic may call for heat again

### Terminal Value

Work backward from the end of the day with terminal value:

```text
V_T(s) = terminal_comfort_penalty(s)
```

For v1:

- `0` if final temperature is within the end-of-day comfort band
- large penalty otherwise

To avoid horizon-end gaming, optionally extend the horizon 1-2 hours beyond the evaluation window or add a terminal penalty for ending far below the normal occupied setpoint.

---

## Transition Construction

### 1. Active Heating Transition Table

For each tuple:

- time index `t`
- outdoor forecast `O_t`
- current temp bin `T`
- action setpoint `theta`
- running flag `r`

if `theta > T`, query the active model for:

```text
tau_heat(T, theta, O_t, time_features_t, r)
```

Then convert that duration into a one-step successor temperature for interval length `dt = 5 min`.

Recommended v1 approximation:

```text
progress = min(dt / tau_heat, 1.0)
T_next = T + progress * (theta - T)
```

and discretize `T_next` back to the temperature grid.

Interpretation:

- if `tau_heat <= dt`, the target is reachable within one interval
- if `tau_heat > dt`, move only partway toward the target

### 2. Passive Drift Transition Table

For each tuple:

- time index `t`
- outdoor forecast `O_t`
- current temp bin `T`
- passive setpoint action `theta_passive = lower_comfort_setpoint_t`

query the current drift model for the time until the home reaches the relevant heating comfort boundary under passive drift:

```text
tau_drift(T, lower_comfort_boundary_t, O_t, time_features_t)
```

Then approximate the one-step passive evolution:

```text
progress = min(dt / tau_drift, 1.0)
T_next = T + progress * (lower_comfort_boundary_t - T)
```

For v1, there is only one passive branch:

- command the minimum allowed heating setpoint for the interval
- model the resulting passive temperature motion toward the lower comfort boundary

Do not build `coast_down_1F` / `coast_down_2F` actions unless a new drift-to-arbitrary-target model is added later.

### 3. Feasibility Rules

During DP, reject or heavily penalize transitions that:

- end below the comfort boundary
- require heating targets outside the thermostat limits
- command setpoints below the scheduled minimum comfort setpoint

### 4. Energy Proxy

The models predict temperature timing, not kW, so the controller needs a separate energy model.

Use a simple energy proxy for v1:

```text
runtime_minutes_t = min(tau_heat, dt)            for active heating actions
runtime_minutes_t = 0                            for passive setpoint actions

energy_t = hvac_power_kw * runtime_minutes_t / 60
```

Where `hvac_power_kw` comes from:

- a fixed home-specific estimate from metadata or calibration, or
- a simple regression from indoor/outdoor conditions if available later

This is sufficient for the first DP baseline because the objective is relative peak suppression, not perfect billing accuracy.

---

## Bellman Recursion

With the transition table in hand, solve backward:

```text
V_t(s) = min_a [ stage_cost_t(s, a) + V_{t+1}(f_t(s, a)) ]
```

Policy extraction:

```text
pi_t(s) = argmin_a [ stage_cost_t(s, a) + V_{t+1}(f_t(s, a)) ]
```

Notes:

- this is deterministic finite-horizon DP
- no stochastic disturbance model is required for v1
- if model uncertainty becomes important later, upgrade to robust DP or add quantile-based safety margins

---

## Receding-Horizon MPC Wrapper

The DP solver is used inside a standard MPC loop:

1. observe current indoor temp, outdoor temp, time, and whether HVAC is running
2. snap the current temp to the nearest grid bin
3. solve DP from `now` to end-of-day using the latest forecast and comfort schedule
4. apply only the first thermostat action:

```python
{"heat_setpoint": theta_star, "cool_setpoint": 76.0}
```

5. repeat at the next timestep

This gives feedback correction even though the value function is computed over a deterministic forecast.

---

## Implementation Steps

## Step 1: Wrap the existing models behind a stable interface

Create a small predictor layer that hides training-script details and exposes:

```python
predict_heat_time(current_temp, target_temp, outdoor_temp, timestamp, system_running, home_id, ...)
predict_drift_time(current_temp, boundary_temp, outdoor_temp, timestamp, home_id, ...)
```

This should live outside the current experiment scripts so MPC code does not depend on notebook-style data pipelines.

## Step 2: Define the MPC problem inputs

Create a config object for:

- timestep minutes
- temperature grid
- action grid
- comfort schedule by time
- home HVAC power estimate
- objective weights

Keep this explicit and serializable so experiments are reproducible.

## Step 3: Build discrete transition tables

For every time index and relevant state-action pair, precompute:

- next temperature bin
- next running flag
- estimated interval energy
- feasibility mask

Caching matters here. The model calls are the expensive part; the backward DP itself is small once the tables exist.

## Step 4: Implement the backward DP solver

Add a solver that:

- initializes terminal values
- iterates backward over time
- stores both value and argmin action tables
- returns the optimal first action and optionally the full open-loop plan

## Step 5: Add the MPC loop

Create a controller class that:

- ingests current observation
- rebuilds or reuses the remaining-horizon transition tables
- runs DP
- emits the first thermostat action

## Step 6: Evaluate in `thermalgym`

Compare against:

- constant setpoint baseline
- `PreHeat` policy in [`thermalgym/policies.py`](/Users/amelamud/Desktop/thermal/thermalgym/policies.py)
- simple setback rules

Metrics:

- peak kW
- peak-to-average ratio
- total kWh
- squared-energy objective value
- comfort violations
- end-of-day comfort

---

## Recommended File Layout

| Action | File |
|--------|------|
| Create | `mpc/model_interfaces.py` |
| Create | `mpc/problem.py` |
| Create | `mpc/transitions.py` |
| Create | `mpc/dp_solver.py` |
| Create | `mpc/controller.py` |
| Create | `scripts/run_mpc_demo.py` |
| Optional update | `thermalgym/__init__.py` |
| Optional update | `CLAUDE.md` |

If you want to stay even lighter for the first pass, all five `mpc/` files can start as one module and be split later.

---

## Verification Plan

### Unit Checks

- Heating transitions should be monotone: higher targets should never produce lower `T_next`
- Drift transitions should be monotone toward the comfort boundary
- `energy_t^2` should increase when the same total heating is concentrated into fewer intervals
- DP should choose the passive lower-comfort-setpoint action when comfort allows

### Scenario Checks

1. **Occupied setpoint increase later in the day**
   The controller should spread heating out rather than waiting and creating a single large burst.

2. **Already warm home**
   The controller should avoid unnecessary heating and let the home coast.

3. **Cold start near a comfort deadline**
   The controller should heat immediately if otherwise comfort would be violated.

4. **Very mild day**
   The controller should largely stay at the lower comfort setpoint and let the home drift.

### Simulator Comparison

Run the chosen MPC actions in `thermalgym` and compare predicted vs realized:

- indoor temperature trajectory
- HVAC runtime / energy
- timing of comfort-boundary crossings

This is the main validation that the first-passage-time approximation is usable for control.

---

## Risks and Follow-On Work

### Risk 1: One-step interpolation may be too crude

The `progress = dt / tau` approximation is simple, but it assumes local motion toward the target is roughly linear over one interval.

If this is too inaccurate, the next upgrade should be:

- precompute multi-step travel-time edges directly, or
- fit a short-horizon delta-temperature model specifically for DP transitions

### Risk 2: Energy proxy may be too weak

If constant `hvac_power_kw` is too crude, add a separate power model:

- input: indoor temp, outdoor temp, action, running flag
- output: expected kW over the next interval

### Risk 3: Uncertainty is ignored

Later versions should use:

- prediction quantiles from the active and drift models
- safety margins near comfort boundaries
- robust or risk-sensitive DP

---

## Suggested Success Criteria for V1

- MPC beats fixed `PreHeat` on peak reduction in `thermalgym`
- comfort violations stay near zero
- the controller visibly spreads heating out before the peak instead of creating a single large burst
- runtime is fast enough to replan every 5 minutes for one home on a laptop

---

## First Implementation Order

1. build model wrappers
2. build transition table generator
3. implement backward DP on a small temperature grid
4. run a single-home heating-day demo
5. compare against `PreHeat`
6. only then expand action/state complexity
