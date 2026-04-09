# Plan: Reformulate Drift Prediction as Time-to-Comfort-Boundary

## Context

The active model is already framed as a first-passage-time problem: predict how long it takes for indoor temperature to reach the target setpoint during a setpoint response episode.

The current drift model is intended to be parallel, but the implementation uses raw drift episode duration as the target. That mixes passive thermal dynamics with thermostat control, occupant setpoint changes, and data gaps. As a result, drift MAE is inflated and the target is misaligned with MPC.

We should reformulate drift as:

- **Cooling drift**: minutes until `indoor_temp >= cool_setpoint` while HVAC remains off
- **Warming drift**: minutes until `indoor_temp <= heat_setpoint` while HVAC remains off

This makes drift and active symmetric:
- **Active**: time until indoor reaches commanded setpoint
- **Drift**: time until indoor reaches the relevant comfort boundary

---

## Goal

Make the drift task a clean, control-relevant threshold-crossing prediction instead of "how long did this HVAC-off segment last?"

For MPC, the drift model should answer:

- "If HVAC stays off, how many minutes remain before the home violates the comfort band?"

---

## Step 1: Redefine Drift Labels Around Boundary Crossing

**Update**: [`scripts/extract_drift_episodes.py`](/Users/amelamud/Desktop/thermal/scripts/extract_drift_episodes.py)

Keep the current notion of an HVAC-off interval as the source data, but derive a new target from the first crossing of the relevant comfort boundary.

For each candidate drift interval:
- Determine direction from the initial indoor-outdoor delta
- Identify the relevant boundary:
  - `cool_setpoint` for cooling drift
  - `heat_setpoint` for warming drift
- Compute the **first timestep** where indoor temperature crosses that boundary while all runtime columns remain 0
- Define `time_to_boundary_min` from the interval start to that first crossing

If no crossing is observed before the interval ends:
- Option A: drop the episode for the first baseline pass
- Option B: retain it as right-censored data for a later survival model

For the first implementation, use **Option A** to keep the training/evaluation setup simple and directly comparable to the active model.

**New output fields**:
- `target_boundary`
- `crossed_boundary` (bool)
- `crossing_timestep_idx`
- `time_to_boundary_min`
- `distance_to_boundary`

---

## Step 2: Separate Physics From Control/Behavior

**Update**: [`scripts/run_drift_baselines.py`](/Users/amelamud/Desktop/thermal/scripts/run_drift_baselines.py)

Replace the current target:
- `duration_min`

with:
- `time_to_boundary_min`

Evaluation should only use episodes with an observed crossing in the initial pass.

This removes label noise from:
- thermostat turn-on logic
- occupant setpoint changes
- interval termination unrelated to comfort violation
- missing-data truncation

The script header and comments should explicitly state that drift is now a boundary-crossing task, not a generic HVAC-off duration task.

---

## Step 3: Align Feature Semantics With the New Target

**Update**: [`scripts/run_drift_baselines.py`](/Users/amelamud/Desktop/thermal/scripts/run_drift_baselines.py)

Keep useful existing features:
- `log_abs_delta`
- `abs_delta`
- `start_temp`
- `outdoor_temp`
- direction indicator
- time-of-day / season
- humidity
- home encoding

Add target-aligned features:
- `distance_to_boundary`
- `log_distance_to_boundary`
- `boundary_temp`
- `signed_boundary_gap`
- recent indoor slope over the prior 15-30 min, if available
- recent outdoor slope over the prior 15-30 min, if available
- time since HVAC last turned off, if recoverable from source data

Rationale:
- `initial_delta` tells us thermal driving force toward outdoor
- `distance_to_boundary` tells us how much comfort margin remains
- both are needed to estimate crossing time well

---

## Step 4: Keep the Model Family Simple First

Use the same baseline family as the current drift script:

1. Global Newton-style physics baseline
2. Per-home Newton baseline
3. Global GBM
4. Global GBM + home encoding
5. Hybrid GBM

The key change is the target, not the model class.

This isolates whether the current error is mostly a label-definition problem.

---

## Step 5: Handle Non-Crossing Intervals Explicitly

After the threshold-crossing baselines are working, add a second phase for non-crossing intervals.

Two reasonable extensions:

1. **Survival / hazard formulation**
- one row per 5-minute step
- target is whether boundary crossing occurs in the next step
- intervals that end without crossing are right-censored rather than dropped

2. **Short-horizon passive dynamics**
- predict `delta_temp_15m` or `delta_temp_30m`
- roll forward in MPC until the boundary is reached

The survival formulation is the best long-term fit if censoring is common.

---

## Step 6: Verification and Comparison

Run a direct before/after comparison against the current drift setup.

Check:
- number of extracted drift intervals
- fraction with observed boundary crossing
- target distribution before vs after reformulation
- MAE / median / P90 on the new drift task
- calibration by `distance_to_boundary`
- calibration by `abs(indoor - outdoor)`

Success criteria:
- lower MAE than the current drift setup
- tighter P90
- clearer interpretation of errors
- target semantics that match MPC directly

---

## Files

| Action | File |
|--------|------|
| Update | `scripts/extract_drift_episodes.py` |
| Update | `scripts/run_drift_baselines.py` |
| Optional update | `CLAUDE.md` |

---

## Verification

```bash
python scripts/extract_drift_episodes.py --test --n-homes 10
python scripts/run_drift_baselines.py
```

Compare against the current drift baseline results in [`CLAUDE.md`](/Users/amelamud/Desktop/thermal/CLAUDE.md).
