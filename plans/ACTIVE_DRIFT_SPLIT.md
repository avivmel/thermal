# Plan: Separate Active and Drift Thermal Models for MPC

## Context

For MPC we need to model the full thermal cycle:
- **Active**: HVAC on, driving temp toward setpoint → "how long to reach target?"
- **Drift**: HVAC off, temp drifting toward outdoor → "how fast does the home lose/gain heat?"

We currently only have the active model. We need a drift model, and we need to re-train the active model on active-only episodes (system already running).

---

## Step 1: Extract Drift Episodes

**Create**: `scripts/extract_drift_episodes.py` (follow pattern of `extract_setpoint_responses.py`)

**Episode definition**:
- Start: All *available* runtime columns = 0 (some homes only have heating or cooling equipment — check which runtime columns exist per home rather than requiring both), and |indoor - outdoor| > 3°F
- End: Any runtime goes > 0, or setpoint changes > 1°F, or data gap
- Min duration: 15 min (3 timesteps)
- Filter: Only keep episodes where temp actually changed (|end_temp - start_temp| > 0.5°F)

The delta filter is critical — when indoor ≈ outdoor there's no thermal driving force and nothing to learn. Note: with integer quantization this effectively requires ≥1°F change, which drops short drift episodes in well-insulated homes. This is acceptable — those episodes contain too little signal to fit a decay constant.

**Output**: `data/drift_episodes.parquet`

**Schema**: Same structure as setpoint_responses (home_id, state, split, episode_id, timestamp, timestep_idx, temps, humidity, setpoints) plus:
- `start_temp`, `start_outdoor`, `initial_delta` (indoor - outdoor)
- `drift_direction`: `cooling_drift` or `warming_drift`

---

## Step 2: Drift Rate Models

**Create**: `scripts/run_drift_baselines.py`

**Target**: Time-to-target (minutes) — same framing as the active model. For drift episodes, the "target" is a threshold temperature (e.g., comfort bound). This keeps the drift and active models parallel in structure.

For MPC, the drift model answers: "how many minutes until indoor temp drifts to X?"

**Models** (two tracks):

### Physics track
Newton's law of cooling: fit decay constant `k` via least-squares on episode temperature trajectory.
`T(t) = T_outdoor + (T_start - T_outdoor) × exp(-k × t)` → derive time-to-target from k.
1. Global k (single parameter — median across all episodes)
2. Per-home k (captures insulation/thermal mass variation)

### GBM track (reuse existing model machinery)
Same approach as `run_xgboost_baselines.py` — predict time-to-target directly (with log-transform), no k involved.
3. Global GBM (no home info)
4. Global GBM + home target encoding
5. Hybrid GBM (global + per-home residual)

Features: `log_abs_delta`, `abs_delta`, `drift_direction`, `start_temp`, `outdoor_temp`, time/humidity features, home encoding.

**Evaluation**: MAE on time-to-target (minutes), stratified by delta size, direction, home.

---

## Step 3: Re-train Active Model

**Create**: `scripts/run_active_baselines.py` (based on `run_xgboost_baselines.py`)

**Changes**:
1. Keep all episodes (both cold-start and already-running) — MPC needs to predict time-to-target from any starting state
2. Keep `system_running` as a feature (it's highly predictive: 30 min MAE cold-start vs 16 min when running)
3. Recompute home target encodings on this dataset
4. Same model variants: Global, Global+HomeEnc, Per-Home, Hybrid

This is essentially a re-run of `run_xgboost_baselines.py` confirming the active model works standalone.

---

## Files

| Action | File |
|--------|------|
| Create | `scripts/extract_drift_episodes.py` |
| Create | `scripts/run_drift_baselines.py` |
| Create | `scripts/run_active_baselines.py` |

## Verification

```bash
python scripts/extract_drift_episodes.py --test --n-homes 10
python scripts/extract_drift_episodes.py
python scripts/run_drift_baselines.py
python scripts/run_active_baselines.py
```
