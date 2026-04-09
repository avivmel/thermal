# Drift Boundary-Crossing Prediction

## Task Definition

Predict how many minutes remain before an HVAC-off home violates the comfort band.

This is the passive counterpart to the active time-to-target task:

- Active: HVAC is on, predict time until indoor temperature reaches the commanded setpoint
- Drift: HVAC is off, predict time until indoor temperature reaches the relevant comfort boundary

For MPC, the drift model answers:

```text
"If HVAC stays off right now, how many minutes remain before comfort is violated?"
```

## Drift Labels

The raw HVAC-off interval length is not the target. Interval duration mixes thermal physics with:

- thermostat turn-on logic
- occupant setpoint changes
- interval truncation from missing data
- unrelated segment termination

The implemented drift label is first passage time to the comfort boundary:

- `cooling_drift`: minutes until `indoor_temp >= cool_setpoint` while HVAC remains off
- `warming_drift`: minutes until `indoor_temp <= heat_setpoint` while HVAC remains off

If no boundary crossing is observed before the HVAC-off interval ends, the interval is currently dropped rather than treated as censored.

## Dataset

File: `data/drift_episodes.parquet`

Each row is still a 5-minute timestep within an extracted episode, but each retained episode is truncated at the first observed boundary crossing.

### Episode Construction

Start condition:
- all available runtime columns are 0
- `|indoor - outdoor| > 3°F`

End condition for the candidate HVAC-off interval:
- any HVAC runtime becomes positive
- heat or cool setpoint changes by more than 1°F from the interval start
- missing data or a gap larger than 10 minutes

Keep an interval only if:
- the home starts inside the comfort band with a positive distance to the relevant boundary
- the relevant boundary is crossed before the interval ends
- `time_to_boundary_min >= 15`

### Key Fields

| Field | Meaning |
|-------|---------|
| `target_boundary` | `cool_setpoint` or `heat_setpoint` |
| `boundary_temp` | Boundary temperature to be crossed |
| `crossed_boundary` | Always `true` for retained episodes |
| `crossing_timestep_idx` | First timestep index where the boundary is crossed |
| `time_to_boundary_min` | Episode target in minutes |
| `distance_to_boundary` | Absolute comfort margin at episode start |
| `signed_boundary_gap` | `boundary_temp - start_temp` |
| `recent_indoor_slope_15m`, `recent_indoor_slope_30m` | Recent indoor trend before episode start |
| `recent_outdoor_slope_15m`, `recent_outdoor_slope_30m` | Recent outdoor trend before episode start |
| `time_since_hvac_off_min` | Minutes since the current HVAC-off segment began |

## Baseline Models

Implemented in `scripts/run_drift_baselines.py`.

Model family:

1. Global Newton baseline
2. Per-home Newton baseline
3. Global GBM
4. Global GBM + home target encoding
5. Hybrid GBM

### Feature Set

Core features:
- `log_abs_delta`
- `abs_delta`
- `start_temp`
- `outdoor_temp`
- `boundary_temp`
- `distance_to_boundary`
- `log_distance_to_boundary`
- `signed_boundary_gap`
- direction indicator

Additional features:
- hour / month cyclic features
- humidity
- recent indoor and outdoor slopes
- `time_since_hvac_off_min`
- optional home target encoding for the encoded GBM

## Verification Snapshot

Commands used:

```bash
python3 scripts/extract_drift_episodes.py --test --n-homes 10
python3 scripts/run_drift_baselines.py
```

Full-data extraction:

- Candidate HVAC-off intervals: 4,609,481
- Retained boundary-crossing episodes: 403,299
- Crossing fraction: 8.7%
- Dropped without crossing: 1,334,052
- Direction mix: 261,966 `cooling_drift`, 141,333 `warming_drift`
- Split mix: 280,896 train, 37,126 val, 85,277 test
- Time-to-boundary: mean 85.4 min, median 25.0 min, P90 230 min
- Distance-to-boundary: mean 1.7 F, median 1.0 F

10-home test extraction:

- Candidate HVAC-off intervals: 7,238
- Retained boundary-crossing episodes: 326
- Crossing fraction: 4.5%
- Dropped without crossing: 679

10-home reformulated baseline results:

| Model | MAE | Median | P90 |
|-------|-----|--------|-----|
| Global Newton k | 76.1 min | 43.5 min | 162.1 min |
| Per-Home Newton k | 42.4 min | 25.2 min | 95.0 min |
| Global GBM | 36.7 min | 17.4 min | 91.3 min |
| **GBM + Home Enc** | **32.5 min** | **13.4 min** | **85.2 min** |
| Hybrid GBM | 36.6 min | 17.3 min | 90.3 min |

## Notes

- The test extract is heavily dominated by `warming_drift`; only 8 extracted episodes were `cooling_drift` in the 10-home January run.
- Full-data extraction is complete, but the all-data baseline evaluation was started and did not finish within the execution window, so full-data model metrics are still missing.
- Non-crossing HVAC-off intervals are currently excluded. A later survival-model pass can treat them as right-censored data instead of dropping them.
