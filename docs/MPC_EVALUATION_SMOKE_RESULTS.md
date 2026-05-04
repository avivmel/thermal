# MPC Evaluation Smoke Results

Initial single-day ThermalGym evaluations for the peak-aware MPC controller.
These runs are not the full research grid; they are a smoke study to validate
the evaluation harness, screen useful stress cells, and expose metric issues
before scaling up.

## Setup

- Simulation engine: ThermalGym / EnergyPlus
- Timestep: 15 minutes
- Episode length: 1 day
- Peak window: 17:00-20:00
- Forecast: EPW perfect forecast
- Predictor: default `models/active_time_xgb.pkl` and `models/drift_time_xgb.pkl`
- Results directory: `results/mpc_eval/curated_stress/`

## Screened Stress Cells

Baseline screening was used to avoid dates with hot or cold weather but no HVAC
load. Candidate cells were ranked by baseline `peak_energy_kwh`.

| Mode | Building | Date | Baseline peak kWh | Notes |
|---|---|---:|---:|---|
| Heating | `small_cold_heatpump` | 2017-01-23 | 0.603 | Coldest screened heating day; no baseline comfort violations. |
| Cooling | `large_hot_ac` | 2017-07-29 | 2.081 | Strongest screened cooling day; baseline runs through the full peak window. |

The three cold archetypes produced identical metrics for the screened heating
dates, so `small_cold_heatpump` was used as the representative heating stress
cell. This should be revisited because it may indicate that the current
ThermalGym archetypes are less differentiated than intended.

## Controller Results

### Cooling: `large_hot_ac`, 2017-07-29

| Controller | Peak energy reduction | Total energy overhead | Comfort delta |
|---|---:|---:|---:|
| MPC | 74.12% | +10.75% | 0 min |
| PreCool | 48.28% | +30.26% | 0 min |
| Setback | 75.95% | -28.40% | 0 min |

MPC substantially reduces peak energy and is more efficient than PreCool. Simple
Setback slightly beats MPC on peak energy in this one cell and uses less total
energy, so it is a serious comparator for cooling.

### Heating: `small_cold_heatpump`, 2017-01-23

| Controller | Peak energy reduction | Total energy overhead | Comfort delta |
|---|---:|---:|---:|
| MPC | 22.57% | -0.19% | 0 min |
| PreHeat | 38.73% | -0.55% | +165 min |
| Setback | 59.36% | -4.25% | +225 min |

MPC is the only non-baseline controller in this heating cell that reduces peak
energy without adding comfort violations. PreHeat and Setback reduce more peak
energy, but the comfort cost is large.

## Caveats

1. `peak_runtime_min` now uses a power-threshold fallback because `hvac_mode`
   can report `off` while `hvac_power_kw` is nonzero. With the default
   threshold of 0.05 kW, the heating stress cell has 180 baseline peak-runtime
   minutes, 150 MPC minutes, 135 PreHeat minutes, and 120 Setback minutes.
2. `comfort_violation_min` is now mode-aware. Heating counts too-cold minutes;
   cooling counts too-hot minutes. Full-band diagnostics are still retained as
   `comfort_band_violation_min`, `comfort_too_cold_min`, and
   `comfort_too_hot_min`.
3. Cooling comfort is still not discriminating in the current stress cell.
   Inspection of `large_hot_ac`, 2017-07-29 shows every controller starts below
   72 F and spends 510 minutes below the lower comfort bound, while none exceed
   76 F. This now appears as `comfort_violation_min = 0` for cooling, plus
   `comfort_band_violation_min = 510` and `comfort_too_cold_min = 510`.
4. Baseline screening is required. Weather-only screening selected days such as
   2017-07-31 and 2017-09-08 that were hot during the peak window but produced
   zero baseline cooling load.

## Next Fixes

1. Add a compact analysis/report script that reads `metrics.parquet` and emits
   the curated stress comparison table directly.
2. Decide whether full-band comfort diagnostics should be included in the
   headline table or only in appendix-style diagnostics.
