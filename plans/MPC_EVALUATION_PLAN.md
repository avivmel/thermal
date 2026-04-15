# MPC Controller Evaluation Plan

End-to-end plan for evaluating the peak-aware first-passage MPC controller (`mpc/peak_mpc.py`) across a diverse set of simulated environments. The output should be of sufficient rigor and reproducibility to support a research paper.

The plan is organized so an implementing agent can execute it phase-by-phase. Each phase lists: deliverables, inputs, scripts to add, acceptance checks. **Do not skip the acceptance checks.**

---

## 1. Research Question and Hypotheses

**Primary question.** Does the peak-aware first-passage MPC reduce HVAC runtime during configured peak windows, relative to (a) a do-nothing baseline thermostat and (b) standard rule-based DR (PreCool / PreHeat / Setback), without unacceptable comfort loss or excessive total energy?

**Hypotheses to test.**

- **H1 (peak shifting).** MPC reduces peak-window HVAC runtime by ≥30% vs. Baseline across the full building × climate × season grid, and matches or beats the best rule-based policy.
- **H2 (comfort).** Median comfort violation magnitude under MPC stays within the configured comfort band; 95th-percentile excursion ≤ 1.0°F.
- **H3 (energy.)** Total daily HVAC energy with MPC is within 25% of Baseline (i.e., shifting, not adding load).
- **H4 (model dependence).** MPC performance depends on the quality of the first-passage predictor. Predictors trained on data from the same building distribution as the test environment outperform predictors trained on out-of-distribution data, but degradation is bounded (define "bounded" by the gap on H1/H2 metrics).
- **H5 (robustness to forecast error).** MPC degrades gracefully when the weather forecast is perturbed with realistic noise (define noise model below).
- **H6 (sensitivity).** Performance is robust over reasonable values of `precondition_margin_minutes`, `drift_safety_margin_minutes`, and comfort band width.

The paper's contribution is the controller design + the evaluation. Treat all six hypotheses as first-class results.

---

## 2. Environments (the test grid)

Use ThermalGym (`thermalgym/`) as the simulation engine. It wraps EnergyPlus and exposes setpoint-level control, which matches the MPC's output (`ThermostatCommand`).

### 2.1 Building × climate matrix

Use all 9 bundled building archetypes (`thermalgym.BUILDINGS`):

| | Cold (Duluth MN) | Mixed (Columbus OH) | Hot (Houston TX) |
|---|---|---|---|
| Small (1000 sqft) | small_cold_heatpump | small_mixed_heatpump | small_hot_heatpump |
| Medium (2000 sqft) | medium_cold_heatpump | medium_mixed_heatpump | medium_hot_heatpump |
| Large (3000 sqft) | large_cold_resistance | large_mixed_ac | large_hot_ac |

Include all 9. Diversity in vintage and HVAC type is required for the H4 generalization claim.

### 2.2 Seasons (test weeks)

Pick 4 representative weeks per climate, fixed by date so results are reproducible. Each week is a 7-day episode.

- **Winter peak heating week**: 2017-01-15 (cold + mixed); skip for hot climate (use mild winter week 2017-01-15 with cooling disabled).
- **Shoulder week (heating-dominant)**: 2017-03-15.
- **Shoulder week (cooling-dominant)**: 2017-05-15.
- **Summer peak cooling week**: 2017-07-15.

Total simulation episodes per controller: **9 buildings × 4 weeks = 36 weeks**. At `timestep_minutes=15` and 7 days each, this is tractable on a laptop (~10–30 min per episode → ~10–20 hours per full controller sweep; parallelize across processes).

### 2.3 Mode assignment

For each (building, week) pair, derive `mode` from the weekly mean outdoor temperature:

- Mean outdoor < 60°F → `heating`
- Mean outdoor > 70°F → `cooling`
- 60–70°F → run both a heating-only and a cooling-only episode (the controller cannot dynamically switch modes; this gives both arms a fair test in shoulder seasons).

Record the assigned mode in the results manifest.

### 2.4 Peak windows

Use a single peak window per day, varied across **two configurations**:

- **CA-style evening peak**: 17:00–20:00 daily.
- **Morning peak**: 07:00–10:00 daily (winter heating context).

Run every (building, week) pair against both peak configurations. **Total: 36 × 2 = 72 episodes per controller.**

### 2.5 Comfort bands

- Cooling: `[72, 76]°F`, normal cool setpoint 75°F.
- Heating: `[68, 72]°F`, normal heat setpoint 69°F.

Plus a **wider band sensitivity arm** in §6: `[70, 78]°F` cooling, `[66, 74]°F` heating, run on a representative subset of 6 (building, week) pairs.

---

## 3. Data Generation

Goal: training data that covers the same distribution as the evaluation environments, plus held-out distributions for H4.

### 3.1 In-distribution training data ("matched")

Use `thermalgym.generate_episodes` to produce setpoint-response episodes from the same 9 buildings, but on **different dates** than the evaluation weeks (full year minus the evaluation weeks above, with a ±3-day buffer to avoid leakage from autocorrelation).

Suggested counts per building, derived from current GBM training requirements:

- 800 episodes per building (≈ 50/50 heat_increase/cool_decrease)
- gap_range `(1.0, 5.0)` °F
- timestep_minutes `5`

Total: **9 × 800 = 7200 episodes** in `data/sim_episodes_matched.parquet`.

### 3.2 Held-out building data ("OOD-building")

Re-run `generate_episodes` with a fresh seed but **only on 6 buildings**, holding out 3 buildings entirely (one from each climate). Same per-building count.

Output: `data/sim_episodes_ood_building.parquet`. The 3 held-out buildings are evaluation-only — you'll train on data without them and test the controller on them.

### 3.3 Held-out climate data ("OOD-climate")

Generate from 6 buildings spanning only 2 climates (e.g., cold + mixed). Test on hot-climate buildings. Output: `data/sim_episodes_ood_climate.parquet`.

### 3.4 Real-world data ("Ecobee")

Reuse the existing `data/setpoint_responses.parquet` (Ecobee DYD). This is a **distribution-shifted training set** (real homes, integer-quantized temperatures, noisy schedules). Useful for H4 to characterize the cost of training on real data and deploying on simulated buildings (and arguably the more interesting direction for the paper: characterize how much better an in-sim trained model would be).

### 3.5 Drift episode dataset

The drift model needs passive-drift episodes (HVAC off, temperature drifting toward boundary). The current pipeline (`data/drift_episodes.parquet`) is built from Ecobee. Add an analogous generator for simulated data:

- Add `scripts/generate_drift_episodes_sim.py` that, for each building, runs ThermalEnv with thermostat setpoints set far outside the comfort band (so HVAC stays off) for short windows, and extracts segments where indoor temp drifts toward a chosen boundary. Save to `data/drift_episodes_sim_<tag>.parquet`.

Mirror the same matched / OOD-building / OOD-climate splits.

### 3.6 Acceptance for §3

- Each parquet exists, episode counts match within ±5%.
- Summary stats printed: per-building episode counts, mean indoor/outdoor temp, mean episode duration.
- No evaluation-week dates appear in any training parquet (assert in script).

---

## 4. Models to Train and Compare

The controller depends on a `FirstPassagePredictor` (active-time + drift-time models). Train multiple variants so the ablation in H4 is meaningful.

### 4.1 Variants

| Name | Active-time training data | Drift training data |
|---|---|---|
| `ecobee` | `setpoint_responses.parquet` | `drift_episodes.parquet` |
| `sim_matched` | `sim_episodes_matched.parquet` | `drift_episodes_sim_matched.parquet` |
| `sim_ood_building` | `sim_episodes_ood_building.parquet` | `drift_episodes_sim_ood_building.parquet` |
| `sim_ood_climate` | `sim_episodes_ood_climate.parquet` | `drift_episodes_sim_ood_climate.parquet` |
| `mixed` | concat of `ecobee` + `sim_matched` | same | concat |

Each variant produces two pickled artifacts compatible with `XGBFirstPassagePredictor.from_model_files()`:

- `models/active_time_xgb_<variant>.pkl`
- `models/drift_time_xgb_<variant>.pkl`

Reuse `scripts/run_xgboost_baselines.py` and `scripts/evaluate_drift_model.py` as templates. Add a thin training script:

- `scripts/train_predictor_variants.py --variant <name>` — produces the two pickles + a JSON metrics summary.

### 4.2 Predictor offline metrics (sanity gate before MPC eval)

For each variant, report on its held-out test split:

- Active-time: MAE, median AE, P90, all in minutes; both overall and by `system_running` and by `direction`.
- Drift-time: MAE, median AE, P90, by `direction`.
- A residuals-vs-gap plot (sanity check for log-domain bias).

Save to `results/predictor_eval/<variant>/metrics.json` and a 1-page markdown summary.

**Gate**: a variant must achieve test MAE within 2x of the published Ecobee MAE (18.7 min) on its in-distribution split before being included in the full MPC evaluation. Otherwise diagnose before proceeding.

### 4.3 Naive baseline predictor

Add `mpc/model_interfaces.py::ConstantRatePredictor` (a new class in the same file is fine, keeps the footprint small): assumes a fixed °F/min slew rate per direction. Used as a "no-ML" reference in the MPC evaluation. Two parameters: `active_rate_f_per_min`, `drift_rate_f_per_min`, fit by simple linear regression over the matched training data.

This gives the paper a clean answer to "is the ML predictor doing real work for the controller, or would a constant rate be enough?"

---

## 5. Controllers to Compare

All controllers are run inside ThermalGym via the same evaluation harness. They are:

1. **Baseline** — `thermalgym.Baseline` (no DR).
2. **Setback** — `thermalgym.Setback(magnitude=4.0, mode=<season>)`.
3. **PreCool** — `thermalgym.PreCool(precool_offset=2.0, precool_hours=2)` (cooling weeks only).
4. **PreHeat** — `thermalgym.PreHeat(preheat_offset=2.0, preheat_hours=2)` (heating weeks only).
5. **MPC-perfect-forecast** — `peak_mpc.decide` with the `sim_matched` predictor and **true** outdoor-temperature forecast (read directly from the EPW).
6. **MPC-noisy-forecast** — same as 5 but with a forecast perturbed by AR(1) noise (see §7).
7. **MPC-naive-predictor** — same as 5 but using `ConstantRatePredictor`.
8. **MPC-OOD-{building,climate}** — same as 5 but using the OOD predictor variants.
9. **MPC-Ecobee** — predictor trained on real Ecobee data only.

### 5.1 Adapter for ThermalGym → MPC

Create `mpc/thermalgym_adapter.py` (this is the third allowed file in §9 of the SRS). It wraps a ThermalGym `obs` dict into a `ThermostatState` and converts the MPC's `ThermostatCommand` into the ThermalGym action dict.

Key responsibilities:

- Determine `mode` from the season (passed in by the harness, not inferred from obs).
- Determine `system_running` from `obs["hvac_mode"] != "off"`.
- For `precondition` / `peak_coast` / `peak_maintain`, set the inactive setpoint to a wide pass-through value (heating sp = `HEAT_MIN`, cooling sp = `COOL_MAX`) so it doesn't fight the active mode.
- Apply `home_id = building.id` so the predictor's per-home residual lookup works (note: residuals will be missing for synthetic IDs, which is fine — the predictor falls back to the global model).

Add unit tests in `tests/test_thermalgym_adapter.py`.

### 5.2 Forecast wrapper

Add `mpc/forecast.py` (acceptable as the same file that hosts `EPWForecast`, `NoisyForecast`, `PerfectForecast`):

- `EPWForecast.from_epw(path)` — parses an EPW file into a `pd.Series` of hourly outdoor temperatures and exposes `outdoor_temp_at(ts)` via nearest-prior lookup.
- `NoisyForecast(base, ar1_phi=0.7, sigma_f=2.0, seed=None)` — wraps a base forecast and adds an AR(1) noise process with std-dev `sigma_f` °F.
- `PersistenceForecast(state)` — returns the current outdoor temp for any future time. Used as a stress-test "no forecast" baseline.

---

## 6. Evaluation Harness

Add `scripts/run_mpc_evaluation.py`. This is the centerpiece script.

### 6.1 Inputs

- `--config` (yaml/json) describing the (building, week, peak_window, comfort_band, controller, predictor_variant, forecast_kind, seed) grid. The script enumerates the full Cartesian product and dispatches one process per cell.
- `--out_dir` (default `results/mpc_eval/<timestamp>/`).
- `--workers` for `multiprocessing.Pool`.

### 6.2 Per-cell execution

For each cell:

1. Instantiate `ThermalEnv(building=..., timestep_minutes=15, run_period_days=7, ...)`.
2. Call `env.reset(date=week_start)`.
3. Build the controller (Baseline, Setback, MPC variant, etc.).
4. Loop: while not done, get action from controller, call `env.step(action)`.
5. Save `env.history` to `results/mpc_eval/<timestamp>/raw/<cell_id>.parquet`.
6. Compute metrics (§6.3) and append a row to `results/mpc_eval/<timestamp>/metrics.parquet`.

### 6.3 Metrics

Computed from `env.history` plus the cell's peak window definition:

| Metric | Definition |
|---|---|
| `peak_runtime_min` | minutes during peak windows where `hvac_mode != "off"` |
| `peak_runtime_reduction_pct` | `(baseline.peak_runtime_min - this.peak_runtime_min) / baseline.peak_runtime_min` |
| `peak_energy_kwh` | sum of `hvac_power_kw × dt_h` during peak windows |
| `total_energy_kwh` | sum of `hvac_power_kw × dt_h` over the whole episode |
| `energy_overhead_pct` | `(this.total_energy_kwh - baseline.total_energy_kwh) / baseline.total_energy_kwh` |
| `pre_peak_runtime_min` | minutes during the 2h before each peak window where HVAC is on |
| `comfort_violation_min` | minutes outside `[comfort_lower, comfort_upper]` |
| `comfort_violation_degree_min` | `Σ max(0, deviation_f) × dt_min` |
| `max_violation_f` | maximum instantaneous excursion (signed by direction) |
| `cost_usd` | `Σ hvac_power_kw × dt_h × electricity_price` |
| `peak_cost_usd` | same but limited to peak windows |
| `decisions_phase_<phase>_count` | count of MPC decisions per phase, MPC controllers only |

The Baseline cell for each (building, week, peak_window, comfort_band) combination is the reference for the relative metrics. Compute reductions in a second pass after all cells finish, joining each row to its matching Baseline row.

### 6.4 Manifest

Write `results/mpc_eval/<timestamp>/manifest.json` with: git SHA, ThermalGym version, EnergyPlus version, predictor variant metadata (`PredictorMetadata`), seed, full config, wall-clock time. This is required for paper reproducibility.

### 6.5 Resumability

Each cell writes `cell_id.done` as a marker after success. On re-run, skip cells whose marker exists. EnergyPlus episodes can fail mid-run; the harness must catch and log per-cell exceptions and continue.

---

## 7. Forecast Noise Model

Outdoor-temperature forecasts in practice degrade with horizon. Use the following noise model for `MPC-noisy-forecast`:

```
e_t = phi * e_{t-1} + N(0, sigma)
forecast(t) = epw_truth(t) + e_t
sigma(h_hours) = sigma_0 + sigma_slope * h_hours
```

Defaults: `phi = 0.7`, `sigma_0 = 0.5°F`, `sigma_slope = 0.25 °F/h`, capped at `sigma_max = 4.0°F`. Run **3 noise seeds per cell** so error bars on H5 are meaningful.

Document the choice in the paper as a coarse approximation of NOAA point-forecast statistics. If publishing, cite a concrete source for forecast error scale.

---

## 8. Sensitivity Sweep (H6)

On a representative subset (6 cells: 3 buildings × 2 seasons × 1 peak window), run MPC-perfect with these parameter grids:

- `precondition_margin_minutes ∈ {0, 5, 10, 20, 40}`
- `drift_safety_margin_minutes ∈ {0, 5, 10, 20, 40}`
- `comfort_band_width ∈ {2, 4, 6, 8}` °F (centered on normal setpoint)

Output a heatmap per (peak_runtime_min, comfort_violation_min) for each pair. This shows the comfort/peak frontier.

---

## 9. Statistical Analysis

For each hypothesis:

- **H1, H2, H3.** Per-cell paired comparison of MPC vs Baseline and vs the best rule-based policy. Report: median, IQR, mean, 95% bootstrap CI of the per-cell improvement. Run a Wilcoxon signed-rank test per (climate, season). Correct for multiple comparisons (Holm) across the 4 climate × season strata.
- **H4.** Linear mixed-effects model with `building` as a random effect; fixed effects: predictor variant, forecast kind. Report estimated marginal means for `peak_runtime_min` per variant.
- **H5.** Across the 3 noise seeds × all cells, regress `peak_runtime_min` on `mean_forecast_error_f` per cell. Report slope + CI. Compare against `MPC-perfect` and `PersistenceForecast` extremes.
- **H6.** Plot the comfort-vs-peak Pareto frontier. Identify the elbow.

Add `scripts/analyze_mpc_eval.py` that consumes `metrics.parquet` and emits all tables, figures, and CSVs into `results/mpc_eval/<timestamp>/analysis/`. Outputs include:

- `tables/main_results.csv` (one row per controller per (climate, season)).
- `figures/peak_runtime_box.pdf`.
- `figures/comfort_vs_peak_pareto.pdf`.
- `figures/predictor_variant_comparison.pdf`.
- `figures/forecast_noise_degradation.pdf`.
- `figures/sensitivity_heatmaps.pdf`.

---

## 10. Reproducibility Requirements

- Seeds: every random-using script accepts `--seed`; default 42. Record seed in manifest.
- Pin dependencies via `pyproject.toml` lockfile or `requirements.lock`.
- Pin EnergyPlus version (record in manifest).
- All raw `env.history` parquet files retained (≈ 100MB total at 15-min × 7-day × ~150 cells; fine).
- All trained model pickles retained under `models/variants/<variant>/`.
- A `make reproduce` target (or `scripts/reproduce_paper.sh`) that runs §3, §4, §6, §7, §8, §9 end-to-end. Wall-clock budget: under 24 hours on an 8-core workstation.

---

## 11. Phase-by-Phase Implementation Order

An implementing agent should execute in this order. Each phase ends with a `make` or script that produces a concrete artifact.

1. **P0: Adapter + forecast wrappers** (§5.1, §5.2). Add `mpc/thermalgym_adapter.py` and `mpc/forecast.py` plus tests. Quick sanity run: 1 building × 1 day with MPC-perfect should produce a non-empty history and a non-zero number of preconditioning decisions.
2. **P1: Data generation** (§3). All four parquet datasets exist. Acceptance checks pass.
3. **P2: Predictor variants** (§4). All five variants trained, offline metrics gate passed. Naive predictor implemented.
4. **P3: Evaluation harness** (§6). Single-cell run works end-to-end. Resumability and manifest verified.
5. **P4: Full grid sweep — main results** (§6, controllers 1–5 across the full grid). Produces `metrics.parquet`.
6. **P5: Forecast-noise sweep** (§7). 3 seeds × MPC-noisy across the grid.
7. **P6: Predictor-OOD sweep** (controllers 7–9 across the grid).
8. **P7: Sensitivity sweep** (§8).
9. **P8: Analysis + figures** (§9).
10. **P9: Paper-ready writeup**: one combined `docs/MPC_EVALUATION_RESULTS.md` summarizing all hypotheses with embedded tables and figure references; per-section CSVs/PDFs in `results/mpc_eval/<ts>/analysis/`.

---

## 12. Budget and Risks

**Compute.** 9 buildings × 4 weeks × 2 peak windows × ~9 controllers × 3 seeds (noise only on 1 controller) ≈ ~700 cells. At ~10 minutes per cell with `timestep_minutes=15`, ≈ 120 hours single-threaded; ≈ 15 hours on 8 workers. Plan accordingly.

**Risks and mitigations.**

- *EnergyPlus / IDF / EPW files missing.* P0 should fail loudly. Resolve by following `thermalgym/data/buildings/README.md` and `thermalgym/data/weather/README.md` before P1.
- *Predictor pickle schema drift.* Lock the artifact format in `mpc/model_interfaces.py`. Add a versioned `artifact_version` key.
- *Training data leakage.* Enforce the date-buffer assertion in `generate_episodes` (P1 acceptance).
- *EnergyPlus crashes.* Harness must isolate per cell (subprocess), capture stderr, mark failures, and continue.
- *Cooling-disabled hot-climate winter weeks.* Document the asymmetry in the methodology section instead of forcing apples-to-apples.
- *Per-home residuals.* The current XGB predictor has Ecobee `home_id` residuals. Synthetic ThermalGym building IDs will not be in those tables — that's intended; the predictor falls back to the global model. Verify this path in P0.

---

## 13. Open Questions to Resolve Before Writing the Paper

1. Should the paper claim generalization to *real* homes? If so, add a real-home back-test using held-out Ecobee homes via a learned-thermal-model rollout (much harder; out of scope of this plan unless explicitly added).
2. Should we include a published-MPC comparator (e.g., a nonlinear MPC over an RC model)? Useful for stronger framing but doubles implementation cost.
3. Tariff sensitivity: is the CA TOU price the only one we evaluate, or do we add a more aggressive tariff (e.g., Texas wholesale-pass-through) for an economic-savings result?
4. Reporting unit: peak runtime *minutes per home-day* vs total reduction. Decide before running analysis so figures are consistent.
