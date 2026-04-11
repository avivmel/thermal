# Implementation Plan — Multi-Peak Heat/Cool MPC Controller

Derived from `plans/MULTI_PEAK_MPC_SRS_PROMPT.md`. This plan turns the SRS
brief into an actionable build plan for a receding-horizon controller that
minimizes HVAC runtime inside configurable peak windows while respecting
comfort bounds, works symmetrically for heating and cooling, and is fully
introspectable via a JSON plan log.

---

## 0. Context snapshot

Relevant files already in the repo (read before implementing):

- `mpc/model_interfaces.py` — `FirstPassagePredictor` interface,
  `FittedFirstPassagePredictor` (ridge, log-duration target),
  `PhysicsFallbackPredictor`. **Currently fits heating direction only**
  (`change_type == "heat_increase"`, `drift_direction == "warming_drift"`).
- `thermalgym/env.py` — `ThermalEnv` EnergyPlus-backed simulator. 5-min
  resolution, °F, action `{heat_setpoint, cool_setpoint}`, obs includes
  `timestamp, indoor_temp, outdoor_temp, hvac_power_kw, hvac_mode,
  heat_setpoint, cool_setpoint, electricity_price, hour, day_of_week, month`.
  History available via `env.history` as a DataFrame.
- `thermalgym/policies.py` — `Baseline`, `PreHeat`, `PreCool`, `Setback`,
  `PriceResponse`. Comparison baselines for evaluation.
- `thermalgym/buildings.py` — nine-building registry across cold/mixed/hot
  climate zones, three vintages, heatpump/AC.
- `data/setpoint_responses.parquet` — active-episode data, both
  `heat_increase` and `cool_decrease` rows.
- `data/drift_episodes.parquet` — passive-drift data, both `warming_drift`
  and `cooling_drift` rows.

Env specifics that affect the controller:

- Setpoints are clipped inside `ThermalEnv.step()` to
  `HEAT_MIN=55, HEAT_MAX=75, COOL_MIN=70, COOL_MAX=85`.
- The env is **not** a Gymnasium env; a policy is just
  `callable(obs) -> action`. The MPC controller is a policy class.
- Default `run_period_days=1`, so evaluation defaults to single-day
  episodes; state does not carry between calls to `reset()`.
- `hvac_mode` is `"heating" / "cooling" / "off"` — this is the signal used
  to compute the primary metric (runtime minutes inside peak windows).

---

## 1. Deliverables

### New files
- `mpc/controller.py` — `MultiPeakController` policy + `ControllerConfig`
  + `PeakWindow` dataclass + situation classifier.
- `mpc/planner.py` — precondition enumerate-and-score planner (pure
  functions, no state).
- `mpc/plan_log.py` — `StepPlan` dataclass, reason-code enum, JSON
  serialization helpers.
- `mpc/evaluate.py` — `run_episode`, `peak_runtime_minutes`, other
  metrics, `compare_policies`.
- `tests/test_mpc_controller.py` — unit tests using a `StubPredictor`.
- `tests/test_mpc_planner.py` — tests the pure planner logic in isolation.
- `tests/test_mpc_evaluate.py` — tests metric functions on synthetic
  histories.
- `scripts/run_mpc_eval.py` — evaluation driver: runs baselines + MPC
  across the building × season × peak-shape matrix, emits a markdown
  report and per-run JSON plan logs.

### Extended files
- `mpc/model_interfaces.py` — cooling-direction fits, `direction`
  parameter on prediction methods, symmetric physics fallback. Backwards
  compatible: existing callers still work.

### Generated outputs (not committed by default)
- `outputs/mpc_eval/<timestamp>/summary.md` — side-by-side metrics table.
- `outputs/mpc_eval/<timestamp>/<building>_<mode>_<date>/plan_log.json` —
  per-run JSON plan logs.
- `outputs/mpc_eval/<timestamp>/<building>_<mode>_<date>/history.parquet`
  — raw per-step traces.

---

## 2. Work streams

Numbered by dependency order. Each stream is meant to land in its own PR
so reviewers can evaluate it on its own.

### Stream A — Predictor cooling symmetry (blocker)

**Goal**: make `FirstPassagePredictor` usable in both heating and cooling
modes so downstream controller logic can treat direction as a config flag.

Steps:

1. Introduce `Direction = Literal["heating", "cooling"]`.
2. Add `direction: Direction = "heating"` kwarg to
   `FirstPassagePredictor.predict_heat_time` (rename visible API to
   `predict_active_time`) and `predict_drift_time`. Keep
   `predict_heat_time` as a thin alias that forwards to
   `predict_active_time(direction="heating")`. Do **not** delete it —
   nothing depends on the name externally, but the alias keeps the
   refactor contained.
3. Sign semantics for cooling direction:
   - `predict_active_time(current, target, …, direction="cooling")`
     requires `target < current`; "gap" is `max(current - target, 0)`.
   - `predict_drift_time(current, boundary, …, direction="cooling")`
     requires `boundary > current` (passive warming up toward boundary);
     "margin" is `max(boundary - current, 0)`.
   - The active feature vector for cooling uses the same columns as
     heating but with the gap computed with the sign flipped. Reuse
     `_active_feature_vector` by passing an abs-gap and a direction
     indicator feature, or introduce a sibling `_cool_feature_vector`.
     Prefer a single builder with a `direction` parameter — avoids
     drift between the two.
4. New training-row preparers:
   - `_prepare_cool_active_training_rows` — filters
     `change_type == "cool_decrease"`, duration clamp same as heating.
   - `_prepare_cooling_drift_rows` — filters
     `drift_direction == "cooling_drift"`, `crossed_boundary == True`,
     `time_to_boundary_min` in `[5, 480]`.
5. `FittedFirstPassagePredictor` now holds four ridge models:
   `heat_active`, `cool_active`, `warm_drift`, `cool_drift`. Routing:
   prediction method picks the model by direction. Falls back to
   `PhysicsFallbackPredictor` on any side where fitting failed (mirrors
   current behavior).
6. Extend `PhysicsFallbackPredictor`:
   - Add `cooling_rate_f_per_hour` (default ~2.5 — cooling is slightly
     faster than heating per CLAUDE.md: 28% heating vs cooling
     difference), and `passive_warming_rate_per_hour` for cooling drift.
   - `predict_active_time` and `predict_drift_time` branch on direction.
7. `PredictorMetadata` gains `cool_active_rows`, `cool_drift_rows`,
   `source_cool_active`, `source_cool_drift`.
8. Sanity-log: in `from_local_data`, print (or return) held-out MAE on a
   random 10% of each split. Expected ~20 min active, larger drift. If
   cooling MAE is dramatically worse (>40 min), annotate the metadata
   (`cooling_degraded=True`) so the controller can raise its default
   `uncertainty_scale` — never blocks.

**Acceptance**:
- `FittedFirstPassagePredictor.from_local_data()` returns a predictor
  whose four methods work for both directions without raising.
- Existing `predict_heat_time`/`predict_drift_time` calls still return
  the same values (pure refactor; alias preserves behavior).
- `tests/test_model_interfaces.py` covers: heating direction still
  matches prior outputs on a canned input; cooling active / drift
  return plausible positive durations on canned inputs; physics
  fallback used when parquet files are missing.

### Stream B — Controller config and situation classifier

**Goal**: deterministic, pure-function pieces that are trivial to test.

1. `ControllerConfig` dataclass (frozen):
   ```python
   @dataclass(frozen=True)
   class PeakWindow:
       start_hour: float  # [0, 24), fractional allowed
       end_hour: float    # (start_hour, 24]

   @dataclass(frozen=True)
   class ControllerConfig:
       mode: Direction                                # "heating" | "cooling"
       peak_windows: tuple[PeakWindow, ...]
       comfort_low: float                             # hard lower bound °F
       comfort_high: float                            # hard upper bound °F
       normal_heat_sp: float
       normal_cool_sp: float
       precondition_offsets: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
       safety_margin_minutes: float = 15.0
       uncertainty_scale: float = 1.3
       precondition_lookahead_hours: float = 3.0
       temp_margin: float = 0.5                       # never pre-cond within this of opposite bound
       truncate_coverage_at_next_window: bool = True
       setback_heat_sp: float | None = None           # default: comfort_low
       setback_cool_sp: float | None = None           # default: comfort_high
   ```
2. Validation (raise `ValueError` in `__post_init__`):
   - `comfort_low < comfort_high`
   - `normal_heat_sp < normal_cool_sp`
   - `comfort_low ≤ normal_heat_sp` and `normal_cool_sp ≤ comfort_high`
   - Peak windows: each `start_hour < end_hour`, each ⊂ `[0, 24]`,
     no two windows overlap (sort + compare). Adjacent is allowed.
   - `precondition_offsets` strictly positive and sorted ascending.
   - `safety_margin_minutes ≥ 0`, `uncertainty_scale ≥ 1.0`.
3. `Situation` enum:
   `IDLE`, `PRE_PEAK`, `IN_PEAK`, `BETWEEN_PEAKS_PRE_PEAK`,
   `BETWEEN_PEAKS_IDLE`. Motivation: "between peaks" collapses to
   either "plan for the next one" or "hold normal" — the distinction
   is load-bearing for the plan-log reason code.
4. `classify_situation(now_h: float, config) -> (Situation, window_idx)`:
   - `IN_PEAK` if any window contains `now_h`.
   - Else locate next window (smallest `start_hour > now_h` modulo 24).
     If `minutes_until_next ≤ precondition_lookahead_hours * 60`, then
     either `PRE_PEAK` (no prior window today) or
     `BETWEEN_PEAKS_PRE_PEAK` (there is one).
   - Else `IDLE` or `BETWEEN_PEAKS_IDLE`.
5. Helper `minutes_until(window_start_h, now_timestamp)` — returns
   minutes to next occurrence (handles day-wrap via the timestamp's
   date, not modular arithmetic on floats, to avoid DST surprises).

**Acceptance**:
- `tests/test_mpc_controller.py::TestSituationClassifier` exercises
  every state transition for the two-window case
  `[(7, 9), (17, 20)]` at every hour of the day.
- Invalid configs raise at construction time with specific messages.

### Stream C — Precondition planner (pure)

**Goal**: enumerate-and-score planner that is a pure function of
`(indoor_temp, outdoor_temp, timestamp, window, predictor, config)`.

1. Entry point:
   ```python
   def plan_precondition(
       obs: dict,
       window: PeakWindow,
       config: ControllerConfig,
       predictor: FirstPassagePredictor,
       now: pd.Timestamp,
       next_window: PeakWindow | None,
   ) -> PreconditionDecision
   ```
2. `PreconditionDecision` dataclass:
   ```python
   @dataclass
   class Candidate:
       offset: float          # from normal setpoint, °F
       target_temp: float     # after clamping to comfort bound - temp_margin
       t_active_min: float    # predictor estimate (unscaled)
       t_drift_min: float     # predictor estimate, scaled by 1/uncertainty_scale
       feasible: bool
       coverage_min: float    # min(t_drift, peak_duration [, clipped])
       score: float           # = coverage (primary), tie-break by -offset
       reject_reason: str | None

   @dataclass
   class PreconditionDecision:
       candidates: list[Candidate]
       chosen: Candidate | None
       chosen_reason: str        # reason code
   ```
3. For heating mode (cooling symmetric — mirror sign + bound):
   - `peak_start = now.replace(hour=int(window.start_hour), minute=...)`
     — use fractional-hour → `pd.Timedelta` conversion.
   - `minutes_until_peak = (peak_start - now).total_seconds() / 60`.
   - `peak_duration = (window.end_hour - window.start_hour) * 60`.
   - For each offset `Δ`:
     - `target_temp = min(config.normal_heat_sp + Δ, config.comfort_high - config.temp_margin)`
     - If `target_temp ≤ obs.indoor_temp + 1e-3`: mark as "already
       there," feasible, `t_active = 0`, proceed to drift.
     - Else `t_active = predictor.predict_active_time(..., direction="heating")`.
     - `feasible = t_active + config.safety_margin_minutes ≤ minutes_until_peak`.
     - `t_drift = predictor.predict_drift_time(target_temp, comfort_low,
       outdoor, peak_start, direction="heating") / config.uncertainty_scale`.
     - `coverage = min(t_drift, peak_duration)`; if
       `truncate_coverage_at_next_window` and `next_window`, further
       clip coverage so it does not extend into `next_window`.
     - `score = coverage - 0.01 * Δ` (tie-break: prefer smallest Δ).
4. Selection:
   - Among feasible candidates, pick highest score. Ties → smallest Δ.
   - If no feasible: `chosen = None`, reason
     `"PRECONDITION_INFEASIBLE"`.
   - If only `Δ = 0` (or "no preconditioning needed" — e.g. indoor is
     already at/above target): chosen with reason
     `"PRECONDITION_NOT_NEEDED"`.
5. Cooling mirror:
   - `target_temp = max(normal_cool_sp - Δ, comfort_low + temp_margin)`
   - `t_drift = predict_drift_time(target_temp, comfort_high, …,
     direction="cooling")`
   - The planner is a single function with a mode switch, not a
     duplicated cooling variant — symmetry is enforced by construction.

**Acceptance** (all against `StubPredictor`):
- Stub returns huge `t_active` → no feasible candidate → reason
  `PRECONDITION_INFEASIBLE`.
- Stub returns small `t_active` and large `t_drift` for every target →
  planner selects the largest offset that fits in
  `comfort_high - temp_margin`.
- Stub returns `t_drift` equal to half `peak_duration` → coverage
  correctly capped to `t_drift`.
- With `truncate_coverage_at_next_window=True`, for
  `peak_windows=[(17,18),(18.5,19.5)]` the first-window candidate's
  coverage is at most 60 min, not 150 min (even if drift would reach).

### Stream D — `MultiPeakController` glue and plan log

**Goal**: wire streams B and C into a policy class that (a) implements
`__call__(obs) -> action_dict`, (b) logs a structured plan for every
step.

1. `MultiPeakController.__init__(self, config, predictor)` — stores
   both, initializes `self._plan_log: list[dict]`.
2. `__call__(self, obs: dict) -> dict`:
   - `now = obs["timestamp"]`
   - `now_h = now.hour + now.minute / 60`
   - `situation, window_idx = classify_situation(now_h, self.config)`
   - Dispatch:
     - `IDLE` / `BETWEEN_PEAKS_IDLE` → normal setpoints, reason
       `IDLE_NORMAL` or `BETWEEN_PEAKS_NORMAL`.
     - `PRE_PEAK` / `BETWEEN_PEAKS_PRE_PEAK` → call `plan_precondition`,
       command chosen target (or normal if `chosen is None`).
     - `IN_PEAK` → command setback setpoints, then run safety check:
       - `t_breach = predictor.predict_drift_time(indoor,
         comfort_bound, outdoor, now, direction=config.mode) / uncertainty_scale`
       - If `t_breach < config.safety_margin_minutes`: override to
         normal setpoints, reason
         `PEAK_SETBACK_ABORTED_COMFORT`. Otherwise reason
         `PEAK_SETBACK`.
3. Action construction (heating-mode example):
   - Preconditioning: `heat_sp = target_temp, cool_sp = normal_cool_sp`
     (never both adjusted — avoids deadband cross).
   - In-peak setback: `heat_sp = setback_heat_sp (default comfort_low),
     cool_sp = normal_cool_sp`.
   - Cooling mirror: only `cool_sp` moves; `heat_sp` stays at
     `normal_heat_sp`.
4. Always clamp both setpoints to `[comfort_low, comfort_high]` before
   returning. The env does its own clamp but that is a second line of
   defense, not the primary one.
5. Plan log entry (JSON-serializable — no numpy scalars, no timestamps
   except as ISO strings):
   ```python
   {
     "step_index": int,
     "timestamp": "2017-07-01T17:05:00",
     "situation": "IN_PEAK",
     "active_window_idx": 1,
     "minutes_until_next_peak": None,
     "observed": {
       "indoor_temp": 76.3,
       "outdoor_temp": 91.4,
       "hvac_mode": "cooling",
       "heat_setpoint_prev": 68.0,
       "cool_setpoint_prev": 76.0
     },
     "candidates": [
       {"offset": 0.5, "target_temp": 75.5, "t_active_min": 12.4,
        "t_drift_min": 44.1, "feasible": True, "coverage_min": 44.1,
        "score": 44.095, "reject_reason": None, "chosen": False},
       ...
     ],
     "chosen": {"heat_sp": 68.0, "cool_sp": 80.0, "reason": "PEAK_SETBACK"},
     "predictor_metadata": {"active_rows": 12032, "drift_rows": 4501,
                            "source_active": "data/setpoint_responses.parquet",
                            "source_drift": "data/drift_episodes.parquet"}
   }
   ```
6. `plan_log` property returns a deep copy so callers cannot mutate
   internal state.
7. Reason code enum (string values, stable — these end up in tests):
   `IDLE_NORMAL`, `BETWEEN_PEAKS_NORMAL`, `PRECONDITION_CHOSEN`,
   `PRECONDITION_INFEASIBLE`, `PRECONDITION_NOT_NEEDED`, `PEAK_SETBACK`,
   `PEAK_SETBACK_ABORTED_COMFORT`.

**Acceptance**:
- `json.dumps(controller.plan_log)` never raises.
- Running the same policy on the same env twice yields byte-identical
  plan logs (determinism).
- For every step, the commanded `heat_sp`/`cool_sp` can be derived from
  the plan log alone (no code reading required).

### Stream E — Evaluation harness

**Goal**: one script that runs every baseline + MPC across the matrix
and produces a single comparison table.

1. `mpc/evaluate.py`:
   - `run_episode(env, policy) -> pd.DataFrame`:
     - `obs = env.reset(date)`; loop `env.step(policy(obs))` until
       `env.done`; return `env.history`.
   - `peak_runtime_minutes(history, peak_windows, timestep_minutes)`:
     - Primary metric. Count rows where `hvac_mode != "off"` and the
       row's hour-of-day falls inside any window. Multiply by
       `timestep_minutes`.
   - `comfort_violation_minutes(history, comfort_low, comfort_high,
     timestep_minutes)` — count rows where `indoor_temp` is outside
     the bounds.
   - `energy_kwh(history, timestep_minutes)` —
     `(hvac_power_kw * timestep_minutes / 60).sum()`.
   - `peak_window_energy_kwh(history, peak_windows, timestep_minutes)`.
   - `tou_cost(history, timestep_minutes)` — uses
     `electricity_price` column.
   - `setpoint_change_count(history)` — diagnostic for how twitchy the
     controller is.
   - `metric_row(name, history, config)` → dict of all metrics for one
     run.
2. `compare_policies(env_factory, policies, date) -> pd.DataFrame`:
   - Takes an `env_factory` (callable that returns a fresh env) so
     each policy runs on an isolated episode — avoids state leakage.
   - Returns one row per policy.
3. `scripts/run_mpc_eval.py`:
   - Matrix:
     - Buildings: full `BUILDINGS` registry (9 homes).
     - Modes: `"heating"` on 2017-01-15, `"cooling"` on 2017-07-15.
       (Winter in cold zone, summer in hot zone — pick dates where
       mode matters.)
     - Peak-window shapes: `[(17,20)]`, `[(7,9),(17,20)]`,
       `[(17,18),(18.5,19.5)]` (closely-spaced stress test).
   - Policies: `Baseline`, `PreHeat` (heating mode), `PreCool`
     (cooling mode), `Setback`, `MultiPeakController`.
   - Output: `outputs/mpc_eval/<timestamp>/summary.md` — markdown
     table grouped by (building, mode, peak-shape), one column per
     policy, rows for each metric. Per-run artifacts in
     `outputs/mpc_eval/<timestamp>/<run_id>/`.

**Acceptance**:
- `run_mpc_eval.py` runs end-to-end without error on at least one
  building/mode pair.
- `peak_runtime_minutes` on a handcrafted history with two peak
  windows and known off/on rows returns the expected value (unit
  test).
- For a simple handcrafted policy that pre-heats one degree before a
  17–20 peak, the MPC controller's peak runtime is less-than-or-equal
  to `PreHeat`'s on at least one building (smoke-level sanity — full
  success criteria below).

### Stream F — Unit tests

Two levels.

**Pure-logic tests** (no env, fast, 100% coverage expected):

1. `test_mpc_controller.py::TestSituationClassifier`:
   - Zero peaks → always `IDLE`.
   - Single peak `[(17,20)]` at 05:00, 15:30, 17:00, 19:59, 20:00, 22:00
     → expected situations.
   - Two peaks `[(7,9),(17,20)]` at every hour.
   - Closely-spaced `[(17,18),(18.5,19.5)]` at 17:30, 18:15, 18:45.
2. `test_mpc_controller.py::TestConfigValidation`:
   - Overlapping windows raise.
   - `comfort_low >= comfort_high` raises.
   - `normal_cool_sp < normal_heat_sp` raises.
   - Inverted individual window raises.
   - Non-ascending offsets raise.
3. `test_mpc_planner.py::TestPlanner` (uses `StubPredictor`):
   - Big `t_active` → `PRECONDITION_INFEASIBLE`.
   - Tiny `t_active`, big `t_drift` → picks largest feasible offset
     (capped by `comfort_high - temp_margin`).
   - Moderate `t_active`, `t_drift ≈ peak_duration` → coverage equals
     peak duration (not larger).
   - `truncate_coverage_at_next_window` clips coverage at next
     window start.
   - Cooling mode: feeds symmetric inputs, asserts chosen target is
     `normal_cool_sp - chosen_offset` (not `+`).
4. `test_mpc_controller.py::TestSymmetry`:
   - Build a heating config and a cooling config that are mirrored
     through the normal setpoints. Feed mirrored obs (indoor/outdoor
     reflected). Assert the chosen offset magnitude is identical and
     the reason codes match. This is the explicit test for FR
     "operating-mode symmetry."
5. `test_mpc_controller.py::TestPeakSetbackSafety`:
   - In-peak, stub returns `t_drift / uncertainty_scale < safety_margin`
     → controller overrides to normal setpoints, reason
     `PEAK_SETBACK_ABORTED_COMFORT`.
6. `test_mpc_controller.py::TestPlanLogJson`:
   - `json.dumps(controller.plan_log)` does not raise.
   - All numeric fields are plain `float`/`int` (no `np.float64`).
7. `test_mpc_controller.py::TestDeterminism`:
   - Run same obs sequence twice, assert `plan_log` byte-identical
     after `json.dumps`.
8. `test_mpc_evaluate.py::TestMetrics`:
   - Handcrafted histories exercise `peak_runtime_minutes`,
     `comfort_violation_minutes`, `energy_kwh`, `tou_cost`.

**Integration smoke tests** (opt-in, requires EnergyPlus):
9. `test_mpc_controller.py::TestShortRun`:
   - Build `ThermalEnv` (smallest building), `MultiPeakController` with
     `PhysicsFallbackPredictor` (no parquet needed), 1-day episode.
   - Assert: no comfort violations; `len(plan_log) == number_of_steps`;
     peak runtime ≤ `Baseline`'s on the same day.

### Stream G (optional) — Home-specific predictor
Left for a follow-up if the cooling-direction predictor MAE is too
large. Idea: pass `home_id` through (already plumbed in the interface
signature), train per-building residuals. Not required for v1.

---

## 3. Key design decisions encoded in this plan

1. **Enumerate-and-score planner over rollout.** Deterministic,
   cheap, trivially unit-testable, and the planner output maps 1:1 to
   the plan-log entries. Rollout would be more flexible but much
   harder to verify.
2. **Direction as a parameter, not a subclass.** Heating/cooling
   symmetry is enforced by a single code path with a `direction` flag.
   This is the load-bearing mechanism for the SRS "operating-mode
   symmetry" requirement — symmetry comes for free from the
   implementation shape.
3. **Safety is a single knob.** `uncertainty_scale` divides every
   drift prediction; `safety_margin_minutes` adds a floor. Tuning
   conservativeness is one number, not a tangle of if-statements.
4. **Closely-spaced windows: coverage truncates at next window.**
   Stated as an explicit config flag, default on. Alternative ("let
   them overlap") is wrong because the pre-condition would get
   interrupted anyway.
5. **Setback target = comfort bound, not normal setpoint with
   offset.** Simplest symmetric rule: during peak, command the worst
   setpoint the comfort bound allows, then rely on the safety check
   to claw it back if drift is imminent.
6. **Controller clamps to comfort bounds internally.** The env also
   clamps, but to the wider `HEAT_MIN…HEAT_MAX` / `COOL_MIN…COOL_MAX`
   range. The controller's own clamp is the primary safety.
7. **Plan log is the source of truth.** Any reviewer question
   ("why did the controller do that?") must be answerable from
   `plan_log[step_index]` alone.
8. **Single-day episodes for evaluation.** Matches env default, matches
   the problem statement (morning+evening peaks within one day),
   avoids multi-day state carryover confusions. Multi-day comes later.

---

## 4. Risks and open questions

1. **Cooling-direction predictor accuracy is unknown.** If the fitted
   cool MAE is much worse than heating (>40 min), the default
   `uncertainty_scale` may need to be higher for cooling mode.
   Mitigation: Stream A reports cooling MAE; controller raises
   `uncertainty_scale` automatically if `predictor.metadata` flags the
   cool side as degraded.
2. **Pre-condition candidate discretization.** 0.5°F steps × 6
   candidates = 6 predictor calls per step in PRE_PEAK. Fine for <50
   ms latency budget, but if we ever want 0.1°F resolution, we'd want
   a bisection search instead of enumeration.
3. **"Runtime" metric is a proxy.** `hvac_mode != "off"` on a 5-min
   obs counts the entire 5-min interval as "on." The true compressor
   duty cycle is shorter. Acceptable for ThermalEnv evaluation, but
   document as a limitation; real field deployment would need
   `hvac_power_kw > ε` or actual runtime fraction.
4. **Price-aware optimization is out of scope.** The primary metric is
   peak runtime, not TOU cost. We report cost as a secondary, but do
   not optimize for it. If cost becomes the goal later, the planner's
   scoring function grows a cost term.
5. **Back-to-back peaks with a gap shorter than the preconditioning
   horizon.** Current plan: each window has its own lookahead,
   evaluated independently, and `truncate_coverage_at_next_window`
   prevents over-crediting. Edge case: if the gap is shorter than the
   precondition time for the second window, the controller may fail
   the second window. Document, don't fix in v1.
6. **Heat pumps in cold climates can be capacity-limited.** The fitted
   model may not capture this well. If evaluation shows the controller
   commands pre-heat targets the HVAC cannot actually reach in the
   predicted time, we'll need to add a "predicted unreachable" reject.
   Watch for it in Stream E results.
7. **Determinism around timestamp arithmetic.** Fractional peak hours
   (e.g., 18.5) plus DST-adjacent dates could give weird results if we
   use modular arithmetic on `float` hours. The plan uses
   timestamp-based computation (`peak_start = now.normalize() +
   Timedelta(hours=window.start_hour)`) to avoid this; tests cover at
   least one DST-boundary date.

---

## 5. Success criteria (v1 of the controller)

All of the following must hold on the evaluation matrix in Stream E:

- **No comfort violations** on any run with
  `comfort_low = 66, comfort_high = 78`, `safety_margin_minutes = 15`,
  `uncertainty_scale = 1.3`.
- **Primary metric**: `MultiPeakController` peak-window runtime is at
  least **20% lower** than the best of `{Baseline, PreHeat/PreCool,
  Setback}` on at least **6 of 9 buildings** in each mode.
- **Explainability**: every commanded setpoint in every plan log can
  be justified by the entry's `candidates` + `chosen_reason` alone.
- **Determinism**: rerunning the full eval with the same seeds
  produces byte-identical summary tables.
- **Latency**: per-step controller call < 50 ms on a dev laptop
  (measured in Stream F's integration test; hard-fail if it
  regresses >3×).

If any criterion fails, tune `safety_margin_minutes`,
`uncertainty_scale`, or `precondition_offsets` rather than restructuring
the planner. If tuning can't close the gap, the failure mode goes into
Section 4 as a documented risk and v2 planning.

---

## 6. Execution order (PR sequence)

1. **PR 1 — Stream A**: predictor cooling symmetry + tests for
   `model_interfaces.py`. Small, mechanical, isolated.
2. **PR 2 — Streams B + C + D + F (pure-logic tests)**: controller,
   planner, plan log, and all unit tests. Self-contained behind
   `StubPredictor`; no EnergyPlus dependency.
3. **PR 3 — Stream E**: `evaluate.py`, `run_mpc_eval.py`, integration
   smoke test. Produces the first real numbers.
4. **PR 4 — Tuning**: based on PR 3 results, adjust defaults in
   `ControllerConfig` and re-run. Docs-only PR if no config changes
   needed.

Each PR is independently reviewable; none requires the next to be
functional on its own.

---

## 7. Out of scope (explicit non-goals for v1)

- Multi-day episodes with carryover state.
- TOU-cost optimization in the planner's objective.
- Per-home residual fits (`home_id` is plumbed through the interface
  but unused by the controller).
- Learning-based or rollout-based planners.
- Real-building deployment concerns (network jitter, sensor dropouts,
  thermostat UI feedback).
- Uncertainty quantiles from the predictor — v1 treats predictions as
  point estimates and scales them with a single `uncertainty_scale`.
  Adding P50/P90 outputs is a future predictor upgrade, not a
  controller change.
