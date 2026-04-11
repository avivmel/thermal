# Prompt — Draft a SRS for a Multi-Peak, Heat/Cool HVAC MPC Controller

You are drafting a **Software Requirements Specification (SRS)** for a new
model-predictive HVAC controller in the `thermal` project. You are writing
requirements, not an implementation. Design the controller fresh from the
problem statement and the interfaces below — do not anchor on any prior
controller in this repo. If you find older designs while exploring, treat
them only as examples of the problem shape, not as templates.

Write the SRS to `plans/MULTI_PEAK_MPC_SRS.md`. When in doubt, prefer fewer
requirements stated clearly over many requirements stated vaguely.

---

## 1. Problem statement

The target system is a smart-thermostat controller for a single residential
home participating in a demand-response program. Utilities define one or
more **peak windows** per day (typically two: a morning block and an
evening block) during which reducing HVAC runtime is valuable. Outside
peak windows the home should be held inside a normal comfort band with
normal setpoints.

The controller must:

1. Support **multiple peak windows per day** at arbitrary, configurable
   clock-hour intervals. Two windows is the typical case; the design
   should not hard-code that number.
2. Support **both heating and cooling** operation. A single config must
   select the mode; the controller's logic should be symmetric between
   the two (pre-heat vs. pre-cool; drift toward lower vs. upper comfort
   bound).
3. Respect hard occupant comfort bounds at all times, even during
   pre-conditioning and peak setback.
4. Be receding-horizon: re-plan at every control step from the latest
   observation.

The **primary evaluation metric** is total HVAC runtime (in minutes)
accumulated inside configured peak windows over an evaluation period. The
controller should minimize this quantity subject to comfort constraints.
Secondary metrics are left to the SRS author to propose.

## 2. Available building blocks

The SRS should treat the following as given. Read the referenced files to
understand their exact contracts before writing requirements that depend
on them.

### 2.1 First-passage thermal models
`mpc/model_interfaces.py` defines a `FirstPassagePredictor` interface
exposing two kinds of predictions:

- **Active time**: given current indoor temp, a target temp, outdoor
  temp, timestamp, and whether HVAC is currently running, predict the
  minutes until indoor reaches target with HVAC actively driving it.
- **Drift time**: given current indoor temp, a boundary temp, outdoor
  temp, and timestamp, predict the minutes until indoor crosses the
  boundary with HVAC off.

Read that file to see the exact signatures, the fitted-ridge
implementation (`FittedFirstPassagePredictor`), and the
`PhysicsFallbackPredictor`. The current fits are heating-direction only;
cooling-direction data is available in
`data/setpoint_responses.parquet` (`change_type == "cool_decrease"`)
and `data/drift_episodes.parquet` (`drift_direction == "cooling_drift"`).
Your SRS may require the interface and/or fits to be extended to cover
both modes. See `CLAUDE.md` for dataset and model context.

These models have non-trivial error (roughly 20 min MAE on active time,
larger on drift). The SRS should treat their outputs as point estimates
with meaningful uncertainty and require the controller to carry explicit
safety margins.

### 2.2 Simulation environment
`thermalgym/env.py` defines `ThermalEnv`, an EnergyPlus-backed simulator.
A policy is any callable `obs -> action_dict`:

- `obs` fields include: `timestamp`, `indoor_temp` (°F), `outdoor_temp`
  (°F), `hvac_power_kw`, `hvac_mode` (`"heating"` / `"cooling"` / `"off"`),
  `heat_setpoint`, `cool_setpoint`, `electricity_price`, `hour`,
  `day_of_week`, `month`.
- `action` fields: `{"heat_setpoint": float, "cool_setpoint": float}`.

The env runs a full simulation day at 5-minute resolution by default.
Temperatures are in Fahrenheit. Setpoints are clamped to safety bounds
by the env itself.

Read `thermalgym/env.py`, `thermalgym/policies.py`, and `thermalgym/buildings.py`
to confirm the exact observation shape, the baseline and preheat policies
already available for comparison, and the building registry.

### 2.3 Project context
Read `CLAUDE.md` for project goals, dataset details, and prior modeling
results. This is background, not design input.

## 3. Scope of the SRS

The SRS you produce SHALL cover:

- **Configuration model**: all parameters the operator can set, including
  mode (heating/cooling), the list of peak windows, comfort bounds,
  normal setpoints, safety margins, and anything else your design needs.
  Validation rules for the config (overlapping windows, inverted bounds,
  etc.) belong here.
- **Operating-mode symmetry**: how heating and cooling share logic. State
  what is symmetric and what is not.
- **Peak window semantics**: how windows are specified (e.g., clock
  hours, fractional hours, per-day), how the controller selects which
  window it is planning for on a given step, and how closely-spaced
  windows are handled (note this as an explicit decision, not an accident).
- **Per-step control logic**: what the controller does in each distinct
  situation (before first peak, mid-peak, between peaks, after last
  peak, comfort violation imminent, etc.).
- **Planning semantics**: what "planning" means for this controller.
  Whether the design is enumerate-and-score, trajectory rollout, or
  something else is up to you, but the SRS must specify the contract
  clearly enough that two independent implementers would produce
  equivalent behavior on standard cases.
- **Safety / feasibility**: every scenario where the controller must
  refuse to pre-condition or must abort a pre-condition action.
- **Interface contracts**: predictor, observation, action, and plan
  introspection. Be explicit about what fields the controller reads and
  writes.
- **Observability**: what the controller must expose after each step so
  that a developer can fully explain the commanded setpoint from the
  plan log (no code reading required). Plan data must be
  JSON-serializable.
- **Non-functional requirements**: determinism, per-step latency budget,
  testability (the controller must be unit-testable with a stub
  predictor), and backwards compatibility if relevant.
- **Evaluation requirements**: how the primary metric (peak-window
  runtime) is computed from a simulation trace, what secondary metrics
  must be reported, what comparison baselines the evaluation must
  include, and what success criteria signal that v1 of the spec'd
  system is working.
- **Out-of-scope / non-goals**: state them explicitly.
- **Open questions**: list design decisions you were not willing to
  pin down in requirements and explain why.

The SRS SHOULD NOT:

- Prescribe a specific algorithm class (DP, MILP, rule enumeration,
  MPC rollout) unless the choice is load-bearing for a requirement.
- Hard-code coefficients or thresholds beyond illustrative defaults.
- Specify file layout, class names, or method signatures beyond what is
  needed to state an interface contract.

## 4. Style and format

- Use numbered, testable requirements (`FR-1`, `FR-2`, …, `NFR-1`, …,
  `ER-1`, …). Each requirement is a single statement; rationale goes in
  a sub-paragraph, not inside the requirement.
- Use "SHALL" for binding requirements and "SHOULD" for recommendations.
- Define every term that is not standard HVAC / controls vocabulary in a
  Definitions section at the top.
- When a requirement depends on an external interface, name the file and
  symbol so a reader can verify the contract without searching.
- Length target: thorough but not bloated. Favor requirements that a
  reviewer could turn directly into acceptance tests.

## 5. Process

1. Read the files listed in section 2 to pin down the existing
   interfaces. Do not read or cite any existing controller implementation
   in this repo — you are specifying a new controller from first
   principles.
2. Draft the SRS at `plans/MULTI_PEAK_MPC_SRS.md`.
3. Before finishing, self-check: can every requirement be verified from
   either (a) reading the config and plan output of one control step, or
   (b) running a short simulation and inspecting the trace? If a
   requirement fails both tests, rewrite or drop it.
4. Return the path to the written file and a short summary (≤ 10
   bullets) of the most consequential design decisions you encoded as
   requirements, plus any open questions you flagged.
