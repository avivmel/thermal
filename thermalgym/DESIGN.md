# ThermalGym Design Document

**Version**: 1.1  
**Date**: 2026-04-05

---

## 1. Design Philosophy

**Simplicity first.** ThermalGym is a thin wrapper around EnergyPlus that speaks Gymnasium. The entire library is five Python files. Users who know Gymnasium need to learn almost nothing new.

**Policies are just functions.** No base classes to inherit, no interfaces to implement. A policy is any callable that takes an observation and returns an action.

**Two primary entry points.** Most users need one of two things: (1) an interactive env for developing control policies, or (2) batch data generation for training ML models. Both are one-liners.

---

## 2. Package Structure

```
thermalgym/
├── __init__.py       # Public API — re-exports everything users need
├── env.py            # ThermalEnv: the Gymnasium environment
├── buildings.py      # Building registry and metadata
├── policies.py       # Four built-in baseline policies
├── generate.py       # generate_episodes() and evaluate()
└── data/
    ├── buildings/    # EnergyPlus IDF files (ResStock archetypes)
    └── weather/      # EPW weather files by climate zone
```

No subdirectories beyond `data/`. No abstract base classes. No plugin system.

---

## 3. API Overview

### 3.1 Interactive Environment

```python
import thermalgym

env = thermalgym.ThermalEnv(building="medium_cold_heatpump")

obs = env.reset(date="2017-07-15")

while not env.done:
    action = {"heat_setpoint": 70, "cool_setpoint": 76}
    obs = env.step(action)

# After the run: full timeseries as a DataFrame
ts = env.history
# Columns: timestamp, indoor_temp, outdoor_temp, hvac_power_kw, hvac_mode,
#          heat_setpoint, cool_setpoint, electricity_price, hour, day_of_week, month
```

### 3.2 Training Data Generation

```python
# Generate episodes matching setpoint_responses.parquet schema
thermalgym.generate_episodes(
    output="sim_episodes.parquet",
    n_episodes=5000,          # total across all buildings
    buildings="all",          # or list of building IDs
)
```

### 3.3 Policy Evaluation

```python
results = thermalgym.evaluate(
    policy=my_policy,         # callable: obs -> action
    buildings="all",
    scenarios=["precool", "setback"],
    n_days=30,
)
# returns DataFrame with PAR, peak_kw, total_kwh, comfort_violations per building/scenario
```

### 3.4 Built-in Policies

```python
from thermalgym.policies import Baseline, PreCool, Setback, PriceResponse

# All policies follow the same interface: policy(obs) -> action
policy = PreCool(precool_offset=2.0, precool_hours=2, peak_start=17, peak_end=20)
action = policy(obs)
```

---

## 4. Core Components

### 4.1 `ThermalEnv` (`env.py`)

A simulation environment that drives EnergyPlus via the `pyenergyplus` Python API using timestep callbacks. Not a Gymnasium env — no reward, no RL assumptions.

**Observation** (plain dict, returned by `reset()` and `step()`):

| Key | Units | Description |
|-----|-------|-------------|
| `indoor_temp` | °F | Current indoor air temperature |
| `outdoor_temp` | °F | Current outdoor temperature |
| `hvac_power_kw` | kW | HVAC electric power draw |
| `hvac_mode` | str | HVAC operating mode: `"heating"`, `"cooling"`, or `"off"` |
| `heat_setpoint` | °F | Active heating setpoint |
| `cool_setpoint` | °F | Active cooling setpoint |
| `electricity_price` | $/kWh | Current hourly electricity price from `price_signal` |
| `hour` | 0–23 | Hour of day |
| `day_of_week` | 0–6 | Day of week (0 = Monday) |
| `month` | 1–12 | Month of year |

**Action** (plain dict, passed to `step()`):

```python
{"heat_setpoint": float, "cool_setpoint": float}
```

Setpoints are clamped to valid ranges (55–75°F heat, 70–85°F cool) internally.

**Key constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `building` | required | Building ID string (see §4.2) |
| `timestep_minutes` | `5` | Simulation resolution (5, 15, or 60) |
| `run_period_days` | `1` | How many days to simulate per episode |
| `price_signal` | `None` | Array of 24 hourly prices ($/kWh); uses CA TOU default if None |

**EnergyPlus integration:**  
`ThermalEnv` spawns EnergyPlus as a subprocess using the `pyenergyplus` runtime API. On each `step()`, the env writes setpoints via `setpoint_actuators`, waits for the next timestep callback, then reads back sensor values. The callback mechanism lets EnergyPlus and Python run in lockstep without polling.

**Deadband:**  
Thermostat deadband (default ±1°F) is configured via `ZoneControl:Thermostat` objects in each bundled IDF. `ThermalEnv` accepts a `deadband_f: float = 1.0` constructor parameter that patches this value at load time. Users adding custom IDFs are responsible for their own deadband configuration.

**Warm-up:**  
`reset()` silently runs a 7-day warm-up before the requested start date. This is transparent — the user's episode starts at the date they specify. Warm-up state is reused if `reset()` is called with the same `date` and `building`, avoiding redundant simulation. Note: this cache is per-process and does not carry across `multiprocessing` workers; see §4.4 for how `generate_episodes()` handles this.

**`env.history`:**  
After an episode, `env.history` returns a DataFrame with one row per timestep. Columns match the observation keys plus `timestamp`. Available mid-episode too (partial timeseries).

**Episode termination:**  
`step()` returns the observation for the current timestep. After the final timestep of the run period, `env.done` is `True` and the returned obs is the terminal state. Calling `step()` again raises `RuntimeError("Episode has ended; call reset().")`.

### 4.2 Building Registry (`buildings.py`)

Buildings are plain dataclasses registered in a dict. Nothing dynamic.

```python
@dataclass
class Building:
    id: str
    idf_path: Path
    epw_path: Path
    climate_zone: str        # "cold", "hot", "mixed"
    floor_area_sqft: int
    vintage: str             # "pre1980", "1980_2000", "post2000"
    hvac_type: str           # "heatpump", "ac_resistance", "resistance_only"
    hvac_capacity_kw: float
```

**Accessing buildings:**

```python
from thermalgym.buildings import BUILDINGS, get_building

# Dict of all available buildings
BUILDINGS  # {"medium_cold_heatpump": Building(...), ...}

# Filter helpers
get_building("medium_cold_heatpump")          # by ID
get_buildings(climate_zone="cold")            # by attribute
get_buildings(vintage="post2000", hvac_type="heatpump")
```

**Initial building library** (9 archetypes — 3 climate zones × 3 sizes):

| ID | Climate | Size | HVAC |
|----|---------|------|------|
| `small_cold_heatpump` | Cold (Zone 6) | ~1000 sqft | Heat pump |
| `medium_cold_heatpump` | Cold (Zone 6) | ~2000 sqft | Heat pump |
| `large_cold_resistance` | Cold (Zone 6) | ~3000 sqft | AC + resistance |
| `small_hot_heatpump` | Hot (Zone 2) | ~1000 sqft | Heat pump |
| `medium_hot_heatpump` | Hot (Zone 2) | ~2000 sqft | Heat pump |
| `large_hot_ac` | Hot (Zone 2) | ~3000 sqft | Central AC |
| `small_mixed_heatpump` | Mixed (Zone 4) | ~1000 sqft | Heat pump |
| `medium_mixed_heatpump` | Mixed (Zone 4) | ~2000 sqft | Heat pump |
| `large_mixed_ac` | Mixed (Zone 4) | ~3000 sqft | Central AC |

Users can add custom buildings by passing `idf_path` and `epw_path` directly to `ThermalEnv`.

### 4.3 Policies (`policies.py`)

Four standalone classes. Each is callable: `policy(obs) -> action`.

**`Baseline`**: Holds setpoints constant. No DR.

**`PreCool`** / **`PreHeat`**: Shifts setpoint by `offset` degrees during `[peak_start - lead_hours, peak_start]`, relaxes during peak. Configurable offset, lead time, peak window.

**`Setback`**: Raises cooling (or lowers heating) setpoint by `magnitude` during peak hours.

**`PriceResponse`**: Adjusts setpoint based on `obs["electricity_price"]`. Linear mapping: price above threshold → tighten setpoint; below → loosen.

All four accept a `price_signal` array for price-aware decisions when needed.

### 4.4 Data Generation and Evaluation (`generate.py`)

Two top-level functions. Both use `ThermalEnv` internally.

**`generate_episodes()`**: Runs many short episodes across buildings and initial conditions, collecting the `setpoint_responses.parquet`-compatible schema (see SRS Appendix A). Parallelizes across buildings using `multiprocessing`.

Episodes are distributed to workers **batched by building**: each worker handles all episodes for one building sequentially, so the warm-up cache hits after the first episode per worker. Grouping by building before dispatch is the caller's responsibility in `generate_episodes()`.

```python
def generate_episodes(
    output: str | Path,
    n_episodes: int = 1000,
    buildings: str | list[str] = "all",
    modes: list[str] = ["heat_increase", "cool_decrease"],
    gap_range: tuple = (1.0, 5.0),          # °F
    outdoor_temp_range: tuple | None = None, # °F; None = use weather file as-is
    timestep_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame: ...
```

**`evaluate()`**: Runs one or more policies over a set of buildings and DR scenarios for `n_days`. When multiple policies are passed, all run against identical scenario instances (same seed, same building state) so results are directly comparable. Returns a tidy DataFrame of metrics.

```python
def evaluate(
    policy: Callable | dict[str, Callable],  # dict keys become the "policy" column
    buildings: str | list[str] = "all",
    scenarios: list[str] = ["baseline", "precool", "setback"],
    n_days: int = 7,
    start_date: str = "2017-07-01",
    price_signal: np.ndarray | None = None,
    timestep_minutes: int = 15,
) -> pd.DataFrame: ...
```

Output columns: `policy`, `building`, `scenario`, `total_kwh`, `peak_kw`, `par`, `peak_kwh`, `shifted_kwh`, `comfort_violations_h`, `discomfort_degree_hours`, `max_temp_deviation_f`, `cost_usd`, `cost_savings_usd`.

- `shifted_kwh`: energy moved from peak to off-peak vs. the `"baseline"` scenario (0 for baseline rows).
- `discomfort_degree_hours`: sum of (deviation × duration) for all timesteps outside comfort bounds.
- `max_temp_deviation_f`: largest single-timestep deviation from the active setpoint.
- `cost_savings_usd`: cost reduction vs. baseline policy over the same scenario (0 for baseline rows).

`"baseline"` is always run automatically when a dict of policies is passed, so relative metrics can be computed. If `policy` is a plain `Callable`, it is treated as `{"policy": policy}` and a `Baseline()` run is added implicitly.

---

## 5. Key Design Decisions

### 5.1 pyenergyplus over subprocess

EnergyPlus 23.1+ ships a Python API (`pyenergyplus`) that lets Python and EnergyPlus share a process via callbacks. This gives per-timestep interactive control without file I/O between steps. The alternative (subprocess + CSV) requires writing input files and reading output files each step, which is slower and more fragile.

### 5.2 No Gym Wrappers for Setpoints

CityLearn uses `action = power_fraction ∈ [0, 1]`. The CityLearn DR experiment showed this fails because it bypasses thermostat logic. ThermalGym actions are setpoint temperatures — the thermostat decides when to run the HVAC. This matches real thermostat APIs and avoids the open-loop failure mode.

### 5.3 No Abstract Base Class for Policy

Policies don't inherit from anything. Any callable `(obs) -> action` works. This makes it trivial to wrap any existing controller or lambda.

### 5.4 `hvac_mode` Over `hvac_on`

The observation exposes `hvac_mode: str` (`"heating"` / `"cooling"` / `"off"`) rather than a boolean. A boolean is ambiguous during cooling — policies that adjust cooling setpoints need to know whether the unit is currently heating or cooling. `hvac_mode` is read directly from the EnergyPlus `SystemNode` sensor on the supply air loop.

### 5.5 Buildings Bundled, Not Downloaded

IDF and EPW files ship with the package. No internet required, no setup scripts. Users get a working env immediately after `pip install thermalgym`. Custom IDF/EPW paths are supported for advanced users.

---

## 6. Dependencies

| Package | Purpose |
|---------|---------|
| `pyenergyplus` | EnergyPlus Python API (bundled with EnergyPlus) |
| `numpy` | Numerical operations |
| `pandas` | Timeseries and episode DataFrames |
| `pyarrow` | Parquet output |

No PyTorch, no scikit-learn, no Gymnasium. The gym is physics-only; ML lives upstream.

---

## 7. What's Explicitly Out of Scope

- Multi-zone buildings
- Variable-speed / staged HVAC equipment  
- Humidity / latent loads
- Graphical interface
- Pre-trained ML policies
- Non-residential buildings
- Cloud / distributed simulation

These can be added later without breaking the core API.
