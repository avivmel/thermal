# ThermalGym: Complete Technical Reference

**Version**: 0.1.0  
**Last Updated**: 2026-04-05

---

## Overview

ThermalGym is an EnergyPlus-backed simulation environment for developing and evaluating residential thermostat demand response (DR) control policies. It is a thin Python wrapper around EnergyPlus that exposes a simple setpoint-control interface.

**What it is:**
- A physics-accurate building simulation for testing DR policies before real deployment
- A training data generator that produces episodes compatible with the existing `setpoint_responses.parquet` schema
- A policy evaluation framework with energy, comfort, and cost metrics

**What it is not:**
- A Gymnasium environment (no reward signal, no action/obs spaces, no RL assumptions)
- A real thermostat controller
- A multi-zone or non-residential simulator

**Why it exists:** CityLearn uses LSTM-learned thermal models and direct power-fraction control (`action ∈ [0,1]`). That approach failed in the DR experiment (3× energy overuse) because it bypassed thermostat logic. ThermalGym uses EnergyPlus physics with thermostat setpoints as actions — matching how real thermostats actually work.

---

## Package Structure

```
thermalgym/
├── __init__.py       # Public API — re-exports everything users need
├── env.py            # ThermalEnv: the core simulation environment
├── buildings.py      # Building registry and metadata dataclasses
├── policies.py       # Five built-in baseline DR policies
├── generate.py       # generate_episodes() and evaluate()
├── pyproject.toml    # Package config (setuptools)
├── SRS.md            # Software requirements specification
├── DESIGN.md         # Design document with architectural decisions
└── data/
    ├── buildings/    # EnergyPlus IDF files (ResStock archetypes)
    │   └── README.md # How to source IDF files
    └── weather/      # EPW weather files (3 climate zones)
        └── README.md # How to download EPW files
```

Five Python files total. No abstractions beyond what the task requires.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pyenergyplus` | EnergyPlus Python API — bundled with EnergyPlus 23.1+, NOT pip-installable |
| `numpy>=1.24` | Numerical operations |
| `pandas>=2.0` | Timeseries and episode DataFrames |
| `pyarrow>=12.0` | Parquet output |

**EnergyPlus is a prerequisite.** It must be installed separately (version 23.1+). `env.py` auto-detects it on macOS by scanning `/Applications/EnergyPlus-*`. No PyTorch, no scikit-learn, no Gymnasium.

---

## Module: `buildings.py`

### Purpose
Defines the `Building` dataclass and the registry of 9 pre-configured residential building archetypes. Maps climate zones to EPW weather files.

### `Building` Dataclass

```python
@dataclass(frozen=True)
class Building:
    id: str                  # e.g. "medium_cold_heatpump"
    idf_path: Path           # Path to EnergyPlus IDF file
    epw_path: Path           # Path to EPW weather file
    climate_zone: str        # "cold" | "hot" | "mixed"
    floor_area_sqft: int     # Approximate conditioned floor area
    vintage: str             # "pre1980" | "1980_2000" | "post2000"
    hvac_type: str           # "heatpump" | "ac_resistance" | "resistance_only"
    hvac_capacity_kw: float  # Rated HVAC capacity
    zone_name: str           # EnergyPlus thermal zone name (default: "conditioned space")
```

Frozen (immutable) — safe to use as dict keys or in sets.

### Building Registry

9 archetypes covering 3 climate zones × 3 sizes:

| ID | Climate | Size (sqft) | Vintage | HVAC Type | Capacity (kW) |
|----|---------|-------------|---------|-----------|----------------|
| `small_cold_heatpump` | Cold (Zone 6, Duluth MN) | 1000 | pre1980 | heatpump | 5.0 |
| `medium_cold_heatpump` | Cold | 2000 | 1980_2000 | heatpump | 8.0 |
| `large_cold_resistance` | Cold | 3000 | post2000 | ac_resistance | 12.0 |
| `small_hot_heatpump` | Hot (Zone 2, Houston TX) | 1000 | pre1980 | heatpump | 5.0 |
| `medium_hot_heatpump` | Hot | 2000 | 1980_2000 | heatpump | 8.0 |
| `large_hot_ac` | Hot | 3000 | post2000 | ac_resistance | 14.0 |
| `small_mixed_heatpump` | Mixed (Zone 4, Columbus OH) | 1000 | pre1980 | heatpump | 5.0 |
| `medium_mixed_heatpump` | Mixed | 2000 | 1980_2000 | heatpump | 8.0 |
| `large_mixed_ac` | Mixed | 3000 | post2000 | ac_resistance | 12.0 |

### EPW Weather Files

Climate zones map to specific TMY3 weather files:

| Climate | Location | File |
|---------|----------|------|
| `cold` | Duluth, MN | `USA_MN_Duluth.Intl.AP.727450_TMY3.epw` |
| `hot` | Houston, TX | `USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw` |
| `mixed` | Columbus, OH | `USA_OH_Columbus-Port.Columbus.Intl.AP.724280_TMY3.epw` |

### Public API

```python
from thermalgym.buildings import BUILDINGS, get_building, get_buildings

# All buildings as a dict
BUILDINGS  # {"small_cold_heatpump": Building(...), ...}

# Look up by exact ID — raises KeyError with helpful message if not found
b = get_building("medium_cold_heatpump")

# Filter by any Building field
cold_buildings = get_buildings(climate_zone="cold")
new_heatpumps  = get_buildings(vintage="post2000", hvac_type="heatpump")
```

`get_buildings()` raises `TypeError` for invalid field names, never raises for empty results.

---

## Module: `env.py`

### Purpose
The core simulation environment. Wraps EnergyPlus via its `pyenergyplus` Python API, exposing a simple `reset() / step()` interface.

### Constants

```python
HEAT_MIN, HEAT_MAX = 55.0, 75.0   # Heating setpoint valid range (°F)
COOL_MIN, COOL_MAX = 70.0, 85.0   # Cooling setpoint valid range (°F)
```

Default CA TOU price signal (24 hourly prices $/kWh):
- Off-peak (0–11h): $0.09
- Mid-peak (12–16h): $0.15
- On-peak (17–20h): $0.45
- Off-peak again (21–23h): $0.09

### `ThermalEnv` Class

#### Constructor

```python
ThermalEnv(
    building: str | Building,       # Required. Building ID or Building instance.
    timestep_minutes: int = 5,      # Simulation resolution: must be 5, 15, or 60
    run_period_days: int = 1,       # Episode length in days
    price_signal: np.ndarray = None,# 24-element hourly prices ($/kWh). None = CA TOU default.
    deadband_f: float = 1.0,        # Thermostat deadband in °F
    idf_path: Path = None,          # Override IDF path (for custom buildings)
    epw_path: Path = None,          # Override EPW path (for custom buildings)
)
```

**Raises:**
- `ValueError` if building string not in `BUILDINGS`, or `timestep_minutes` not in `{5, 15, 60}`, or `price_signal` not length 24
- `FileNotFoundError` if IDF or EPW file does not exist

On construction, only validates inputs and initializes state variables. EnergyPlus does not start until `reset()` is called.

#### `reset(date: str = "2017-07-01") -> dict`

Starts a new episode. Stops any running EnergyPlus process, resets all state, then launches a new EnergyPlus simulation via a daemon thread.

**Steps:**
1. Parses `date` (raises `ValueError` if not valid ISO 8601 `YYYY-MM-DD`)
2. Patches the IDF file with the correct deadband and run period
3. Spawns EnergyPlus in a daemon thread
4. Waits (up to 5 minutes) for the first post-warmup observation
5. Returns the first observation dict

**Returns:** Observation dict (see Observation Schema below).

#### `step(action: dict) -> dict`

Advances one timestep. The action is the setpoint to apply at the *current* timestep; the returned observation reflects the state at the *next* timestep.

**Process:**
1. Clamps `heat_setpoint` to `[55, 75]°F` and `cool_setpoint` to `[70, 85]°F`
2. Appends the current observation + action to `history`
3. Signals EnergyPlus to advance
4. Waits (up to 60 seconds) for the next timestep observation
5. Returns the new observation

After the final timestep (`step_count >= run_period_days × 24×60 / timestep_minutes`), sets `done = True`, waits briefly for EnergyPlus to finish, then stops the EnergyPlus thread.

**Raises:**
- `RuntimeError` if called before `reset()` or after episode ends
- Propagates any EnergyPlus exception stored in `_ep_exception`

#### `done` (property)

`True` after the final timestep of `run_period_days` has been returned.

#### `history` (property)

Returns a `pd.DataFrame` copy of all timesteps recorded so far. Available mid-episode. Returns an empty DataFrame with the correct schema before the first `step()`.

**Columns:** `timestamp`, `indoor_temp`, `outdoor_temp`, `hvac_power_kw`, `hvac_mode`, `heat_setpoint`, `cool_setpoint`, `electricity_price`, `hour`, `day_of_week`, `month`

### Observation Schema

Both `reset()` and `step()` return a dict with these keys:

| Key | Type | Units | Description |
|-----|------|-------|-------------|
| `timestamp` | `pd.Timestamp` | — | Absolute datetime of the observation |
| `indoor_temp` | `float` | °F | Zone mean air temperature |
| `outdoor_temp` | `float` | °F | Site outdoor air drybulb temperature |
| `hvac_power_kw` | `float` | kW | Total facility HVAC electricity demand |
| `hvac_mode` | `str` | — | `"heating"`, `"cooling"`, or `"off"` |
| `heat_setpoint` | `float` | °F | Currently active heating setpoint |
| `cool_setpoint` | `float` | °F | Currently active cooling setpoint |
| `electricity_price` | `float` | $/kWh | Price from `price_signal[hour]` |
| `hour` | `int` | 0–23 | Hour of day |
| `day_of_week` | `int` | 0–6 | 0 = Monday |
| `month` | `int` | 1–12 | Month of year |

All temperatures are returned in °F. EnergyPlus internally uses °C; conversion happens in `_read_obs()`.

### Action Schema

```python
{"heat_setpoint": float, "cool_setpoint": float}
```

Both values are clamped internally. Missing keys default to 68.0 / 76.0.

### EnergyPlus Integration (Private Methods)

#### How the threading works

EnergyPlus runs in a **daemon thread** (`_ep_thread`). The main Python thread and the EnergyPlus thread communicate via two `threading.Event` objects:

- `_ep_ready`: set by EnergyPlus callback when a new observation is ready; cleared by Python after reading
- `_py_ready`: set by Python when a new action is ready; cleared by EnergyPlus callback after reading
- `_stop_flag`: set by Python to tell EnergyPlus to stop

**Per-timestep sequence:**
```
EP callback fires at end of timestep
  → reads sensor values → stores in _current_obs
  → writes setpoints from _pending_action
  → sets _ep_ready  ←── Python: wait on _ep_ready
                         Python: reads _current_obs
                         Python: stores new action in _pending_action
                         Python: sets _py_ready
  → waits on _py_ready ←── (clears _py_ready)
  → returns from callback (EP advances to next timestep)
```

This lockstep ensures Python and EnergyPlus are always in sync without polling.

#### IDF Patching

Before each episode, two IDF modifications are made:

1. **`_patch_idf_deadband(deadband_f)`**: Modifies `ThermostatSetpoint:DualSetpoint` "Temperature Difference Between Cutout And Setpoint" field. Converts from °F to °C (`deadband_c = deadband_f × 5/9`). Result is cached in `_IDF_PATCH_CACHE` keyed by `(idf_path, deadband_f)` — patches are written once per process.

2. **`_patch_idf_run_period(idf_path, start_date)`**: Replaces the `RunPeriod` block to set begin/end month, day, and year. Also patches the `Timestep` line to match `timestep_minutes` (converts to `timesteps_per_hour = 60 // timestep_minutes`). Writes a new temp file for every episode.

Both patching methods use `tempfile.NamedTemporaryFile` and write to the OS temp directory.

#### EnergyPlus Variables and Actuators

Variables requested before the run (required for `get_variable_handle` to work):

| Variable | Object Type | Used For |
|----------|-------------|----------|
| `Zone Mean Air Temperature` | zone name | `indoor_temp` |
| `Site Outdoor Air Drybulb Temperature` | `"Environment"` | `outdoor_temp` |
| `Facility Total HVAC Electricity Demand Rate` | `"Whole Building"` | `hvac_power_kw` |
| `Facility Total Heating Electricity Rate` | `"Whole Building"` | detecting `hvac_mode = "heating"` |
| `Facility Total Cooling Electricity Rate` | `"Whole Building"` | detecting `hvac_mode = "cooling"` |

Actuators used to write setpoints:

| Actuator | Component Type | Control Type |
|----------|---------------|--------------|
| Heating setpoint | `"Zone Temperature Control"` | `"Heating Setpoint"` |
| Cooling setpoint | `"Zone Temperature Control"` | `"Cooling Setpoint"` |

Handles are initialized once after warmup completes (stored in `_handles_initialized`).

#### Warmup

EnergyPlus runs a warmup period before the simulation date to stabilize thermal mass. The callback checks `api.exchange.warmup_flag(state)` and returns early during warmup — Python sees no observations until warmup is complete. The warmup duration is controlled by EnergyPlus internally (the IDF's `Building` object specifies warmup days; typical default is 7 days).

---

## Module: `policies.py`

### Purpose
Five standalone callable classes implementing DR strategies. All follow the same interface: `policy(obs: dict) -> dict`.

No base class. Any callable `(obs) -> {"heat_setpoint": float, "cool_setpoint": float}` is a valid policy.

### `Baseline`

Holds setpoints constant. The no-DR reference.

```python
Baseline(heat_setpoint=68.0, cool_setpoint=76.0)
```

Returns the same setpoints regardless of time, price, or temperature.

### `PreCool`

Pre-cools before peak, then relaxes during peak. Designed for summer cooling DR.

```python
PreCool(
    precool_offset=2.0,    # °F to lower cool_setpoint during pre-cool
    precool_hours=2,       # How many hours before peak to start pre-cooling
    peak_start=17,         # Peak period start hour (inclusive)
    peak_end=20,           # Peak period end hour (exclusive)
    setback=2.0,           # °F to raise cool_setpoint during peak
    base_heat=68.0,
    base_cool=76.0,
)
```

**Logic:**
- `[peak_start - precool_hours, peak_start)`: `cool_sp = base_cool - precool_offset`
- `[peak_start, peak_end)`: `cool_sp = base_cool + setback`
- Otherwise: `cool_sp = base_cool`

Heating setpoint is always `base_heat`.

### `PreHeat`

Pre-heats before peak, then setbacks during peak. Designed for winter heating DR.

```python
PreHeat(
    preheat_offset=2.0,    # °F to raise heat_setpoint during pre-heat
    preheat_hours=2,
    peak_start=17,
    peak_end=20,
    setback=2.0,           # °F to lower heat_setpoint during peak
    base_heat=68.0,
    base_cool=76.0,
)
```

Mirror of `PreCool` for the heating setpoint. Cooling setpoint is always `base_cool`.

### `Setback`

Raises cooling (or lowers heating) setpoint during peak. No pre-conditioning phase.

```python
Setback(
    magnitude=4.0,         # °F shift applied during peak
    peak_start=17,
    peak_end=20,
    mode="cooling",        # "cooling" | "heating" | "both"
    base_heat=68.0,
    base_cool=76.0,
)
```

**Logic during peak:**
- `mode="cooling"`: `cool_sp = base_cool + magnitude`
- `mode="heating"`: `heat_sp = base_heat - magnitude`
- `mode="both"`: both adjustments applied simultaneously

Outside peak: always baseline setpoints.

### `PriceResponse`

Adjusts cooling setpoint proportional to electricity price.

```python
PriceResponse(
    threshold_low=0.10,    # $/kWh — at or below this, apply adjust_low
    threshold_high=0.25,   # $/kWh — at or above this, apply adjust_high
    adjust_low=-1.0,       # °F offset at low price (pre-cool slightly)
    adjust_high=2.0,       # °F offset at high price (relax cooling)
    base_heat=68.0,
    base_cool=76.0,
)
```

**Logic:**
- `price >= threshold_high`: `cool_sp = base_cool + adjust_high`
- `price <= threshold_low`: `cool_sp = base_cool + adjust_low`
- Between thresholds: linear interpolation of the adjustment

Heating setpoint is never adjusted.

---

## Module: `generate.py`

### Purpose
Two top-level functions for batch use: `generate_episodes()` creates training data; `evaluate()` benchmarks policies.

### `generate_episodes()`

Generates setpoint-response training episodes compatible with `setpoint_responses.parquet`.

```python
generate_episodes(
    output: str | Path,                            # Output Parquet file path
    n_episodes: int = 1000,                        # Total episodes across all buildings
    buildings: str | list[str] = "all",            # Building IDs or "all"
    modes: list[str] = ["heat_increase", "cool_decrease"],
    gap_range: tuple[float, float] = (1.0, 5.0),  # Setpoint-to-temp gap range (°F)
    outdoor_temp_range: tuple | None = None,       # Filter by outdoor temp (°F)
    timestep_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame
```

**How it works:**
1. Distributes `n_episodes` proportionally across buildings (base + remainder assignment)
2. Spawns one worker process per building (via `multiprocessing.Pool` with `spawn` context)
3. Each worker creates its own `ThermalEnv`, runs episodes sequentially for that building
4. For each episode: picks a random date (Jan–Nov 2017), picks a random mode, resets env, applies a random setpoint gap, steps until target reached or 4-hour timeout
5. Records `time_to_target_min` on the first row of each episode only
6. Combines all worker DataFrames, remaps `episode_id` to global monotonic integers
7. Writes to Parquet and returns the combined DataFrame

**Episode generation logic (per worker):**
- Random date in 2017 (day_of_year 1–334 to avoid year-end boundary issues)
- `heat_increase`: `new_setpoint = indoor_temp + gap` (clamped to `[55, 75]°F`)
- `cool_decrease`: `new_setpoint = indoor_temp - gap` (clamped to `[70, 85]°F`)
- Episode succeeds if target reached within 4 hours; otherwise discarded
- Up to `n_episodes × 10` attempts per building before giving up

**Output schema** (matches `setpoint_responses.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| `home_id` | str | Building ID |
| `episode_id` | int | Global unique episode ID |
| `mode` | str | `"heat_increase"` or `"cool_decrease"` |
| `timestep` | int | Timestep within episode (0, 1, 2, …) |
| `timestamp` | datetime | Absolute timestamp |
| `indoor_temp` | float | Indoor temperature (°F) |
| `outdoor_temp` | float | Outdoor temperature (°F) |
| `setpoint` | float | Active setpoint (heating or cooling, °F) |
| `hvac_power_kw` | float | HVAC electric power (kW) |
| `hvac_on` | bool | HVAC running (True) or off |
| `hour` | int | Hour of day (0–23) |
| `month` | int | Month (1–12) |
| `time_to_target_min` | float | Minutes to reach setpoint (first row only; NaN for all other rows) |

### `evaluate()`

Runs one or more policies across buildings and DR scenarios, returning a tidy metrics DataFrame.

```python
evaluate(
    policy: Callable | dict[str, Callable],  # Single policy or {name: callable}
    buildings: str | list[str] = "all",
    scenarios: list[str] = ["baseline", "precool", "setback"],
    n_days: int = 7,
    start_date: str = "2017-07-01",
    price_signal: np.ndarray | None = None,
    timestep_minutes: int = 15,
) -> pd.DataFrame
```

**How it works:**
1. **First pass** (baseline): For each (building, scenario) combination, runs the scenario's built-in policy (e.g., `Setback()` for the "setback" scenario) and stores `peak_kwh` and `cost_usd` as baseline reference values.
2. **Second pass**: For each named policy, runs it against every (building, scenario) and computes metrics relative to the baseline from pass 1.
3. `"baseline"` is always added as a policy (using `Baseline()`) if not already in the dict.

Scenario → built-in policy mapping:
- `"baseline"` → `Baseline()`
- `"precool"` → `PreCool()`
- `"setback"` → `Setback()`
- `"price_response"` → `PriceResponse()`

If IDF/EPW files are missing or EnergyPlus is not installed, that (building, scenario) row is filled with `NaN` values instead of raising.

**Output columns:**

| Column | Description |
|--------|-------------|
| `policy` | Policy name |
| `building` | Building ID |
| `scenario` | Scenario name |
| `total_kwh` | Total HVAC energy consumed |
| `peak_kw` | Maximum instantaneous HVAC power |
| `par` | Peak-to-average power ratio |
| `peak_kwh` | HVAC energy during peak hours (17–20h) |
| `shifted_kwh` | Energy moved off-peak vs. baseline scenario (0 for baseline rows) |
| `comfort_violations_h` | Hours with indoor temp outside `[heat_sp - 1, cool_sp + 1]°F` |
| `discomfort_degree_hours` | Sum of (deviation × dt) for all out-of-bounds timesteps |
| `max_temp_deviation_f` | Largest single-timestep deviation from comfort bounds |
| `cost_usd` | Total electricity cost |
| `cost_savings_usd` | Cost reduction vs. baseline policy (0 for baseline rows) |

### `_compute_metrics()` (private)

Called internally after each episode. Infers timestep duration from timestamp spacing.

**Comfort bounds:** `[heat_setpoint - 1°F, cool_setpoint + 1°F]` — 1°F slack on each side.

**Peak period:** hours 17–19 (inclusive start, exclusive end = `[17, 20)`).

---

## Module: `__init__.py`

The public API surface. Everything a user needs is importable directly from `thermalgym`.

```python
import thermalgym

# Core
thermalgym.ThermalEnv
thermalgym.HEAT_MIN, thermalgym.HEAT_MAX   # 55.0, 75.0
thermalgym.COOL_MIN, thermalgym.COOL_MAX   # 70.0, 85.0

# Buildings
thermalgym.Building
thermalgym.BUILDINGS         # dict of all 9 archetypes
thermalgym.get_building()    # by ID
thermalgym.get_buildings()   # filtered by attribute

# Data generation
thermalgym.generate_episodes()
thermalgym.evaluate()

# Policies
thermalgym.Baseline
thermalgym.PreCool
thermalgym.PreHeat
thermalgym.Setback
thermalgym.PriceResponse
```

---

## Data Files

### IDF Files (`data/buildings/`)

Nine EnergyPlus Input Data Files, one per building archetype. Derived from NREL ResStock archetypes. **Not included in the repository** — must be sourced from ResStock or EnergyPlus example files.

Each IDF must:
- Define exactly one thermal zone named `"LIVING ZONE"` (or match `Building.zone_name`)
- Include `ZoneControl:Thermostat` referencing `ThermostatSetpoint:DualSetpoint` objects (for deadband patching to work)
- Include an electric HVAC system matching `Building.hvac_type`
- Use the `RunPeriod` object (dates are overridden at runtime by `_patch_idf_run_period`)

At runtime, two patched copies of each IDF are written to `tempfile` for each episode:
1. Deadband-patched copy (cached per `(idf_path, deadband_f)`)
2. Run-period + timestep patched copy (new temp file per episode)

### EPW Files (`data/weather/`)

Three EnergyPlus Weather files in TMY3 format. **Not included** — download from the EnergyPlus weather database.

| File | Location | Climate |
|------|----------|---------|
| `USA_MN_Duluth.Intl.AP.727450_TMY3.epw` | Duluth, MN | Cold (Zone 6) |
| `USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw` | Houston, TX | Hot (Zone 2) |
| `USA_OH_Columbus-Port.Columbus.Intl.AP.724280_TMY3.epw` | Columbus, OH | Mixed (Zone 4) |

---

## Usage Examples

### Basic simulation loop

```python
import thermalgym

env = thermalgym.ThermalEnv(building="medium_cold_heatpump")
obs = env.reset(date="2017-07-15")

while not env.done:
    action = {"heat_setpoint": 68.0, "cool_setpoint": 76.0}
    obs = env.step(action)

df = env.history
# df has columns: timestamp, indoor_temp, outdoor_temp, hvac_power_kw,
#                 hvac_mode, heat_setpoint, cool_setpoint, electricity_price,
#                 hour, day_of_week, month
```

### Custom policy

```python
def my_policy(obs):
    # Pre-cool before peak
    if 15 <= obs["hour"] < 17:
        return {"heat_setpoint": 68.0, "cool_setpoint": 74.0}
    elif 17 <= obs["hour"] < 20:
        return {"heat_setpoint": 68.0, "cool_setpoint": 78.0}
    return {"heat_setpoint": 68.0, "cool_setpoint": 76.0}
```

### Multi-day run

```python
env = thermalgym.ThermalEnv(
    building="large_hot_ac",
    timestep_minutes=15,
    run_period_days=30,
)
obs = env.reset(date="2017-06-01")
policy = thermalgym.PreCool(precool_offset=3.0)

while not env.done:
    obs = env.step(policy(obs))

print(env.history[["timestamp", "indoor_temp", "hvac_power_kw"]].describe())
```

### Generate training data

```python
df = thermalgym.generate_episodes(
    output="sim_episodes.parquet",
    n_episodes=5000,
    buildings="all",
    modes=["heat_increase", "cool_decrease"],
    gap_range=(1.0, 5.0),
    seed=42,
)
print(f"Generated {df['episode_id'].nunique()} episodes")
```

### Evaluate policies

```python
results = thermalgym.evaluate(
    policy={
        "precool": thermalgym.PreCool(),
        "setback": thermalgym.Setback(magnitude=3.0),
        "price_response": thermalgym.PriceResponse(),
    },
    buildings=["medium_cold_heatpump", "medium_hot_heatpump"],
    scenarios=["baseline", "precool", "setback"],
    n_days=14,
    start_date="2017-07-01",
)
print(results[["policy", "building", "par", "peak_kwh", "comfort_violations_h", "cost_usd"]])
```

### Custom building (advanced)

```python
from pathlib import Path
import thermalgym

env = thermalgym.ThermalEnv(
    building="medium_cold_heatpump",   # Used for metadata only
    idf_path=Path("my_custom_building.idf"),
    epw_path=Path("my_custom_weather.epw"),
)
```

### Filter buildings by attribute

```python
# All cold-climate buildings
cold = thermalgym.get_buildings(climate_zone="cold")

# Only large post-2000 buildings
large_new = thermalgym.get_buildings(vintage="post2000")
# -> [large_cold_resistance, large_hot_ac, large_mixed_ac]

# Run evaluation on only hot-climate buildings
results = thermalgym.evaluate(
    policy=thermalgym.PreCool(),
    buildings=[b.id for b in thermalgym.get_buildings(climate_zone="hot")],
)
```

---

## Key Design Decisions

### Setpoints as actions (not power fractions)

The CityLearn DR experiment showed that `action ∈ [0, 1]` (power fraction) causes massive overheating because it bypasses thermostat logic — `action=0.7` runs the HVAC at 70% max power (~20 kW) regardless of actual need (~5 kW). ThermalGym uses setpoint temperatures. The HVAC turns on/off based on whether the indoor temp is above or below the setpoint. This matches real thermostat APIs exactly.

### `pyenergyplus` over subprocess

EnergyPlus 23.1+ ships a Python API that lets Python and EnergyPlus share a process via callbacks. This gives per-timestep interactive control without file I/O between steps. The alternative (subprocess + CSV) requires writing input files and reading output files each step — slower, more fragile.

### No Gymnasium wrapper

No reward, no `gym.Env`, no action/obs spaces, no RL assumptions. A policy is just a callable. This makes it trivially easy to wrap any existing controller, heuristic, or MPC optimizer without inheriting from anything.

### `hvac_mode` over `hvac_on`

The observation exposes `hvac_mode: str` (`"heating"` / `"cooling"` / `"off"`) rather than a boolean. A boolean is ambiguous — during cooling, `hvac_on=True` doesn't tell you which direction. Policies that adjust setpoints based on HVAC state need to know whether it's heating or cooling.

### Single-zone buildings only

Real thermostats control one zone. Multi-zone adds complexity without benefit for the DR use case. All 9 bundled buildings use single-zone representations.

### Fixed occupancy schedules

ResStock IDFs already include occupancy schedules with internal heat gains. Dynamic occupancy would add complexity without clear benefit for DR policy testing — the primary thermal drivers are weather and HVAC.

### Humidity excluded

The Ecobee `setpoint_responses.parquet` doesn't include humidity. Latent loads complicate HVAC modeling significantly. May be added as an optional observation in a future release.

---

## Constraints and Limitations

| Constraint | Detail |
|------------|--------|
| EnergyPlus required | Version 23.1+. `pyenergyplus` is bundled with EnergyPlus, not pip-installable. |
| IDF/EPW files not included | Must source from ResStock or download separately. |
| Simulation timestep | Must be 5, 15, or 60 minutes exactly. |
| Setpoint bounds | Heating: 55–75°F. Cooling: 70–85°F. Hard-clamped, no exceptions. |
| Electric HVAC only | Gas, oil, and other fuel types not supported. |
| Single-zone | No multi-zone building models. |
| Single-stage equipment | Variable-speed/staged HVAC not modeled. |
| No humidity | Latent loads excluded. Less accurate for hot-humid climates (Zone 2A). |
| Python 3.9+ | Required. |
| Not faster-than-realtime guaranteed | SRS specifies ≥1 day in ≤60 seconds on a standard workstation. |

---

## Error Handling Reference

| Situation | Exception | Where |
|-----------|-----------|-------|
| Unknown building ID | `KeyError` | `get_building()`, `ThermalEnv.__init__()` |
| Invalid timestep_minutes | `ValueError` | `ThermalEnv.__init__()` |
| Invalid price_signal shape | `ValueError` | `ThermalEnv.__init__()` |
| IDF file not found | `FileNotFoundError` | `ThermalEnv.__init__()` |
| EPW file not found | `FileNotFoundError` | `ThermalEnv.__init__()` |
| Invalid date string | `ValueError` | `ThermalEnv.reset()` |
| EnergyPlus startup timeout (5 min) | `RuntimeError` | `ThermalEnv.reset()` |
| step() before reset() or after done | `RuntimeError` | `ThermalEnv.step()` |
| EnergyPlus timestep timeout (60 sec) | `RuntimeError` | `ThermalEnv.step()` |
| EnergyPlus internal exception | Re-raised from `_ep_exception` | `step()` or `reset()` |
| pyenergyplus not found | `ImportError` | `_start_energyplus()` |
| Invalid filter key in get_buildings | `TypeError` | `get_buildings()` |
| Invalid Setback mode | `ValueError` | `Setback.__init__()` |
| generate_episodes with no buildings | `ValueError` | `generate_episodes()` |
| Missing IDF/EPW in generate/evaluate | Returns empty DataFrame / NaN row | worker functions |
