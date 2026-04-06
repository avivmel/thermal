# ThermalGym Implementation Plan

**Version**: 1.0  
**Date**: 2026-04-05  
**Based on**: `thermalgym/DESIGN.md` v1.1, `thermalgym/SRS.md` v1.1

---

## Overview

Five Python modules + package scaffolding. Four work streams are fully independent and can be parallelized immediately. `generate.py` and `__init__.py` must wait for their upstream modules.

```
Stream A: buildings.py          ─┐
Stream B: policies.py           ─┤─ parallel, no internal deps
Stream C: pyproject.toml + data ─┘
                                  ↓
Stream D: env.py                ──── depends on buildings.py (Building type)
                                  ↓
Stream E: generate.py           ──── depends on env.py + policies.py
                                  ↓
Stream F: __init__.py           ──── depends on all streams
```

---

## Shared Type Contracts

All agents must agree on these — they are the interface boundaries between modules.

### Observation dict (`Obs`)

```python
{
    "indoor_temp":       float,  # °F, current zone mean air temperature
    "outdoor_temp":      float,  # °F, site outdoor drybulb
    "hvac_power_kw":     float,  # kW, facility HVAC electricity demand
    "hvac_mode":         str,    # "heating" | "cooling" | "off"
    "heat_setpoint":     float,  # °F, active heating setpoint
    "cool_setpoint":     float,  # °F, active cooling setpoint
    "electricity_price": float,  # $/kWh, from price_signal[hour]
    "hour":              int,    # 0–23
    "day_of_week":       int,    # 0–6 (0 = Monday)
    "month":             int,    # 1–12
}
```

### Action dict (`Action`)

```python
{
    "heat_setpoint": float,  # °F, will be clamped to [55, 75] in env
    "cool_setpoint": float,  # °F, will be clamped to [70, 85] in env
}
```

### Policy callable

```python
Policy = Callable[[dict], dict]  # obs -> action
```

### Setpoint clamp constants (used in `env.py`, referenced by `policies.py` docstrings)

```python
HEAT_MIN, HEAT_MAX = 55.0, 75.0
COOL_MIN, COOL_MAX = 70.0, 85.0
```

---

## Stream A: `thermalgym/buildings.py`

**Dependencies**: none  
**Can start**: immediately

### File: `thermalgym/buildings.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import importlib.resources

_DATA_DIR = Path(__file__).parent / "data"

@dataclass(frozen=True)
class Building:
    id: str
    idf_path: Path
    epw_path: Path
    climate_zone: str       # "cold" | "hot" | "mixed"
    floor_area_sqft: int
    vintage: str            # "pre1980" | "1980_2000" | "post2000"
    hvac_type: str          # "heatpump" | "ac_resistance" | "resistance_only"
    hvac_capacity_kw: float
```

**`BUILDINGS` registry** — all 9 archetypes hardcoded:

| ID | climate_zone | floor_area_sqft | vintage | hvac_type | hvac_capacity_kw |
|----|-------------|-----------------|---------|-----------|-----------------|
| `small_cold_heatpump` | cold | 1000 | pre1980 | heatpump | 5.0 |
| `medium_cold_heatpump` | cold | 2000 | 1980_2000 | heatpump | 8.0 |
| `large_cold_resistance` | cold | 3000 | post2000 | ac_resistance | 12.0 |
| `small_hot_heatpump` | hot | 1000 | pre1980 | heatpump | 5.0 |
| `medium_hot_heatpump` | hot | 2000 | 1980_2000 | heatpump | 8.0 |
| `large_hot_ac` | hot | 3000 | post2000 | ac_resistance | 14.0 |
| `small_mixed_heatpump` | mixed | 1000 | pre1980 | heatpump | 5.0 |
| `medium_mixed_heatpump` | mixed | 2000 | 1980_2000 | heatpump | 8.0 |
| `large_mixed_ac` | mixed | 3000 | post2000 | ac_resistance | 12.0 |

idf_path: `_DATA_DIR / "buildings" / f"{id}.idf"`  
epw_path: uses climate zone → file mapping:
- cold → `_DATA_DIR / "weather" / "USA_MN_Duluth.726550_TMY3.epw"`
- hot → `_DATA_DIR / "weather" / "USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw"`
- mixed → `_DATA_DIR / "weather" / "USA_OH_Columbus.Intl.AP.724280_TMY3.epw"`

**Public functions:**

```python
BUILDINGS: dict[str, Building]  # populated at module import

def get_building(id: str) -> Building:
    """Return building by ID. Raises KeyError with helpful message if not found."""
    ...

def get_buildings(**filters) -> list[Building]:
    """
    Filter BUILDINGS by attribute equality.
    
    Examples:
        get_buildings(climate_zone="cold")
        get_buildings(vintage="post2000", hvac_type="heatpump")
    
    Returns list (possibly empty). Never raises.
    Raises TypeError if a filter key is not a Building field.
    """
    ...
```

**Error behavior**: `get_building("unknown")` raises `KeyError(f"Unknown building 'unknown'. Available: {sorted(BUILDINGS)}")`.

---

## Stream B: `thermalgym/policies.py`

**Dependencies**: none (only uses the `Obs`/`Action` dict contracts above)  
**Can start**: immediately

### File: `thermalgym/policies.py`

All policies are callable classes. All implement `__call__(self, obs: dict) -> dict`.  
No inheritance. No abstract base classes.

#### `Baseline`

Holds setpoints constant regardless of time or price.

```python
class Baseline:
    def __init__(
        self,
        heat_setpoint: float = 68.0,   # °F
        cool_setpoint: float = 76.0,   # °F
    ) -> None: ...

    def __call__(self, obs: dict) -> dict:
        return {"heat_setpoint": self.heat_setpoint, "cool_setpoint": self.cool_setpoint}
```

#### `PreCool`

Lowers cooling setpoint by `precool_offset` during `[peak_start - precool_hours, peak_start)`.  
Raises cooling setpoint by `setback` during `[peak_start, peak_end)` to rest HVAC.  
Otherwise holds baseline setpoints.

```python
class PreCool:
    def __init__(
        self,
        precool_offset: float = 2.0,    # °F below base_cool during pre-cool window
        precool_hours: float = 2,        # hours before peak_start to begin pre-cooling
        peak_start: int = 17,            # hour when peak period begins (inclusive)
        peak_end: int = 20,              # hour when peak period ends (exclusive)
        setback: float = 2.0,            # °F above base_cool during peak
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None: ...

    def __call__(self, obs: dict) -> dict:
        """
        Logic:
            hour in [peak_start - precool_hours, peak_start):
                cool_setpoint = base_cool - precool_offset
            hour in [peak_start, peak_end):
                cool_setpoint = base_cool + setback
            otherwise:
                cool_setpoint = base_cool
            heat_setpoint always = base_heat
        
        peak_start - precool_hours may be fractional; compare obs["hour"] >= threshold.
        """
        ...
```

#### `PreHeat`

Mirrors PreCool for heating. Raises heating setpoint before peak, lowers during peak.

```python
class PreHeat:
    def __init__(
        self,
        preheat_offset: float = 2.0,    # °F above base_heat during pre-heat window
        preheat_hours: float = 2,
        peak_start: int = 17,
        peak_end: int = 20,
        setback: float = 2.0,            # °F below base_heat during peak
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None: ...

    def __call__(self, obs: dict) -> dict:
        """
        Logic:
            hour in [peak_start - preheat_hours, peak_start):
                heat_setpoint = base_heat + preheat_offset
            hour in [peak_start, peak_end):
                heat_setpoint = base_heat - setback
            otherwise:
                heat_setpoint = base_heat
            cool_setpoint always = base_cool
        """
        ...
```

#### `Setback`

Raises cooling setpoint (or lowers heating setpoint) during peak to reduce load.  
Does NOT pre-condition.

```python
class Setback:
    def __init__(
        self,
        magnitude: float = 4.0,     # °F
        peak_start: int = 17,
        peak_end: int = 20,
        mode: str = "cooling",      # "cooling" | "heating" | "both"
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None: ...

    def __call__(self, obs: dict) -> dict:
        """
        During peak (hour in [peak_start, peak_end)):
            if mode in ("cooling", "both"): cool_setpoint = base_cool + magnitude
            if mode in ("heating", "both"): heat_setpoint = base_heat - magnitude
        Outside peak: baseline setpoints.
        """
        ...
```

#### `PriceResponse`

Linear mapping from price to setpoint adjustment.

```python
class PriceResponse:
    def __init__(
        self,
        threshold_low: float = 0.10,    # $/kWh; below this, loosen setpoint
        threshold_high: float = 0.25,   # $/kWh; above this, tighten setpoint
        adjust_low: float = -1.0,       # °F applied to cool_setpoint below threshold_low
        adjust_high: float = 2.0,       # °F applied to cool_setpoint above threshold_high
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None: ...

    def __call__(self, obs: dict) -> dict:
        """
        price = obs["electricity_price"]
        
        if price >= threshold_high:
            cool_setpoint = base_cool + adjust_high
        elif price <= threshold_low:
            cool_setpoint = base_cool + adjust_low
        else:
            # linear interpolation between adjustments
            t = (price - threshold_low) / (threshold_high - threshold_low)
            cool_setpoint = base_cool + adjust_low + t * (adjust_high - adjust_low)
        
        heat_setpoint = base_heat  # never adjusted by price in this policy
        """
        ...
```

**Note**: All policy outputs are plain dicts with keys `heat_setpoint` and `cool_setpoint`. The env clamps values to valid ranges; policies do not need to clamp.

---

## Stream C: Package Scaffolding + Data Stubs

**Dependencies**: none  
**Can start**: immediately

### File: `thermalgym/pyproject.toml`

Place in the **repo root** (`/Users/amelamud/Desktop/thermal/thermalgym/` is just a source dir; `pyproject.toml` goes in `thermal/thermalgym/` if it's a standalone package, or in `thermal/` if installing from there).

Given the project structure (`thermal/thermalgym/` as the package source), create `thermal/thermalgym/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "thermalgym"
version = "0.1.0"
requires-python = ">=3.9"
description = "EnergyPlus-backed Gymnasium environment for residential HVAC demand response"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "pyarrow>=12.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["."]
include = ["thermalgym*"]

[tool.setuptools.package-data]
thermalgym = ["data/buildings/*.idf", "data/weather/*.epw"]
```

### File: `thermalgym/__init__.py` (scaffold only — finalized in Stream F)

Create now with a version marker:
```python
__version__ = "0.1.0"
```

### Directory structure to create:

```
thermalgym/
├── __init__.py
├── env.py          (Stream D)
├── buildings.py    (Stream A)
├── policies.py     (Stream B)
├── generate.py     (Stream E)
└── data/
    ├── buildings/
    │   ├── small_cold_heatpump.idf
    │   ├── medium_cold_heatpump.idf
    │   ├── large_cold_resistance.idf
    │   ├── small_hot_heatpump.idf
    │   ├── medium_hot_heatpump.idf
    │   ├── large_hot_ac.idf
    │   ├── small_mixed_heatpump.idf
    │   ├── medium_mixed_heatpump.idf
    │   └── large_mixed_ac.idf
    └── weather/
        ├── USA_MN_Duluth.726550_TMY3.epw        (cold / Zone 6)
        ├── USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw  (hot / Zone 2)
        └── USA_OH_Columbus.Intl.AP.724280_TMY3.epw                  (mixed / Zone 4)
```

**IDF files**: Must be real EnergyPlus IDF files. Source from ResStock archetypes (NREL) or EnergyPlus example files. The IDF must:
- Define exactly one thermal zone
- Include `ZoneControl:Thermostat` referencing `ThermostatSetpoint:DualSetpoint` objects
- Include electric HVAC system matching the building's `hvac_type`
- Use the `RunPeriod` object (dates will be overridden at runtime)

**EPW files**: Download from EnergyPlus weather database. Three files needed (one per climate zone). Filenames must match the paths in `buildings.py`.

**For development without real IDF/EPW files**: Create `thermalgym/data/buildings/.gitkeep` and `thermalgym/data/weather/.gitkeep` with a `README` noting download instructions.

### Default price signal (embed in `env.py` or a `_defaults.py`)

California TOU default — 24 values, $/kWh:
```python
# Off-peak: $0.09, On-peak (4pm-9pm): $0.45, Mid-peak: $0.15
_CA_TOU_PRICES = np.array([
    0.09, 0.09, 0.09, 0.09, 0.09, 0.09,  # 0-5
    0.09, 0.09, 0.09, 0.09, 0.09, 0.09,  # 6-11
    0.15, 0.15, 0.15, 0.15, 0.45, 0.45,  # 12-17 (peak starts h17)
    0.45, 0.45, 0.45, 0.15, 0.09, 0.09,  # 18-23 (peak ends h20)
])
```

---

## Stream D: `thermalgym/env.py`

**Dependencies**: `buildings.py` (import `Building`, `get_building`, `BUILDINGS`)  
**Can start**: after Stream A is complete (or in parallel if `Building` type is treated as a forward reference / stub)

### File: `thermalgym/env.py`

```python
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from thermalgym.buildings import Building, get_building, BUILDINGS

# Setpoint validity bounds — also documented in __init__.py
HEAT_MIN, HEAT_MAX = 55.0, 75.0
COOL_MIN, COOL_MAX = 70.0, 85.0

# Default CA TOU price signal
_CA_TOU_PRICES = np.array([
    0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
    0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
    0.15, 0.15, 0.15, 0.15, 0.45, 0.45,
    0.45, 0.45, 0.45, 0.15, 0.09, 0.09,
])

# Per-process warmup cache: (building_id, date_str) → True
# Signals that warmup was completed; subsequent resets re-run from warmup start
# but this sentinel prevents double-counting warmup time in logs.
_WARMUP_CACHE: dict[tuple[str, str], bool] = {}


class ThermalEnv:
    """
    EnergyPlus-backed thermostat setpoint environment.
    
    NOT a Gymnasium env. No reward. No action/obs spaces. No RL assumptions.
    A policy is any callable: obs_dict -> action_dict.
    """

    def __init__(
        self,
        building: str | Building,
        timestep_minutes: int = 5,
        run_period_days: int = 1,
        price_signal: Optional[np.ndarray] = None,
        deadband_f: float = 1.0,
        idf_path: Optional[Path] = None,
        epw_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            building: Building ID string or Building dataclass instance.
                      If str, looked up via get_building().
            timestep_minutes: Simulation resolution. Must be 5, 15, or 60.
            run_period_days: Episode length in days.
            price_signal: Array of 24 hourly electricity prices ($/kWh).
                          If None, uses CA TOU default.
            deadband_f: Thermostat deadband in °F. Patches ZoneControl:Thermostat
                        in IDF at load time.
            idf_path: Override IDF path (for custom buildings not in registry).
            epw_path: Override EPW path (for custom buildings not in registry).
        
        Raises:
            ValueError: if building string not in BUILDINGS, or timestep_minutes invalid.
            FileNotFoundError: if IDF or EPW file does not exist.
        """
        ...

    # ── Public interface ─────────────────────────────────────────────────────

    def reset(self, date: str = "2017-07-01") -> dict:
        """
        Start a new episode.
        
        Args:
            date: ISO 8601 date string ("YYYY-MM-DD"). Episode begins at 00:00 on this date.
                  EnergyPlus runs a 7-day warmup starting at (date - 7 days) silently.
        
        Returns:
            First observation dict (at 00:00 of the specified date, after warmup).
        
        Raises:
            ValueError: if date string is not parseable.
        
        Side effects:
            - Terminates any running EnergyPlus process.
            - Clears episode history.
            - Sets self.done = False.
        """
        ...

    def step(self, action: dict) -> dict:
        """
        Advance one timestep.
        
        Args:
            action: {"heat_setpoint": float, "cool_setpoint": float}
                    Values are clamped internally to [HEAT_MIN,HEAT_MAX] / [COOL_MIN,COOL_MAX].
                    Extra keys in action dict are ignored.
        
        Returns:
            Observation dict for the NEW timestep (after applying action).
        
        Raises:
            RuntimeError: if called before reset() or after episode ends.
                          Message: "Episode has ended; call reset()."
        
        Side effects:
            - Appends row to internal history buffer.
            - Sets self.done = True on final timestep.
        """
        ...

    @property
    def done(self) -> bool:
        """True after the final timestep of run_period_days has been returned."""
        ...

    @property
    def history(self) -> pd.DataFrame:
        """
        Timeseries of all timesteps so far in the current episode.
        
        Columns (in order):
            timestamp      datetime64[ns]  wall-clock timestamp
            indoor_temp    float64         °F
            outdoor_temp   float64         °F
            hvac_power_kw  float64         kW
            hvac_mode      object          "heating" | "cooling" | "off"
            heat_setpoint  float64         °F (after clamping)
            cool_setpoint  float64         °F (after clamping)
            electricity_price float64      $/kWh
            hour           int64           0–23
            day_of_week    int64           0–6
            month          int64           1–12
        
        Returns a copy (not the internal buffer). Available mid-episode.
        Returns empty DataFrame (correct schema) before first step().
        """
        ...

    # ── EnergyPlus integration (private) ────────────────────────────────────

    def _start_energyplus(self, start_date: str) -> None:
        """
        Spawn EnergyPlus via pyenergyplus runtime API.
        
        Implementation notes:
        - Import pyenergyplus.api.EnergyPlusAPI lazily (not all environments have it).
        - Create a new API state: api.state_manager.new_state()
        - Patch IDF deadband before running (see _patch_idf_deadband).
        - Register callback: api.runtime.callback_begin_zone_timestep_after_init_heat_balance()
        - Run EnergyPlus in a daemon thread so Python retains control.
        - Synchronize with main thread using two threading.Event objects:
            _ep_ready: EnergyPlus signals it has a new observation ready
            _py_ready: Python signals it has written new setpoints
        
        Warmup period handling:
        - EnergyPlus runs warmup automatically before the simulation period.
        - The callback is invoked during warmup too; check api.exchange.warmup_flag()
          and skip (do not expose) warmup timesteps to the user.
        - After warmup, the first real callback populates self._current_obs and
          sets _ep_ready so reset() can return the first observation.
        
        Callback function signature (registered with pyenergyplus):
            def _ep_callback(state) -> None
        
        Inside the callback:
        1. Initialize actuator/sensor handles on first non-warmup call.
        2. Read sensors into obs dict.
        3. Set self._current_obs = obs.
        4. Signal _ep_ready.
        5. Wait on _py_ready.
        6. Write setpoint actuators using self._pending_action.
        """
        ...

    def _get_handles(self, state) -> None:
        """
        Retrieve and cache EnergyPlus variable/actuator handles.
        Called once on first real (non-warmup) timestep.
        
        Sensor handles (api.exchange.get_variable_handle):
            indoor_temp:   ("Zone Mean Air Temperature", zone_name)
            outdoor_temp:  ("Site Outdoor Air Drybulb Temperature", "Environment")
            hvac_power_w:  ("Facility Total HVAC Electricity Demand Rate", "Whole Building")
            heating_rate:  ("Zone Mechanical Ventilation Mass Flow Rate", zone_name)  # for mode detection
        
        For hvac_mode detection, use:
            heating_power: ("Heating Coil Electric Power", coil_name)
            cooling_power: ("Cooling Coil Electric Power", coil_name)
            If both are 0: "off". If heating > 0: "heating". If cooling > 0: "cooling".
            Alternative: ("HVAC System Solver Iteration Count", ...) — see EnergyPlus output vars.
        
        Actuator handles (api.exchange.get_actuator_handle):
            heat_sp: ("Zone Temperature Control", "Heating Setpoint", zone_name)
            cool_sp: ("Zone Temperature Control", "Cooling Setpoint", zone_name)
        
        zone_name: Read from IDF. For bundled IDFs, zone name is "LIVING ZONE".
                   Store as class constant per building or read dynamically.
        """
        ...

    def _read_obs(self, state) -> dict:
        """
        Read all sensor values and return obs dict.
        Converts hvac_power from W → kW.
        Sets electricity_price from self._price_signal[current_hour].
        Sets day_of_week from EnergyPlus date (or compute from timestamp).
        """
        ...

    def _write_setpoints(self, state, action: dict) -> None:
        """
        Write clamped setpoints via actuators.
        heat = clamp(action["heat_setpoint"], HEAT_MIN, HEAT_MAX)
        cool = clamp(action["cool_setpoint"], COOL_MIN, COOL_MAX)
        api.exchange.set_actuator_value(state, self._heat_handle, heat)
        api.exchange.set_actuator_value(state, self._cool_handle, cool)
        """
        ...

    def _patch_idf_deadband(self, deadband_f: float) -> Path:
        """
        Write a modified copy of the IDF with the deadband value patched.
        
        EnergyPlus IDF field to patch:
            Object: ThermostatSetpoint:DualSetpoint
            Field: Temperature Difference Between Cutout And Setpoint  [deltaC]
        
        deadband_f is in °F; convert to °C before writing: deadband_c = deadband_f * 5/9.
        
        Returns path to the patched IDF (written to a tempfile).
        Cached per (idf_path, deadband_f) to avoid re-patching.
        """
        ...

    def _stop_energyplus(self) -> None:
        """
        Signal EnergyPlus to stop cleanly.
        Set a stop flag, then set _py_ready so the callback unblocks and sees the flag.
        Join the EnergyPlus thread with a timeout (5s).
        api.state_manager.delete_state(state)
        """
        ...
```

### EnergyPlus variable / actuator names reference

These are standard EnergyPlus output variable names. Confirm against EnergyPlus I/O Reference:

| Purpose | Type | Component Type | Control Type / Variable | Component Name |
|---------|------|---------------|------------------------|----------------|
| Indoor temp | Variable | "Zone Mean Air Temperature" | — | zone_name |
| Outdoor temp | Variable | "Site Outdoor Air Drybulb Temperature" | — | "Environment" |
| HVAC power | Variable | "Facility Total HVAC Electricity Demand Rate" | — | "Whole Building" |
| Heating SP | Actuator | "Zone Temperature Control" | "Heating Setpoint" | zone_name |
| Cooling SP | Actuator | "Zone Temperature Control" | "Cooling Setpoint" | zone_name |

**hvac_mode** detection: Query `"Heating Coil Electric Power"` and `"Cooling Coil Electric Power"` on the coil objects defined in the IDF. Names vary by IDF — read coil names from IDF header, or use `"Facility Total Heating Electricity Rate"` vs `"Facility Total Cooling Electricity Rate"` (whole-building aggregates).

---

## Stream E: `thermalgym/generate.py`

**Dependencies**: `env.py` (ThermalEnv), `policies.py` (Baseline, PreCool, Setback)  
**Can start**: after Streams A, B, D are complete

### File: `thermalgym/generate.py`

```python
from __future__ import annotations

import multiprocessing
import random
from pathlib import Path
from typing import Callable, Optional, Union
import numpy as np
import pandas as pd

from thermalgym.env import ThermalEnv
from thermalgym.buildings import BUILDINGS, get_building, Building
from thermalgym.policies import Baseline, PreCool, Setback
```

#### `generate_episodes()`

```python
def generate_episodes(
    output: Union[str, Path],
    n_episodes: int = 1000,
    buildings: Union[str, list[str]] = "all",
    modes: list[str] = ["heat_increase", "cool_decrease"],
    gap_range: tuple[float, float] = (1.0, 5.0),
    outdoor_temp_range: Optional[tuple[float, float]] = None,
    timestep_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate setpoint-response training episodes.
    
    Args:
        output: Path to write Parquet file.
        n_episodes: Total episodes across all buildings (distributed proportionally).
        buildings: "all" or list of building IDs.
        modes: ["heat_increase"], ["cool_decrease"], or both.
        gap_range: (min_gap, max_gap) in °F for setpoint-to-temp gap.
        outdoor_temp_range: Filter episodes by outdoor temp. None = no filter.
        timestep_minutes: Simulation resolution.
        seed: RNG seed for reproducibility.
    
    Returns:
        DataFrame with setpoint_responses.parquet schema (see below).
        Also writes to output path as Parquet.
    
    Output schema (matches SRS Appendix A):
        home_id          str       Building ID
        episode_id       int64     Unique episode identifier (global, not per-building)
        mode             str       "heat_increase" | "cool_decrease"
        timestep         int64     0, 1, 2, ... within episode
        timestamp        datetime64[ns]
        indoor_temp      float64   °F
        outdoor_temp     float64   °F
        setpoint         float64   °F (active setpoint: heating SP for heat_increase, else cooling SP)
        hvac_power_kw    float64   kW
        hvac_on          bool      hvac_mode != "off"
        hour             int64     0–23
        month            int64     1–12
        time_to_target_min float64  Minutes from episode start to setpoint reached (NaN after first row)
    
    Implementation:
    1. Resolve building list from BUILDINGS if "all".
    2. Distribute n_episodes proportionally across buildings (round-robin remainder).
    3. Dispatch to multiprocessing.Pool:
       - Group episodes by building (batched by building per worker — see DESIGN §4.4).
       - Each worker calls _run_building_episodes(building_id, n, rng_seed, ...).
    4. Collect results, assign global episode_ids (monotonically increasing).
    5. Concatenate, write Parquet, return DataFrame.
    
    Episode generation logic (in _run_building_episodes):
    - Pick random date in 2017 (Jan 1 – Nov 30 to allow run_period_days buffer).
    - Pick random hour for episode start (0–23).
    - For "heat_increase":
        - gap = rng.uniform(*gap_range)
        - heat_setpoint_new = indoor_temp + gap (clamped to [HEAT_MIN, HEAT_MAX])
        - Run until indoor_temp >= heat_setpoint_new, or 4 hours, whichever first.
    - For "cool_decrease":
        - gap = rng.uniform(*gap_range)
        - cool_setpoint_new = indoor_temp - gap (clamped to [COOL_MIN, COOL_MAX])
        - Run until indoor_temp <= cool_setpoint_new, or 4 hours, whichever first.
    - time_to_target_min: computed at episode end; written only in first row.
      If setpoint not reached (timeout), set to NaN.
    - Apply outdoor_temp_range filter: if outdoor_temp on episode start not in range, skip.
    
    Warmup reuse across episodes (per worker):
    - Each worker creates one ThermalEnv per building.
    - Call reset() with the episode's start date.
    - Warmup runs automatically inside ThermalEnv.reset().
    - Workers do not share ThermalEnv instances across processes.
    """
    ...
```

#### `evaluate()`

```python
def evaluate(
    policy: Union[Callable, dict[str, Callable]],
    buildings: Union[str, list[str]] = "all",
    scenarios: list[str] = ["baseline", "precool", "setback"],
    n_days: int = 7,
    start_date: str = "2017-07-01",
    price_signal: Optional[np.ndarray] = None,
    timestep_minutes: int = 15,
) -> pd.DataFrame:
    """
    Evaluate one or more policies across buildings and DR scenarios.
    
    Args:
        policy: Single callable OR dict mapping name→callable.
                If dict, all policies run against identical scenario instances.
                If single callable, treated as {"policy": policy}.
                A Baseline() run is always added implicitly (as "baseline").
        buildings: "all" or list of building IDs.
        scenarios: Subset of ["baseline", "precool", "setback", "price_response"].
        n_days: Number of days per evaluation run.
        start_date: ISO date string for the start of evaluation.
        price_signal: 24-element array ($/kWh). None = CA TOU default.
        timestep_minutes: Resolution.
    
    Returns:
        Tidy DataFrame. One row per (policy, building, scenario).
        Columns:
            policy                  str
            building                str
            scenario                str
            total_kwh               float64   total energy consumption
            peak_kw                 float64   max instantaneous power
            par                     float64   peak-to-average ratio
            peak_kwh                float64   energy during peak period (17–20h by default)
            shifted_kwh             float64   energy moved from peak to off-peak vs "baseline" scenario
                                              (0.0 for baseline rows)
            comfort_violations_h    float64   hours with temp outside comfort bounds
                                              (comfort = [heat_sp - 1°F, cool_sp + 1°F])
            discomfort_degree_hours float64   sum(deviation × duration) over all OOB timesteps
            max_temp_deviation_f    float64   largest single-timestep deviation from active setpoint
            cost_usd                float64   total cost given price_signal
            cost_savings_usd        float64   cost_usd reduction vs baseline policy, same scenario
                                              (0.0 for baseline rows)
    
    Implementation:
    - Normalize policy to dict, always include Baseline() as "baseline".
    - For each (building, scenario) pair:
        - Determine the DR scenario's policy for each named policy:
            "baseline": use Baseline()
            "precool":  use PreCool() with default params
            "setback":  use Setback() with default params
            "price_response": use PriceResponse()
          NOTE: the *scenario* determines how the ENVIRONMENT is configured
          (which price signal, which peak window, etc.), NOT which policy runs.
          All named policies in the `policy` dict run against each scenario.
        - Run ThermalEnv for n_days with the given policy.
        - Compute metrics from env.history.
    - Append row to results list.
    - Return pd.DataFrame(results).
    
    Peak period for metrics: hours 17–19 inclusive (matches CA TOU signal).
    Comfort bounds: [heat_setpoint - 1, cool_setpoint + 1] (1°F deadband each side).
    PAR = peak_kw / mean_kw (over all timesteps in run).
    shifted_kwh: peak_kwh(baseline_scenario) - peak_kwh(this_scenario); 0 if baseline.
    cost_savings_usd: cost_usd("baseline" policy, same scenario) - cost_usd(this policy, same scenario).
    """
    ...
```

#### Private helper: `_compute_metrics()`

```python
def _compute_metrics(
    history: pd.DataFrame,
    price_signal: np.ndarray,
    baseline_peak_kwh: float,
    baseline_cost: float,
    peak_hours: tuple[int, int] = (17, 20),
) -> dict:
    """
    Compute all evaluation metrics from a completed episode history.
    
    Args:
        history: env.history DataFrame.
        price_signal: 24-element price array.
        baseline_peak_kwh: peak_kwh from the baseline scenario (for shifted_kwh).
        baseline_cost: cost_usd from the baseline policy run (for cost_savings_usd).
        peak_hours: (start_inclusive, end_exclusive) hours of peak period.
    
    Returns dict with keys matching evaluate() output columns (excluding policy/building/scenario).
    """
    ...
```

---

## Stream F: `thermalgym/__init__.py` (final)

**Dependencies**: all other streams complete  
**Can start**: last

```python
"""
ThermalGym — EnergyPlus-backed demand response simulation environment.

Quick start::

    import thermalgym

    env = thermalgym.ThermalEnv(building="medium_cold_heatpump")
    obs = env.reset(date="2017-07-15")
    while not env.done:
        action = {"heat_setpoint": 70, "cool_setpoint": 76}
        obs = env.step(action)
    print(env.history.head())

"""

from thermalgym.env import ThermalEnv, HEAT_MIN, HEAT_MAX, COOL_MIN, COOL_MAX
from thermalgym.buildings import BUILDINGS, get_building, get_buildings, Building
from thermalgym.generate import generate_episodes, evaluate
from thermalgym.policies import Baseline, PreCool, PreHeat, Setback, PriceResponse

__version__ = "0.1.0"
__all__ = [
    # Environment
    "ThermalEnv",
    # Setpoint bounds
    "HEAT_MIN", "HEAT_MAX", "COOL_MIN", "COOL_MAX",
    # Buildings
    "Building", "BUILDINGS", "get_building", "get_buildings",
    # Data generation
    "generate_episodes", "evaluate",
    # Policies
    "Baseline", "PreCool", "PreHeat", "Setback", "PriceResponse",
]
```

---

## Parallel Execution Matrix

| Stream | File | Depends On | Start Condition |
|--------|------|-----------|----------------|
| A | `buildings.py` | — | Now |
| B | `policies.py` | — | Now |
| C | scaffolding, `data/` dirs | — | Now |
| D | `env.py` | A | After A lands |
| E | `generate.py` | A, B, D | After D lands |
| F | `__init__.py` (final) | A, B, C, D, E | Last |

Recommended agent assignments for 3 simultaneous agents:
- **Agent 1**: Streams A + C (buildings.py, scaffolding, data dirs)
- **Agent 2**: Stream B (policies.py — self-contained, heavily testable)
- **Agent 3**: Start on Stream D (env.py) — can stub `from thermalgym.buildings import Building` and proceed

---

## Implementation Constraints

### Must match SRS Appendix A schema exactly
Generated Parquet columns must include `home_id`, `episode_id`, `mode`, `timestep`, `timestamp`, `indoor_temp`, `outdoor_temp`, `setpoint`, `hvac_power_kw`, `hvac_on`, `hour`, `month`, `time_to_target_min`.  
`time_to_target_min` is a float (NaN after first row per episode, NaN if timeout).

### pyenergyplus import guard
`env.py` must import pyenergyplus lazily (inside `__init__` or `reset()`), not at module top. This lets `buildings.py`, `policies.py`, and `generate.py` import without EnergyPlus installed.

```python
# Inside ThermalEnv.__init__ or _start_energyplus:
try:
    from pyenergyplus.api import EnergyPlusAPI
except ImportError:
    raise ImportError(
        "pyenergyplus not found. Install EnergyPlus 23.1+ and ensure its Python API "
        "is on sys.path. See: https://energyplus.net/downloads"
    )
```

### Thread synchronization in env.py
EnergyPlus runs in a background thread. Use `threading.Event` for lockstep:
```python
self._ep_ready = threading.Event()   # EP signals observation ready
self._py_ready = threading.Event()   # Python signals action written
self._stop_flag = threading.Event()  # Python signals shutdown
```
Callback flow:
1. EP callback fires → reads sensors → sets `_ep_ready` → waits on `_py_ready`
2. `step()` → waits on `_ep_ready` → reads `_current_obs` → writes `_pending_action` → sets `_py_ready`
3. Repeat

### Episode end handling
The episode ends after `run_period_days * 24 * 60 / timestep_minutes` timesteps.  
The env counts completed timesteps internally. On the final timestep:
- Return the terminal obs from `step()`.
- Set `self.done = True`.
- Signal EnergyPlus to stop via `_stop_energyplus()`.

### History column order
Must be exactly: `timestamp, indoor_temp, outdoor_temp, hvac_power_kw, hvac_mode, heat_setpoint, cool_setpoint, electricity_price, hour, day_of_week, month`

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| IDF zone name varies by archetype | Read zone name programmatically from IDF before simulation; store in `Building.zone_name` field or parse in `_get_handles()` |
| hvac_mode coil names vary by IDF | Use whole-building aggregates: `"Facility Total Heating Electricity Rate"` and `"Facility Total Cooling Electricity Rate"` instead of per-coil |
| pyenergyplus API differences across EnergyPlus versions | Target EnergyPlus 23.1+; document minimum version in pyproject.toml |
| Warmup cache non-trivial with pyenergyplus (can't save/restore EnergyPlus state) | For v0.1, warmup always runs. Cache sentinel prevents log double-counting but does not skip computation |
| Multiprocessing on Windows (spawn vs fork) | Use `multiprocessing.get_context("spawn")` explicitly in `generate_episodes()` |
| EnergyPlus thread crash leaves env in bad state | Wrap EnergyPlus thread in try/except; propagate exception to main thread via queue; `step()` re-raises it |
