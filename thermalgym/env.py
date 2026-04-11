from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from thermalgym.buildings import Building, get_building, BUILDINGS

# Setpoint validity bounds
HEAT_MIN, HEAT_MAX = 55.0, 75.0
COOL_MIN, COOL_MAX = 70.0, 85.0

# Default CA TOU price signal — off-peak $0.09, mid-peak $0.15, on-peak (16-21h) $0.45
_CA_TOU_PRICES = np.array([
    0.09, 0.09, 0.09, 0.09, 0.09, 0.09,  # 0-5
    0.09, 0.09, 0.09, 0.09, 0.09, 0.09,  # 6-11
    0.15, 0.15, 0.15, 0.15, 0.45, 0.45,  # 12-17 (peak starts h17)
    0.45, 0.45, 0.45, 0.15, 0.09, 0.09,  # 18-23 (peak ends h20)
])

_HISTORY_COLUMNS = [
    "timestamp", "indoor_temp", "outdoor_temp", "hvac_power_kw", "hvac_mode",
    "heat_setpoint", "cool_setpoint", "electricity_price", "hour", "day_of_week", "month",
]

# Cache of patched IDF paths: (original_idf_path, deadband_f) -> patched_path
_IDF_PATCH_CACHE: dict[tuple[str, float], Path] = {}


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
            deadband_f: Thermostat deadband in °F.
            idf_path: Override IDF path (for custom buildings not in registry).
            epw_path: Override EPW path (for custom buildings not in registry).

        Raises:
            ValueError: if building string not in BUILDINGS, or timestep_minutes invalid.
            FileNotFoundError: if IDF or EPW file does not exist.
        """
        if isinstance(building, str):
            building = get_building(building)
        self._building = building

        if timestep_minutes not in (5, 15, 60):
            raise ValueError(f"timestep_minutes must be 5, 15, or 60; got {timestep_minutes}")
        self._timestep_minutes = timestep_minutes
        self._run_period_days = run_period_days
        self._total_steps = run_period_days * 24 * 60 // timestep_minutes

        self._price_signal = _CA_TOU_PRICES.copy() if price_signal is None else np.asarray(price_signal)
        if self._price_signal.shape != (24,):
            raise ValueError(f"price_signal must have 24 elements; got {self._price_signal.shape}")

        self._deadband_f = deadband_f
        self._idf_path = idf_path or building.idf_path
        self._epw_path = epw_path or building.epw_path

        if not self._idf_path.exists():
            raise FileNotFoundError(
                f"IDF file not found: {self._idf_path}\n"
                "Download building IDF files — see thermalgym/data/buildings/README.md"
            )
        if not self._epw_path.exists():
            raise FileNotFoundError(
                f"EPW file not found: {self._epw_path}\n"
                "Download weather EPW files — see thermalgym/data/weather/README.md"
            )

        # Episode state
        self._done: bool = True
        self._step_count: int = 0
        self._history_rows: list[dict] = []
        self._current_obs: Optional[dict] = None
        self._pending_action: Optional[dict] = None
        self._start_date: Optional[str] = None

        # Threading synchronization
        self._ep_ready = threading.Event()
        self._py_ready = threading.Event()
        self._stop_flag = threading.Event()
        self._ep_thread: Optional[threading.Thread] = None
        self._ep_exception: Optional[Exception] = None

        # EnergyPlus handle cache
        self._handles_initialized: bool = False
        self._indoor_temp_handle: Optional[int] = None
        self._outdoor_temp_handle: Optional[int] = None
        self._hvac_power_handle: Optional[int] = None
        self._heating_rate_handle: Optional[int] = None
        self._cooling_rate_handle: Optional[int] = None
        self._heat_sp_handle: Optional[int] = None
        self._cool_sp_handle: Optional[int] = None
        self._ep_state = None
        self._api = None

        # Current setpoints (updated each step from pending action)
        self._current_heat_sp: float = 68.0
        self._current_cool_sp: float = 76.0

    # ── Public interface ─────────────────────────────────────────────────────

    def reset(self, date: str = "2017-07-01") -> dict:
        """
        Start a new episode.

        Args:
            date: ISO 8601 date string ("YYYY-MM-DD"). Episode begins at 00:00.

        Returns:
            First observation dict (at 00:00 of the specified date, after warmup).

        Raises:
            ValueError: if date string is not parseable.
        """
        try:
            pd.Timestamp(date)
        except Exception:
            raise ValueError(f"Invalid date string: {date!r}. Expected 'YYYY-MM-DD'.")

        # Stop any running EnergyPlus process
        if self._ep_thread is not None and self._ep_thread.is_alive():
            self._stop_energyplus()

        # Reset state
        self._done = False
        self._step_count = 0
        self._history_rows = []
        self._current_obs = None
        self._pending_action = {"heat_setpoint": 68.0, "cool_setpoint": 76.0}
        self._start_date = date
        self._ep_exception = None
        self._handles_initialized = False
        self._ep_ready.clear()
        self._py_ready.clear()
        self._stop_flag.clear()

        # Start EnergyPlus
        self._start_energyplus(date)

        # Wait for first observation (after warmup) — timeout guards against EP startup failure
        if not self._ep_ready.wait(timeout=300.0):
            raise RuntimeError("EnergyPlus did not produce first observation within 5 minutes.")
        self._ep_ready.clear()

        if self._ep_exception is not None:
            raise self._ep_exception

        return dict(self._current_obs)

    def step(self, action: dict) -> dict:
        """
        Advance one timestep.

        Args:
            action: {"heat_setpoint": float, "cool_setpoint": float}

        Returns:
            Observation dict for the new timestep.

        Raises:
            RuntimeError: if called before reset() or after episode ends.
        """
        if self._done:
            raise RuntimeError("Episode has ended; call reset().")
        if self._current_obs is None:
            raise RuntimeError("Episode has ended; call reset().")

        if self._ep_exception is not None:
            raise self._ep_exception

        # Clamp and store action
        heat = float(np.clip(action.get("heat_setpoint", 68.0), HEAT_MIN, HEAT_MAX))
        cool = float(np.clip(action.get("cool_setpoint", 76.0), COOL_MIN, COOL_MAX))
        self._pending_action = {"heat_setpoint": heat, "cool_setpoint": cool}
        self._current_heat_sp = heat
        self._current_cool_sp = cool

        # Record current obs to history before advancing
        row = dict(self._current_obs)
        row["heat_setpoint"] = heat
        row["cool_setpoint"] = cool
        self._history_rows.append(row)

        # Signal EnergyPlus to proceed
        self._py_ready.set()

        self._step_count += 1

        if self._step_count >= self._total_steps:
            self._done = True
            # Wait briefly for EP to process final step
            self._ep_ready.wait(timeout=30.0)
            self._ep_ready.clear()
            self._stop_energyplus()
            # Return the obs we had before signaling done
            return dict(row)

        # Wait for next observation
        if not self._ep_ready.wait(timeout=60.0):
            raise RuntimeError("EnergyPlus timed out waiting for next timestep.")
        self._ep_ready.clear()

        if self._ep_exception is not None:
            raise self._ep_exception

        return dict(self._current_obs)

    @property
    def done(self) -> bool:
        """True after the final timestep of run_period_days has been returned."""
        return self._done

    @property
    def history(self) -> pd.DataFrame:
        """
        Timeseries of all timesteps so far in the current episode.

        Returns a copy. Available mid-episode.
        Returns empty DataFrame (correct schema) before first step().
        """
        if not self._history_rows:
            return pd.DataFrame(columns=_HISTORY_COLUMNS).astype({
                "timestamp": "datetime64[ns]",
                "indoor_temp": "float64",
                "outdoor_temp": "float64",
                "hvac_power_kw": "float64",
                "hvac_mode": "object",
                "heat_setpoint": "float64",
                "cool_setpoint": "float64",
                "electricity_price": "float64",
                "hour": "int64",
                "day_of_week": "int64",
                "month": "int64",
            })
        df = pd.DataFrame(self._history_rows, columns=_HISTORY_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ── EnergyPlus integration (private) ────────────────────────────────────

    def _start_energyplus(self, start_date: str) -> None:
        """Spawn EnergyPlus via pyenergyplus runtime API in a daemon thread."""
        try:
            from pyenergyplus.api import EnergyPlusAPI
        except ImportError:
            # Try auto-detecting EnergyPlus on macOS
            import sys
            import glob
            candidates = sorted(glob.glob("/Applications/EnergyPlus-*"), reverse=True)
            if candidates:
                sys.path.insert(0, candidates[0])
            try:
                from pyenergyplus.api import EnergyPlusAPI
            except ImportError:
                raise ImportError(
                    "pyenergyplus not found. Install EnergyPlus 23.1+ and ensure its Python API "
                    "is on sys.path. See: https://energyplus.net/downloads"
                )

        self._api = EnergyPlusAPI()
        self._ep_state = self._api.state_manager.new_state()

        patched_idf = self._patch_idf_deadband(self._deadband_f)

        # Patch run period in IDF
        run_idf = self._patch_idf_run_period(patched_idf, start_date)

        api = self._api
        state = self._ep_state

        # Request output variables BEFORE the run (required for get_variable_handle to work)
        zone = self._building.zone_name
        api.exchange.request_variable(state, "Zone Mean Air Temperature", zone)
        api.exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")
        api.exchange.request_variable(state, "Facility Total HVAC Electricity Demand Rate", "Whole Building")
        api.exchange.request_variable(state, "Facility Total Heating Electricity Rate", "Whole Building")
        api.exchange.request_variable(state, "Facility Total Cooling Electricity Rate", "Whole Building")

        def _ep_callback(state_handle) -> None:
            try:
                # Skip warmup timesteps
                if api.exchange.warmup_flag(state_handle):
                    return

                if self._stop_flag.is_set():
                    api.runtime.stop_simulation(state_handle)
                    return

                # Initialize handles once (after warmup, variables are fully registered)
                if not self._handles_initialized:
                    self._get_handles(state_handle)
                    self._handles_initialized = True

                # Read observation
                obs = self._read_obs(state_handle)
                self._current_obs = obs

                # Write setpoints from previous action (actuators persist across timesteps)
                self._write_setpoints(state_handle, self._pending_action)

                # Signal Python that obs is ready
                self._ep_ready.set()

                # Wait for Python to provide next action
                self._py_ready.wait()
                self._py_ready.clear()

                if self._stop_flag.is_set():
                    api.runtime.stop_simulation(state_handle)

            except Exception as e:
                self._ep_exception = e
                self._ep_ready.set()  # Unblock waiting Python thread
                api.runtime.stop_simulation(state_handle)

        # Use end-of-timestep callback so all output variables are computed
        api.runtime.callback_end_zone_timestep_after_zone_reporting(state, _ep_callback)

        def _run_ep():
            try:
                argv = [
                    "-w", str(self._epw_path),
                    "-d", str(run_idf.parent),
                    str(run_idf),
                ]
                api.runtime.run_energyplus(state, argv)
            except Exception as e:
                self._ep_exception = e
                self._ep_ready.set()

        self._ep_thread = threading.Thread(target=_run_ep, daemon=True)
        self._ep_thread.start()

    def _get_handles(self, state) -> None:
        """Retrieve and cache EnergyPlus variable/actuator handles. Called once."""
        api = self._api
        zone_name = self._building.zone_name

        self._indoor_temp_handle = api.exchange.get_variable_handle(
            state, "Zone Mean Air Temperature", zone_name
        )
        self._outdoor_temp_handle = api.exchange.get_variable_handle(
            state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        self._hvac_power_handle = api.exchange.get_variable_handle(
            state, "Facility Total HVAC Electricity Demand Rate", "Whole Building"
        )
        self._heating_rate_handle = api.exchange.get_variable_handle(
            state, "Facility Total Heating Electricity Rate", "Whole Building"
        )
        self._cooling_rate_handle = api.exchange.get_variable_handle(
            state, "Facility Total Cooling Electricity Rate", "Whole Building"
        )
        self._heat_sp_handle = api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Heating Setpoint", zone_name
        )
        self._cool_sp_handle = api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Cooling Setpoint", zone_name
        )

    def _read_obs(self, state) -> dict:
        """Read all sensor values and return obs dict."""
        api = self._api

        indoor_temp = api.exchange.get_variable_value(state, self._indoor_temp_handle)
        outdoor_temp = api.exchange.get_variable_value(state, self._outdoor_temp_handle)
        hvac_power_w = api.exchange.get_variable_value(state, self._hvac_power_handle)
        heating_rate = api.exchange.get_variable_value(state, self._heating_rate_handle)
        cooling_rate = api.exchange.get_variable_value(state, self._cooling_rate_handle)

        if heating_rate > 0:
            hvac_mode = "heating"
        elif cooling_rate > 0:
            hvac_mode = "cooling"
        else:
            hvac_mode = "off"

        ep_hour = api.exchange.hour(state)
        ep_minute = self._read_energyplus_minute(state, int(ep_hour))
        ep_day_of_year = api.exchange.day_of_year(state)
        ep_year = api.exchange.year(state)
        ep_month = api.exchange.month(state)
        ep_day = api.exchange.day_of_month(state)

        # Compute day_of_week from date
        ts = pd.Timestamp(year=ep_year, month=ep_month, day=ep_day)
        day_of_week = ts.dayofweek  # 0=Monday

        def c_to_f(c: float) -> float:
            return c * 9.0 / 5.0 + 32.0

        return {
            "timestamp": ts + pd.Timedelta(hours=ep_hour, minutes=ep_minute),
            "indoor_temp": c_to_f(indoor_temp),
            "outdoor_temp": c_to_f(outdoor_temp),
            "hvac_power_kw": float(hvac_power_w) / 1000.0,
            "hvac_mode": hvac_mode,
            "heat_setpoint": self._current_heat_sp,
            "cool_setpoint": self._current_cool_sp,
            "electricity_price": float(self._price_signal[ep_hour % 24]),
            "hour": int(ep_hour),
            "day_of_week": int(day_of_week),
            "month": int(ep_month),
        }

    def _read_energyplus_minute(self, state, ep_hour: int) -> int:
        """Return minute within the current hour, falling back to the local step counter."""
        exchange = self._api.exchange
        if hasattr(exchange, "minutes"):
            return int(exchange.minutes(state))
        if hasattr(exchange, "minute"):
            return int(exchange.minute(state))
        current_time = getattr(exchange, "current_time", None)
        if current_time is not None:
            hour_fraction = float(current_time(state))
            return int(round((hour_fraction - ep_hour) * 60.0))
        return int((self._step_count * self._timestep_minutes) % 60)

    def _write_setpoints(self, state, action: dict) -> None:
        """Write clamped setpoints via actuators. EnergyPlus expects °C."""
        def f_to_c(f: float) -> float:
            return (f - 32.0) * 5.0 / 9.0

        api = self._api
        heat_f = float(np.clip(action["heat_setpoint"], HEAT_MIN, HEAT_MAX))
        cool_f = float(np.clip(action["cool_setpoint"], COOL_MIN, COOL_MAX))
        api.exchange.set_actuator_value(state, self._heat_sp_handle, f_to_c(heat_f))
        api.exchange.set_actuator_value(state, self._cool_sp_handle, f_to_c(cool_f))

    def _patch_idf_deadband(self, deadband_f: float) -> Path:
        """
        Write a modified copy of the IDF with deadband patched.

        deadband_f is in °F; converts to °C before writing: deadband_c = deadband_f * 5/9.
        Returns path to patched IDF (cached per (idf_path, deadband_f)).
        """
        cache_key = (str(self._idf_path), deadband_f)
        if cache_key in _IDF_PATCH_CACHE:
            return _IDF_PATCH_CACHE[cache_key]

        deadband_c = deadband_f * 5.0 / 9.0
        idf_text = self._idf_path.read_text(encoding="utf-8", errors="replace")

        # Patch ThermostatSetpoint:DualSetpoint deadband field
        # The field is "Temperature Difference Between Cutout And Setpoint"
        import re
        patched = re.sub(
            r'(ThermostatSetpoint:DualSetpoint[^;]*?Temperature Difference Between Cutout And Setpoint\s*,\s*)[0-9.]+',
            lambda m: m.group(1) + f"{deadband_c:.4f}",
            idf_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        tmp = tempfile.NamedTemporaryFile(
            suffix=".idf", delete=False, prefix=f"thermalgym_{self._building.id}_"
        )
        tmp.write(patched.encode("utf-8"))
        tmp.close()

        patched_path = Path(tmp.name)
        _IDF_PATCH_CACHE[cache_key] = patched_path
        return patched_path

    def _patch_idf_run_period(self, idf_path: Path, start_date: str) -> Path:
        """
        Write a copy of the IDF with RunPeriod and Timestep patched.
        Returns path to patched IDF.
        """
        import re

        start = pd.Timestamp(start_date)
        end = start + pd.Timedelta(days=self._run_period_days - 1)

        idf_text = idf_path.read_text(encoding="utf-8", errors="replace")

        # Patch Timestep to match timestep_minutes
        timesteps_per_hour = 60 // self._timestep_minutes
        idf_text = re.sub(
            r'^\s*Timestep\s*,\s*\d+\s*;',
            f'Timestep,{timesteps_per_hour};',
            idf_text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Replace RunPeriod begin/end month and day fields
        # Standard IDF RunPeriod format:
        #   RunPeriod,
        #     <name>,         ! Name
        #     <month>,        ! Begin Month
        #     <day>,          ! Begin Day of Month
        #     <year>,         ! Begin Year
        #     <month>,        ! End Month
        #     <day>,          ! End Day of Month
        #     <year>,         ! End Year
        # We do a simple replacement of the RunPeriod block.
        run_period_block = (
            f"RunPeriod,\n"
            f"    ThermalGymEpisode,  !- Name\n"
            f"    {start.month},      !- Begin Month\n"
            f"    {start.day},        !- Begin Day of Month\n"
            f"    {start.year},       !- Begin Year\n"
            f"    {end.month},        !- End Month\n"
            f"    {end.day},          !- End Day of Month\n"
            f"    {end.year},         !- End Year\n"
            f"    Sunday,             !- Day of Week for Start Day\n"
            f"    Yes,                !- Use Weather File Holidays and Special Days\n"
            f"    Yes,                !- Use Weather File DST Indicators\n"
            f"    Yes,                !- Apply Weekend Holiday Rule\n"
            f"    Yes,                !- Use Weather File Rain Indicators\n"
            f"    Yes;                !- Use Weather File Snow Indicators\n"
        )

        # Replace existing RunPeriod block
        patched = re.sub(
            r'RunPeriod,.*?;',
            run_period_block.rstrip(),
            idf_text,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )

        tmp = tempfile.NamedTemporaryFile(
            suffix=".idf", delete=False,
            prefix=f"thermalgym_{self._building.id}_{start_date}_",
        )
        tmp.write(patched.encode("utf-8"))
        tmp.close()
        return Path(tmp.name)

    def _stop_energyplus(self) -> None:
        """Signal EnergyPlus to stop cleanly and join the thread."""
        if self._ep_thread is None:
            return

        self._stop_flag.set()
        # Unblock callback if it's waiting on _py_ready
        self._py_ready.set()

        self._ep_thread.join(timeout=5.0)

        if self._ep_state is not None and self._api is not None:
            try:
                self._api.state_manager.delete_state(self._ep_state)
            except Exception:
                pass
            self._ep_state = None

        self._ep_thread = None
        self._handles_initialized = False

    def __del__(self):
        try:
            if self._ep_thread is not None and self._ep_thread.is_alive():
                self._stop_energyplus()
        except Exception:
            pass
