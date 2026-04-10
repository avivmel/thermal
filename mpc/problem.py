from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MPCConfig:
    timestep_minutes: int = 5
    min_temp_f: float = 60.0
    max_temp_f: float = 76.0
    temp_bin_width_f: float = 0.5
    action_min_f: float = 68.0
    action_max_f: float = 74.0
    action_bin_width_f: float = 1.0
    cooling_setpoint_f: float = 76.0
    lambda_energy: float = 1.0
    lambda_comfort: float = 200.0
    terminal_penalty: float = 1_000.0
    comfort_slack_f: float = 0.0
    target_reached_epsilon_f: float = 0.25

    @property
    def temperature_grid(self) -> np.ndarray:
        n_bins = int(round((self.max_temp_f - self.min_temp_f) / self.temp_bin_width_f))
        return self.min_temp_f + np.arange(n_bins + 1) * self.temp_bin_width_f

    @property
    def action_grid(self) -> np.ndarray:
        n_bins = int(round((self.action_max_f - self.action_min_f) / self.action_bin_width_f))
        return self.action_min_f + np.arange(n_bins + 1) * self.action_bin_width_f


@dataclass(frozen=True)
class MPCInputs:
    timestamps: Sequence[pd.Timestamp]
    outdoor_temps_f: Sequence[float]
    lower_comfort_f: Sequence[float]
    hvac_power_kw: float
    initial_indoor_temp_f: float
    initial_running: bool
    initial_timestamp: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if n == 0:
            raise ValueError("timestamps must be non-empty")
        if len(self.outdoor_temps_f) != n:
            raise ValueError("outdoor_temps_f must match timestamps length")
        if len(self.lower_comfort_f) != n:
            raise ValueError("lower_comfort_f must match timestamps length")

    @property
    def horizon_steps(self) -> int:
        return len(self.timestamps)

    def timestamp_at(self, index: int) -> pd.Timestamp:
        return pd.Timestamp(self.timestamps[index])

    def outdoor_at(self, index: int) -> float:
        return float(self.outdoor_temps_f[index])

    def lower_comfort_at(self, index: int) -> float:
        return float(self.lower_comfort_f[index])


def snap_to_grid(value: float, grid: np.ndarray) -> int:
    return int(np.abs(grid - value).argmin())


def build_daily_schedule(
    start: pd.Timestamp,
    horizon_steps: int,
    timestep_minutes: int,
    default_heat_f: float,
    peak_heat_f: float,
    peak_start_hour: int,
    peak_end_hour: int,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    timestamps = [
        pd.Timestamp(start) + pd.Timedelta(minutes=timestep_minutes * step)
        for step in range(horizon_steps)
    ]
    schedule = np.full(horizon_steps, default_heat_f, dtype=float)
    for idx, ts in enumerate(timestamps):
        if peak_start_hour <= ts.hour < peak_end_hour:
            schedule[idx] = peak_heat_f
    return timestamps, schedule
