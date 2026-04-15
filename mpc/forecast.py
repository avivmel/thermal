from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import zlib

import numpy as np
import pandas as pd


class Forecast(Protocol):
    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float:
        raise NotImplementedError


class PerfectForecast:
    """Timestamped outdoor-temperature forecast with nearest-prior lookup."""

    def __init__(self, forecast: pd.Series) -> None:
        if forecast.empty:
            raise ValueError("forecast series must not be empty")
        series = forecast.copy()
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        self.forecast = series.astype(float)

    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float:
        timestamp = pd.Timestamp(timestamp)
        loc = self.forecast.index.searchsorted(timestamp, side="right") - 1
        if loc < 0:
            raise KeyError(timestamp)
        return float(self.forecast.iloc[loc])


class EPWForecast(PerfectForecast):
    """Outdoor-temperature forecast parsed from an EnergyPlus EPW weather file."""

    @classmethod
    def from_epw(cls, path: str | Path, year: int = 2017) -> EPWForecast:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        rows = pd.read_csv(
            path,
            skiprows=8,
            header=None,
            usecols=[1, 2, 3, 6],
            names=["month", "day", "hour", "dry_bulb_c"],
        )
        rows["timestamp"] = [
            _epw_timestamp(year, month, day, hour)
            for month, day, hour in zip(rows["month"], rows["day"], rows["hour"])
        ]
        dry_bulb_f = rows["dry_bulb_c"].astype(float) * 9.0 / 5.0 + 32.0
        series = pd.Series(dry_bulb_f.to_numpy(), index=pd.DatetimeIndex(rows["timestamp"]))
        return cls(series)


class PersistenceForecast:
    """Forecast that holds one outdoor temperature constant for all future times."""

    def __init__(self, outdoor_temp_f: float) -> None:
        self.outdoor_temp_f = float(outdoor_temp_f)

    @classmethod
    def from_state(cls, state) -> PersistenceForecast:
        return cls(outdoor_temp_f=float(state.outdoor_temp_f))

    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float:
        pd.Timestamp(timestamp)
        return self.outdoor_temp_f


@dataclass
class NoisyForecast:
    """Forecast wrapper with deterministic hourly AR(1) temperature noise."""

    base: Forecast
    ar1_phi: float = 0.7
    sigma_f: float = 2.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not -1.0 < self.ar1_phi < 1.0:
            raise ValueError("ar1_phi must be between -1 and 1")
        if self.sigma_f < 0:
            raise ValueError("sigma_f must be non-negative")
        self._errors_by_year: dict[int, pd.Series] = {}
        self._base_index_errors: pd.Series | None = self._build_base_index_errors()

    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float:
        timestamp = pd.Timestamp(timestamp)
        base_temp = float(self.base.outdoor_temp_at(timestamp))
        return base_temp + self._error_at(timestamp)

    def _error_at(self, timestamp: pd.Timestamp) -> float:
        if self._base_index_errors is not None:
            loc = self._base_index_errors.index.searchsorted(timestamp, side="right") - 1
            if loc < 0:
                raise KeyError(timestamp)
            return float(self._base_index_errors.iloc[loc])

        hour = timestamp.floor("h")
        if hour.year not in self._errors_by_year:
            self._errors_by_year[hour.year] = self._generate_year_errors(hour.year)
        return float(self._errors_by_year[hour.year].loc[hour])

    def _build_base_index_errors(self) -> pd.Series | None:
        series = getattr(self.base, "forecast", None)
        if not isinstance(series, pd.Series) or series.empty:
            return None
        index = pd.DatetimeIndex(pd.to_datetime(series.index)).sort_values()
        return pd.Series(self._generate_errors(len(index), self._seed_for_label("base-index")), index=index)

    def _generate_year_errors(self, year: int) -> pd.Series:
        index = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
        errors = self._generate_errors(len(index), self._seed_for_label(str(year)))
        return pd.Series(errors, index=index)

    def _generate_errors(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        innovations = rng.normal(0.0, self.sigma_f, size=n)
        errors = np.empty(n, dtype=float)
        previous = 0.0
        for i, innovation in enumerate(innovations):
            previous = self.ar1_phi * previous + innovation
            errors[i] = previous
        return errors

    def _seed_for_label(self, label: str) -> int:
        seed = 0 if self.seed is None else int(self.seed)
        label_hash = zlib.crc32(label.encode("utf-8"))
        return (seed + label_hash) % (2**32)


def _epw_timestamp(year: int, month: int, day: int, hour: int) -> pd.Timestamp:
    hour = int(hour)
    if not 1 <= hour <= 24:
        raise ValueError(f"EPW hour must be in 1..24, got {hour}")
    return pd.Timestamp(year=int(year), month=int(month), day=int(day)) + pd.Timedelta(hours=hour - 1)

