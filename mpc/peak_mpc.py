from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, Union

import numpy as np
import pandas as pd

from mpc.model_interfaces import Direction, FirstPassagePredictor


Phase = Literal["normal", "precondition", "peak_coast", "peak_maintain"]
Mode = Literal["heating", "cooling"]


class WeatherForecast(Protocol):
    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float:
        raise NotImplementedError


ForecastInput = Union[WeatherForecast, Callable[[pd.Timestamp], float], pd.Series, None]


@dataclass(frozen=True)
class PeakWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if end <= start:
            raise ValueError(f"peak window end must be after start: {start} -> {end}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def contains(self, timestamp: pd.Timestamp) -> bool:
        timestamp = pd.Timestamp(timestamp)
        return self.start <= timestamp < self.end


@dataclass(frozen=True)
class MPCConfig:
    peak_windows: list[PeakWindow | tuple[pd.Timestamp, pd.Timestamp]]
    comfort_lower_f: float
    comfort_upper_f: float
    normal_heat_setpoint_f: float | None = None
    normal_cool_setpoint_f: float | None = None
    control_step_minutes: int = 5
    horizon_minutes: int = 360
    precondition_margin_minutes: float = 10.0
    drift_safety_margin_minutes: float = 10.0
    min_precondition_gap_f: float = 0.5
    max_precondition_lead_minutes: float = 240.0

    def __post_init__(self) -> None:
        if self.comfort_lower_f >= self.comfort_upper_f:
            raise ValueError("comfort_lower_f must be less than comfort_upper_f")
        if self.control_step_minutes <= 0:
            raise ValueError("control_step_minutes must be positive")
        if self.horizon_minutes < 0:
            raise ValueError("horizon_minutes must be non-negative")
        if self.precondition_margin_minutes < 0:
            raise ValueError("precondition_margin_minutes must be non-negative")
        if self.drift_safety_margin_minutes < 0:
            raise ValueError("drift_safety_margin_minutes must be non-negative")
        if self.min_precondition_gap_f < 0:
            raise ValueError("min_precondition_gap_f must be non-negative")
        if self.max_precondition_lead_minutes < 0:
            raise ValueError("max_precondition_lead_minutes must be non-negative")

        windows = [_coerce_window(window) for window in self.peak_windows]
        windows.sort(key=lambda window: window.start)
        for previous, current in zip(windows, windows[1:]):
            if current.start < previous.end:
                raise ValueError(f"peak windows must not overlap: {previous} and {current}")
        object.__setattr__(self, "peak_windows", windows)


@dataclass(frozen=True)
class ThermostatState:
    timestamp: pd.Timestamp
    indoor_temp_f: float
    outdoor_temp_f: float
    system_running: bool
    mode: Mode
    home_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", pd.Timestamp(self.timestamp))
        if self.mode not in ("heating", "cooling"):
            raise ValueError(f"unsupported mode: {self.mode!r}")
        if not np.isfinite(self.indoor_temp_f):
            raise ValueError("indoor_temp_f must be finite")
        if not np.isfinite(self.outdoor_temp_f):
            raise ValueError("outdoor_temp_f must be finite")


@dataclass(frozen=True)
class ThermostatCommand:
    heat_setpoint_f: float | None
    cool_setpoint_f: float | None
    phase: Phase
    reason: str


class SeriesWeatherForecast:
    def __init__(self, forecast: pd.Series, interpolate: bool = False) -> None:
        if forecast.empty:
            raise ValueError("forecast series must not be empty")
        series = forecast.copy()
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        self.forecast = series.astype(float)
        self.interpolate = interpolate

    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float:
        timestamp = pd.Timestamp(timestamp)
        if self.interpolate:
            series = self.forecast
            if timestamp < series.index[0] or timestamp > series.index[-1]:
                raise KeyError(timestamp)
            union_index = series.index.union(pd.DatetimeIndex([timestamp]))
            return float(series.reindex(union_index).interpolate(method="time").loc[timestamp])

        loc = self.forecast.index.searchsorted(timestamp, side="right") - 1
        if loc < 0:
            raise KeyError(timestamp)
        return float(self.forecast.iloc[loc])


class PeakAwareMPCController:
    def __init__(
        self,
        config: MPCConfig,
        predictor: FirstPassagePredictor,
        forecast: ForecastInput = None,
    ) -> None:
        self.config = config
        self.predictor = predictor
        self.forecast = forecast

    def decide(self, state: ThermostatState, forecast: ForecastInput = None) -> ThermostatCommand:
        _validate_plausible_temp(state.indoor_temp_f, "indoor_temp_f")
        window = _active_or_next_peak_window(
            state.timestamp,
            self.config.peak_windows,
            self.config.horizon_minutes,
        )

        if window is None:
            return self._normal_command(state, "no peak window in horizon")

        storage_target, drift_boundary, direction = _mode_targets(state.mode, self.config)
        active_forecast = self.forecast if forecast is None else forecast

        if window.contains(state.timestamp):
            minutes_to_peak_end = _minutes_between(state.timestamp, window.end)
            forecast_temp, forecast_reason = representative_outdoor_temp(
                active_forecast,
                start=state.timestamp,
                end=window.end,
                fallback=state.outdoor_temp_f,
            )
            drift_minutes = self.predictor.predict_drift_time(
                current_temp=state.indoor_temp_f,
                boundary_temp=drift_boundary,
                outdoor_temp=forecast_temp,
                timestamp=state.timestamp,
                home_id=state.home_id,
                direction=direction,
            )

            if _at_or_past_boundary(state.indoor_temp_f, drift_boundary, direction):
                return self._peak_maintain_command(
                    state,
                    f"comfort boundary reached during peak; {forecast_reason}",
                )
            if drift_minutes >= minutes_to_peak_end + self.config.drift_safety_margin_minutes:
                return self._peak_coast_command(
                    state,
                    f"predicted drift stays within comfort through peak; {forecast_reason}",
                )
            return self._peak_coast_command(
                state,
                f"defer HVAC until closer to comfort boundary; {forecast_reason}",
            )

        minutes_to_peak_start = _minutes_between(state.timestamp, window.start)
        if minutes_to_peak_start > self.config.max_precondition_lead_minutes:
            return self._normal_command(state, "peak window too far away")

        forecast_temp, forecast_reason = representative_outdoor_temp(
            active_forecast,
            start=state.timestamp,
            end=window.start,
            fallback=state.outdoor_temp_f,
        )
        active_minutes = self.predictor.predict_active_time(
            current_temp=state.indoor_temp_f,
            target_temp=storage_target,
            outdoor_temp=forecast_temp,
            timestamp=state.timestamp,
            system_running=state.system_running,
            home_id=state.home_id,
            direction=direction,
        )

        gap_to_storage_target = _directional_gap(state.indoor_temp_f, storage_target, direction)
        latest_start_minutes = active_minutes + self.config.precondition_margin_minutes

        if gap_to_storage_target < self.config.min_precondition_gap_f:
            return self._normal_command(
                state,
                f"already near storage target; {forecast_reason}",
            )
        if minutes_to_peak_start <= latest_start_minutes:
            return self._precondition_command(
                state,
                f"active-time prediction says preconditioning should start; {forecast_reason}",
            )
        return self._normal_command(
            state,
            f"not time to precondition yet; {forecast_reason}",
        )

    def _normal_command(self, state: ThermostatState, reason: str) -> ThermostatCommand:
        if state.mode == "heating":
            return ThermostatCommand(
                heat_setpoint_f=_clip_optional(self.config.normal_heat_setpoint_f, self.config),
                cool_setpoint_f=None,
                phase="normal",
                reason=reason,
            )
        return ThermostatCommand(
            heat_setpoint_f=None,
            cool_setpoint_f=_clip_optional(self.config.normal_cool_setpoint_f, self.config),
            phase="normal",
            reason=reason,
        )

    def _precondition_command(self, state: ThermostatState, reason: str) -> ThermostatCommand:
        if state.mode == "heating":
            return ThermostatCommand(
                heat_setpoint_f=self.config.comfort_upper_f,
                cool_setpoint_f=None,
                phase="precondition",
                reason=reason,
            )
        return ThermostatCommand(
            heat_setpoint_f=None,
            cool_setpoint_f=self.config.comfort_lower_f,
            phase="precondition",
            reason=reason,
        )

    def _peak_coast_command(self, state: ThermostatState, reason: str) -> ThermostatCommand:
        if state.mode == "heating":
            return ThermostatCommand(
                heat_setpoint_f=self.config.comfort_lower_f,
                cool_setpoint_f=None,
                phase="peak_coast",
                reason=reason,
            )
        return ThermostatCommand(
            heat_setpoint_f=None,
            cool_setpoint_f=self.config.comfort_upper_f,
            phase="peak_coast",
            reason=reason,
        )

    def _peak_maintain_command(self, state: ThermostatState, reason: str) -> ThermostatCommand:
        if state.mode == "heating":
            return ThermostatCommand(
                heat_setpoint_f=self.config.comfort_lower_f,
                cool_setpoint_f=None,
                phase="peak_maintain",
                reason=reason,
            )
        return ThermostatCommand(
            heat_setpoint_f=None,
            cool_setpoint_f=self.config.comfort_upper_f,
            phase="peak_maintain",
            reason=reason,
        )


def representative_outdoor_temp(
    forecast: ForecastInput,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fallback: float,
) -> tuple[float, str]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    fallback = float(fallback)
    if forecast is None:
        return fallback, "forecast missing, used current outdoor temperature"

    provider = _forecast_provider(forecast)
    timestamps = _sample_times(start, end)
    values: list[float] = []
    try:
        for timestamp in timestamps:
            value = provider.outdoor_temp_at(timestamp)
            if np.isfinite(value):
                values.append(float(value))
    except (KeyError, ValueError, TypeError, AttributeError):
        return fallback, "forecast lookup failed, used current outdoor temperature"

    if not values:
        return fallback, "forecast empty for interval, used current outdoor temperature"
    return float(np.mean(values)), "used forecast outdoor temperature"


def _forecast_provider(forecast: ForecastInput) -> WeatherForecast:
    if isinstance(forecast, pd.Series):
        return SeriesWeatherForecast(forecast)
    if callable(forecast):
        return _CallableWeatherForecast(forecast)
    if hasattr(forecast, "outdoor_temp_at"):
        return forecast
    raise TypeError("forecast must be None, callable, pd.Series, or WeatherForecast")


@dataclass(frozen=True)
class _CallableWeatherForecast:
    func: Callable[[pd.Timestamp], float]

    def outdoor_temp_at(self, timestamp: pd.Timestamp) -> float:
        return float(self.func(pd.Timestamp(timestamp)))


def _sample_times(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if end <= start:
        return [start]
    index = pd.date_range(start=start, end=end, freq="1h")
    if len(index) == 0 or index[0] != start:
        index = pd.DatetimeIndex([start]).append(index)
    if index[-1] != end:
        index = index.append(pd.DatetimeIndex([end]))
    return list(index.unique())


def _coerce_window(window: PeakWindow | tuple[pd.Timestamp, pd.Timestamp]) -> PeakWindow:
    if isinstance(window, PeakWindow):
        return window
    start, end = window
    return PeakWindow(start=start, end=end)


def _active_or_next_peak_window(
    timestamp: pd.Timestamp,
    windows: list[PeakWindow],
    horizon_minutes: int,
) -> PeakWindow | None:
    horizon_end = timestamp + pd.Timedelta(minutes=horizon_minutes)
    for window in windows:
        if window.contains(timestamp):
            return window
        if timestamp < window.start <= horizon_end:
            return window
    return None


def _mode_targets(mode: Mode, config: MPCConfig) -> tuple[float, float, Direction]:
    if mode == "heating":
        return config.comfort_upper_f, config.comfort_lower_f, "heating"
    return config.comfort_lower_f, config.comfort_upper_f, "cooling"


def _directional_gap(current_temp: float, target_temp: float, direction: Direction) -> float:
    if direction == "heating":
        return max(target_temp - current_temp, 0.0)
    return max(current_temp - target_temp, 0.0)


def _at_or_past_boundary(current_temp: float, boundary_temp: float, direction: Direction) -> bool:
    if direction == "heating":
        return current_temp <= boundary_temp
    return current_temp >= boundary_temp


def _clip_optional(value: float | None, config: MPCConfig) -> float | None:
    if value is None:
        return None
    return float(np.clip(value, config.comfort_lower_f, config.comfort_upper_f))


def _minutes_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60.0)


def _validate_plausible_temp(value: float, name: str) -> None:
    if not -50.0 <= value <= 150.0:
        raise ValueError(f"{name} is outside plausible Fahrenheit range: {value}")
