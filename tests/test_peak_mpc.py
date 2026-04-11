from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from mpc.peak_mpc import (
    MPCConfig,
    PeakAwareMPCController,
    PeakWindow,
    SeriesWeatherForecast,
    ThermostatState,
)


@dataclass
class FakePredictor:
    active_minutes: float = 30.0
    drift_minutes: float = 120.0
    active_calls: list[dict] = field(default_factory=list)
    drift_calls: list[dict] = field(default_factory=list)

    def predict_active_time(self, **kwargs) -> float:
        self.active_calls.append(kwargs)
        return self.active_minutes

    def predict_drift_time(self, **kwargs) -> float:
        self.drift_calls.append(kwargs)
        return self.drift_minutes


def ts(hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(2026, 1, 1, hour, minute)


def config(*windows, **kwargs) -> MPCConfig:
    params = {
        "peak_windows": list(windows),
        "comfort_lower_f": 68.0,
        "comfort_upper_f": 72.0,
        "normal_heat_setpoint_f": 70.0,
        "normal_cool_setpoint_f": 74.0,
    }
    params.update(kwargs)
    return MPCConfig(**params)


def state(hour: int, minute: int = 0, **kwargs) -> ThermostatState:
    params = {
        "timestamp": ts(hour, minute),
        "indoor_temp_f": 70.0,
        "outdoor_temp_f": 30.0,
        "system_running": False,
        "mode": "heating",
        "home_id": "home-a",
    }
    params.update(kwargs)
    return ThermostatState(**params)


def test_no_peak_windows_returns_normal() -> None:
    controller = PeakAwareMPCController(config(), FakePredictor())

    command = controller.decide(state(12))

    assert command.phase == "normal"
    assert command.heat_setpoint_f == 70.0


def test_future_peak_outside_horizon_returns_normal() -> None:
    controller = PeakAwareMPCController(
        config((ts(17), ts(20)), horizon_minutes=60),
        FakePredictor(),
    )

    command = controller.decide(state(12))

    assert command.phase == "normal"


def test_future_peak_before_latest_start_returns_normal() -> None:
    predictor = FakePredictor(active_minutes=30.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(15))

    assert command.phase == "normal"
    assert predictor.active_calls


def test_future_peak_at_latest_start_returns_precondition() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(16))

    assert command.phase == "precondition"
    assert command.heat_setpoint_f == 72.0


def test_heating_precondition_uses_upper_comfort_bound() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(16, indoor_temp_f=69.0, mode="heating"))

    assert command.phase == "precondition"
    assert command.heat_setpoint_f == 72.0
    assert command.cool_setpoint_f is None


def test_cooling_precondition_uses_lower_comfort_bound() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    controller = PeakAwareMPCController(
        config((ts(17), ts(20)), normal_cool_setpoint_f=74.0),
        predictor,
    )

    command = controller.decide(state(16, indoor_temp_f=75.0, mode="cooling"))

    assert command.phase == "precondition"
    assert command.cool_setpoint_f == 68.0
    assert command.heat_setpoint_f is None


def test_in_peak_long_drift_returns_peak_coast() -> None:
    predictor = FakePredictor(drift_minutes=240.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(18, indoor_temp_f=70.0))

    assert command.phase == "peak_coast"
    assert command.heat_setpoint_f == 68.0


def test_in_peak_short_drift_defers_until_boundary() -> None:
    predictor = FakePredictor(drift_minutes=10.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(18, indoor_temp_f=69.0))

    assert command.phase == "peak_coast"
    assert "defer HVAC" in command.reason


def test_in_peak_at_boundary_returns_peak_maintain() -> None:
    predictor = FakePredictor(drift_minutes=10.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(18, indoor_temp_f=68.0))

    assert command.phase == "peak_maintain"
    assert command.heat_setpoint_f == 68.0


def test_in_peak_at_heating_boundary_maintains_even_when_drift_prediction_is_long() -> None:
    predictor = FakePredictor(drift_minutes=240.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(18, indoor_temp_f=68.0, mode="heating"))

    assert command.phase == "peak_maintain"
    assert command.heat_setpoint_f == 68.0


def test_in_peak_at_cooling_boundary_maintains_even_when_drift_prediction_is_long() -> None:
    predictor = FakePredictor(drift_minutes=240.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(18, indoor_temp_f=72.0, mode="cooling"))

    assert command.phase == "peak_maintain"
    assert command.cool_setpoint_f == 72.0


def test_forecast_provider_value_is_passed_into_active_prediction() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    forecast = pd.Series(
        [40.0, 44.0],
        index=[ts(16), ts(17)],
    )
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor, forecast)

    controller.decide(state(16, outdoor_temp_f=30.0))

    assert predictor.active_calls[-1]["outdoor_temp"] == pytest.approx(42.0)


def test_forecast_provider_value_is_passed_into_drift_prediction() -> None:
    predictor = FakePredictor(drift_minutes=240.0)
    forecast = pd.Series(
        [40.0, 42.0, 46.0],
        index=[ts(18), ts(19), ts(20)],
    )
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor, forecast)

    controller.decide(state(18, outdoor_temp_f=30.0))

    assert predictor.drift_calls[-1]["outdoor_temp"] == pytest.approx(42.6666667)


def test_missing_forecast_falls_back_to_current_outdoor_temperature() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    forecast = SeriesWeatherForecast(pd.Series([40.0], index=[ts(18)]))
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor, forecast)

    command = controller.decide(state(16, outdoor_temp_f=31.0))

    assert predictor.active_calls[-1]["outdoor_temp"] == 31.0
    assert "used current outdoor temperature" in command.reason


def test_multiple_peak_windows_choose_active_window() -> None:
    predictor = FakePredictor(drift_minutes=240.0)
    controller = PeakAwareMPCController(
        config((ts(13), ts(14)), (ts(17), ts(20))),
        predictor,
    )

    command = controller.decide(state(13, 30))

    assert command.phase == "peak_coast"
    assert predictor.drift_calls
    assert not predictor.active_calls


def test_multiple_peak_windows_choose_next_future_window() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    controller = PeakAwareMPCController(
        config((ts(13), ts(14)), (ts(17), ts(20))),
        predictor,
    )

    command = controller.decide(state(16))

    assert command.phase == "precondition"
    assert predictor.active_calls


def test_overlapping_windows_raise_value_error() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        config((ts(13), ts(15)), (ts(14), ts(16)))


def test_invalid_comfort_bounds_raise_value_error() -> None:
    with pytest.raises(ValueError, match="comfort_lower_f"):
        config(comfort_lower_f=72.0, comfort_upper_f=68.0)


def test_already_near_storage_target_returns_normal() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    controller = PeakAwareMPCController(config((ts(17), ts(20))), predictor)

    command = controller.decide(state(16, indoor_temp_f=71.75))

    assert command.phase == "normal"
    assert "already near storage target" in command.reason
