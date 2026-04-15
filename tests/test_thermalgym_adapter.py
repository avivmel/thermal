from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mpc.peak_mpc import (
    MPCConfig,
    PeakAwareMPCController,
    ThermostatCommand,
)
from mpc.thermalgym_adapter import ThermalGymMPCAdapter
from thermalgym.env import COOL_MAX, HEAT_MIN


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


def obs(**kwargs) -> dict:
    params = {
        "timestamp": pd.Timestamp("2017-01-15 16:00"),
        "indoor_temp": 69.0,
        "outdoor_temp": 25.0,
        "hvac_mode": "off",
    }
    params.update(kwargs)
    return params


def config() -> MPCConfig:
    return MPCConfig(
        peak_windows=[(pd.Timestamp("2017-01-15 17:00"), pd.Timestamp("2017-01-15 20:00"))],
        comfort_lower_f=68.0,
        comfort_upper_f=72.0,
        normal_heat_setpoint_f=69.0,
        normal_cool_setpoint_f=75.0,
    )


def test_state_from_obs_uses_harness_mode_and_building_id() -> None:
    adapter = ThermalGymMPCAdapter(mode="heating", building_id="medium_cold_heatpump")

    state = adapter.state_from_obs(obs(hvac_mode="cooling"))

    assert state.timestamp == pd.Timestamp("2017-01-15 16:00")
    assert state.indoor_temp_f == 69.0
    assert state.outdoor_temp_f == 25.0
    assert state.system_running is True
    assert state.mode == "heating"
    assert state.home_id == "medium_cold_heatpump"


def test_state_from_obs_treats_off_hvac_as_not_running() -> None:
    adapter = ThermalGymMPCAdapter(mode="cooling", building_id="small_hot_heatpump")

    state = adapter.state_from_obs(obs(hvac_mode="off"))

    assert state.system_running is False
    assert state.mode == "cooling"
    assert state.home_id == "small_hot_heatpump"


def test_action_from_heating_command_sets_cooling_to_pass_through() -> None:
    adapter = ThermalGymMPCAdapter(mode="heating")
    command = ThermostatCommand(
        heat_setpoint_f=72.0,
        cool_setpoint_f=None,
        phase="precondition",
        reason="test",
    )

    action = adapter.action_from_command(command)

    assert action == {"heat_setpoint": 72.0, "cool_setpoint": COOL_MAX}


def test_action_from_cooling_command_sets_heating_to_pass_through() -> None:
    adapter = ThermalGymMPCAdapter(mode="cooling")
    command = ThermostatCommand(
        heat_setpoint_f=None,
        cool_setpoint_f=68.0,
        phase="precondition",
        reason="test",
    )

    action = adapter.action_from_command(command)

    assert action == {"heat_setpoint": HEAT_MIN, "cool_setpoint": 68.0}


def test_adapter_drives_controller_and_returns_thermalgym_action() -> None:
    predictor = FakePredictor(active_minutes=50.0)
    controller = PeakAwareMPCController(config(), predictor)
    adapter = ThermalGymMPCAdapter(mode="heating", building_id="medium_cold_heatpump")

    action = adapter.action_for_obs(controller, obs())

    assert action == {"heat_setpoint": 72.0, "cool_setpoint": COOL_MAX}
    assert predictor.active_calls[-1]["home_id"] == "medium_cold_heatpump"
    assert predictor.active_calls[-1]["system_running"] is False


def test_for_building_accepts_building_like_object() -> None:
    @dataclass(frozen=True)
    class BuildingLike:
        id: str

    adapter = ThermalGymMPCAdapter.for_building(BuildingLike("large_hot_ac"), mode="cooling")

    assert adapter.building_id == "large_hot_ac"
    assert adapter.mode == "cooling"

