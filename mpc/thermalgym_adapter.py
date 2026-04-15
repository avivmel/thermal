from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mpc.peak_mpc import (
    ForecastInput,
    Mode,
    PeakAwareMPCController,
    ThermostatCommand,
    ThermostatState,
)
from thermalgym.env import COOL_MAX, HEAT_MIN


@dataclass(frozen=True)
class ThermalGymMPCAdapter:
    """Translate between ThermalGym observations/actions and the MPC API."""

    mode: Mode
    building_id: str | None = None
    inactive_heat_setpoint_f: float = HEAT_MIN
    inactive_cool_setpoint_f: float = COOL_MAX

    def __post_init__(self) -> None:
        if self.mode not in ("heating", "cooling"):
            raise ValueError(f"unsupported mode: {self.mode!r}")

    @classmethod
    def for_building(cls, building: Any, mode: Mode) -> ThermalGymMPCAdapter:
        building_id = building if isinstance(building, str) else getattr(building, "id", None)
        if building_id is not None:
            building_id = str(building_id)
        return cls(mode=mode, building_id=building_id)

    def state_from_obs(self, obs: dict[str, Any]) -> ThermostatState:
        return ThermostatState(
            timestamp=pd.Timestamp(obs["timestamp"]),
            indoor_temp_f=float(obs["indoor_temp"]),
            outdoor_temp_f=float(obs["outdoor_temp"]),
            system_running=obs.get("hvac_mode") != "off",
            mode=self.mode,
            home_id=self.building_id,
        )

    def action_from_command(self, command: ThermostatCommand) -> dict[str, float]:
        heat_setpoint = (
            self.inactive_heat_setpoint_f
            if command.heat_setpoint_f is None
            else float(command.heat_setpoint_f)
        )
        cool_setpoint = (
            self.inactive_cool_setpoint_f
            if command.cool_setpoint_f is None
            else float(command.cool_setpoint_f)
        )
        return {
            "heat_setpoint": heat_setpoint,
            "cool_setpoint": cool_setpoint,
        }

    def action_for_obs(
        self,
        controller: PeakAwareMPCController,
        obs: dict[str, Any],
        forecast: ForecastInput = None,
    ) -> dict[str, float]:
        state = self.state_from_obs(obs)
        command = controller.decide(state, forecast=forecast)
        return self.action_from_command(command)

    def __call__(
        self,
        obs: dict[str, Any],
        controller: PeakAwareMPCController,
        forecast: ForecastInput = None,
    ) -> dict[str, float]:
        return self.action_for_obs(controller, obs, forecast=forecast)

