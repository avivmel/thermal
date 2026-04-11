from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mpc.dp_solver import DpSolution, solve_finite_horizon_dp
from mpc.model_interfaces import FirstPassagePredictor
from mpc.problem import MPCConfig, MPCInputs, build_daily_schedule
from mpc.transitions import TransitionBundle, build_transition_bundle, slice_transition_bundle


ForecastProvider = Callable[[dict, int, int], tuple[list[pd.Timestamp], np.ndarray]]
ScheduleProvider = Callable[[list[pd.Timestamp]], np.ndarray]


@dataclass(frozen=True)
class MPCDecision:
    action: dict
    solution: DpSolution
    transitions: TransitionBundle
    inputs: MPCInputs


@dataclass
class _TransitionCacheEntry:
    timestamps_ns: np.ndarray
    outdoor_temps_f: np.ndarray
    lower_comfort_f: np.ndarray
    bundle: TransitionBundle


class MPCController:
    def __init__(
        self,
        predictor: FirstPassagePredictor,
        config: MPCConfig | None = None,
        forecast_provider: ForecastProvider | None = None,
        comfort_schedule_provider: ScheduleProvider | None = None,
        horizon_steps: int | None = None,
        hvac_power_kw: float = 8.0,
    ) -> None:
        self.predictor = predictor
        self.config = config or MPCConfig()
        self.forecast_provider = forecast_provider or persistence_forecast
        self.comfort_schedule_provider = comfort_schedule_provider or default_comfort_schedule
        self.horizon_steps = horizon_steps or (24 * 60 // self.config.timestep_minutes)
        self.hvac_power_kw = hvac_power_kw
        self.last_decision: MPCDecision | None = None
        self._transition_cache: _TransitionCacheEntry | None = None

    def plan(self, obs: dict) -> MPCDecision:
        timestamps, outdoor_forecast = self.forecast_provider(
            obs,
            self.horizon_steps,
            self.config.timestep_minutes,
        )
        lower_comfort = np.asarray(self.comfort_schedule_provider(timestamps), dtype=float)
        inputs = MPCInputs(
            timestamps=timestamps,
            outdoor_temps_f=outdoor_forecast,
            lower_comfort_f=lower_comfort,
            hvac_power_kw=self.hvac_power_kw,
            initial_indoor_temp_f=float(obs["indoor_temp"]),
            initial_running=obs.get("hvac_mode") == "heating",
            initial_timestamp=pd.Timestamp(obs["timestamp"]),
        )
        transitions = self._get_transition_bundle(inputs)
        solution = solve_finite_horizon_dp(self.config, inputs, transitions)
        action = {
            "heat_setpoint": solution.first_action_temp_f,
            "cool_setpoint": self.config.cooling_setpoint_f,
        }
        decision = MPCDecision(action=action, solution=solution, transitions=transitions, inputs=inputs)
        self.last_decision = decision
        return decision

    def __call__(self, obs: dict) -> dict:
        return self.plan(obs).action

    def _get_transition_bundle(self, inputs: MPCInputs) -> TransitionBundle:
        timestamps_ns = np.array([ts.value for ts in map(pd.Timestamp, inputs.timestamps)], dtype=np.int64)
        outdoor = np.asarray(inputs.outdoor_temps_f, dtype=float)
        lower_comfort = np.asarray(inputs.lower_comfort_f, dtype=float)

        cached = self._transition_cache
        if cached is not None:
            offset = len(cached.timestamps_ns) - len(timestamps_ns)
            if offset >= 0:
                if (
                    np.array_equal(cached.timestamps_ns[offset:], timestamps_ns)
                    and np.array_equal(cached.outdoor_temps_f[offset:], outdoor)
                    and np.array_equal(cached.lower_comfort_f[offset:], lower_comfort)
                ):
                    bundle = slice_transition_bundle(cached.bundle, offset)
                    self._transition_cache = _TransitionCacheEntry(
                        timestamps_ns=timestamps_ns,
                        outdoor_temps_f=outdoor,
                        lower_comfort_f=lower_comfort,
                        bundle=bundle,
                    )
                    return bundle

        bundle = build_transition_bundle(self.config, inputs, self.predictor)
        self._transition_cache = _TransitionCacheEntry(
            timestamps_ns=timestamps_ns,
            outdoor_temps_f=outdoor,
            lower_comfort_f=lower_comfort,
            bundle=bundle,
        )
        return bundle


def persistence_forecast(
    obs: dict,
    horizon_steps: int,
    timestep_minutes: int,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    start = pd.Timestamp(obs["timestamp"])
    timestamps = [
        start + pd.Timedelta(minutes=timestep_minutes * offset)
        for offset in range(horizon_steps)
    ]
    outdoor = np.full(horizon_steps, float(obs["outdoor_temp"]), dtype=float)
    return timestamps, outdoor


def default_comfort_schedule(timestamps: list[pd.Timestamp]) -> np.ndarray:
    if not timestamps:
        return np.array([], dtype=float)
    default_heat = 68.0
    peak_heat = 66.0
    return np.array(
        [
            peak_heat if 17 <= ts.hour < 20 else default_heat
            for ts in timestamps
        ],
        dtype=float,
    )


def make_daily_heating_mpc(
    predictor: FirstPassagePredictor,
    hvac_power_kw: float,
    config: MPCConfig | None = None,
    peak_start_hour: int = 17,
    peak_end_hour: int = 20,
    default_heat_f: float = 68.0,
    peak_heat_f: float = 66.0,
    outdoor_quantization_f: float = 2.0,
) -> MPCController:
    config = config or MPCConfig()
    def forecast_provider(
        obs: dict,
        _: int,
        __: int,
    ) -> tuple[list[pd.Timestamp], np.ndarray]:
        start_timestamp = pd.Timestamp(obs["timestamp"])
        end_of_day = start_timestamp.normalize() + pd.Timedelta(days=1)
        minutes_remaining = max((end_of_day - start_timestamp).total_seconds() / 60.0, config.timestep_minutes)
        horizon_steps = int(np.ceil(minutes_remaining / config.timestep_minutes))
        timestamps = [
            start_timestamp + pd.Timedelta(minutes=config.timestep_minutes * step)
            for step in range(horizon_steps)
        ]
        outdoor_temp = float(obs["outdoor_temp"])
        if outdoor_quantization_f > 0:
            outdoor_temp = round(outdoor_temp / outdoor_quantization_f) * outdoor_quantization_f
        return timestamps, np.full(horizon_steps, outdoor_temp, dtype=float)

    def schedule_provider(timestamps: list[pd.Timestamp]) -> np.ndarray:
        if not timestamps:
            return np.array([], dtype=float)
        _, lower_comfort = build_daily_schedule(
            start=timestamps[0],
            horizon_steps=len(timestamps),
            timestep_minutes=config.timestep_minutes,
            default_heat_f=default_heat_f,
            peak_heat_f=peak_heat_f,
            peak_start_hour=peak_start_hour,
            peak_end_hour=peak_end_hour,
        )
        return lower_comfort

    return MPCController(
        predictor=predictor,
        config=config,
        forecast_provider=forecast_provider,
        comfort_schedule_provider=schedule_provider,
        horizon_steps=24 * 60 // config.timestep_minutes,
        hvac_power_kw=hvac_power_kw,
    )
