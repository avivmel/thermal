from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpc import MPCConfig
from mpc.controller import MPCController
from mpc.model_interfaces import FirstPassagePredictor


@dataclass
class MockFirstPassagePredictor(FirstPassagePredictor):
    heat_rate_f_per_hour: float = 2.0
    drift_rate_f_per_hour: float = 0.5
    mild_drift_rate_f_per_hour: float = 0.1
    min_minutes: float = 5.0

    def predict_heat_time(
        self,
        current_temp: float,
        target_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        system_running: bool,
        home_id: str | None = None,
    ) -> float:
        gap = max(target_temp - current_temp, 0.0)
        if gap <= 1e-9:
            return self.min_minutes

        thermal_penalty = max(current_temp - outdoor_temp, 0.0) / 50.0
        rate = self.heat_rate_f_per_hour * max(0.55, 1.0 - 0.35 * thermal_penalty)
        if system_running:
            rate *= 1.15
        return max(gap / max(rate, 0.2) * 60.0, self.min_minutes)

    def predict_drift_time(
        self,
        current_temp: float,
        boundary_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        home_id: str | None = None,
    ) -> float:
        margin = max(current_temp - boundary_temp, 0.0)
        if margin <= 1e-9:
            return self.min_minutes

        if outdoor_temp >= boundary_temp:
            rate = self.mild_drift_rate_f_per_hour
        else:
            drive = max(boundary_temp - outdoor_temp, 0.0)
            rate = self.drift_rate_f_per_hour * max(0.4, min(drive / 15.0, 2.0))
        return max(margin / max(rate, 0.05) * 60.0, self.min_minutes)


def build_forecast(
    start: str,
    outdoor_profile_f: list[float],
    timestep_minutes: int,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    start_ts = pd.Timestamp(start)
    timestamps = [
        start_ts + pd.Timedelta(minutes=timestep_minutes * step)
        for step in range(len(outdoor_profile_f))
    ]
    return timestamps, np.asarray(outdoor_profile_f, dtype=float)


def run_scenario(
    name: str,
    expectation: str,
    current_temp_f: float,
    hvac_mode: str,
    timestamps: list[pd.Timestamp],
    outdoor_profile_f: np.ndarray,
    lower_comfort_f: np.ndarray,
    predictor: FirstPassagePredictor,
    config: MPCConfig,
    hvac_power_kw: float = 8.0,
) -> dict:
    def forecast_provider(_: dict, __: int, ___: int) -> tuple[list[pd.Timestamp], np.ndarray]:
        return timestamps, outdoor_profile_f

    def schedule_provider(_: list[pd.Timestamp]) -> np.ndarray:
        return lower_comfort_f

    controller = MPCController(
        predictor=predictor,
        config=config,
        forecast_provider=forecast_provider,
        comfort_schedule_provider=schedule_provider,
        horizon_steps=len(timestamps),
        hvac_power_kw=hvac_power_kw,
    )

    obs = {
        "timestamp": timestamps[0],
        "indoor_temp": current_temp_f,
        "outdoor_temp": float(outdoor_profile_f[0]),
        "hvac_mode": hvac_mode,
    }
    decision = controller.plan(obs)

    return {
        "scenario": name,
        "expectation": expectation,
        "first_action_f": decision.action["heat_setpoint"],
        "first_6_actions_f": decision.solution.open_loop_actions_f[:6],
        "outdoor_start_f": float(outdoor_profile_f[0]),
        "comfort_start_f": float(lower_comfort_f[0]),
        "initial_temp_f": current_temp_f,
    }


def main() -> None:
    config = MPCConfig(
        timestep_minutes=5,
        action_min_f=66.0,
        action_max_f=74.0,
        action_bin_width_f=1.0,
        lambda_energy=12.0,
        lambda_comfort=400.0,
        terminal_penalty=1500.0,
    )
    predictor = MockFirstPassagePredictor()

    scenarios: list[dict] = []

    timestamps, outdoor = build_forecast(
        start="2017-01-15 08:00:00",
        outdoor_profile_f=[30.0] * 24,
        timestep_minutes=config.timestep_minutes,
    )
    lower_comfort = np.array([68.0] * 24, dtype=float)
    scenarios.append(
        run_scenario(
            name="already_warm_home",
            expectation="coast at lower comfort setpoint",
            current_temp_f=71.0,
            hvac_mode="off",
            timestamps=timestamps,
            outdoor_profile_f=outdoor,
            lower_comfort_f=lower_comfort,
            predictor=predictor,
            config=config,
        )
    )

    timestamps, outdoor = build_forecast(
        start="2017-01-15 15:00:00",
        outdoor_profile_f=[34.0] * 12 + [28.0] * 12,
        timestep_minutes=config.timestep_minutes,
    )
    lower_comfort = np.array([66.0] * 12 + [68.0] * 12, dtype=float)
    scenarios.append(
        run_scenario(
            name="future_comfort_increase",
            expectation="wait at 66F, then heat when comfort floor rises",
            current_temp_f=66.2,
            hvac_mode="off",
            timestamps=timestamps,
            outdoor_profile_f=outdoor,
            lower_comfort_f=lower_comfort,
            predictor=predictor,
            config=config,
        )
    )

    timestamps, outdoor = build_forecast(
        start="2017-01-15 17:30:00",
        outdoor_profile_f=[24.0] * 18,
        timestep_minutes=config.timestep_minutes,
    )
    lower_comfort = np.array([68.0] * 18, dtype=float)
    scenarios.append(
        run_scenario(
            name="cold_start_near_deadline",
            expectation="heat immediately to current comfort floor",
            current_temp_f=66.4,
            hvac_mode="off",
            timestamps=timestamps,
            outdoor_profile_f=outdoor,
            lower_comfort_f=lower_comfort,
            predictor=predictor,
            config=config,
        )
    )

    mild_predictor = MockFirstPassagePredictor(
        heat_rate_f_per_hour=2.0,
        drift_rate_f_per_hour=0.25,
        mild_drift_rate_f_per_hour=0.05,
    )
    timestamps, outdoor = build_forecast(
        start="2017-03-15 11:00:00",
        outdoor_profile_f=[67.0] * 24,
        timestep_minutes=config.timestep_minutes,
    )
    lower_comfort = np.array([68.0] * 24, dtype=float)
    scenarios.append(
        run_scenario(
            name="mild_day_should_coast",
            expectation="coast because drift toward boundary is slow",
            current_temp_f=69.5,
            hvac_mode="off",
            timestamps=timestamps,
            outdoor_profile_f=outdoor,
            lower_comfort_f=lower_comfort,
            predictor=mild_predictor,
            config=config,
        )
    )

    df = pd.DataFrame(scenarios)
    df["passes_smell_test"] = [
        row["scenario"] == "already_warm_home" and row["first_action_f"] == 68.0
        or row["scenario"] == "future_comfort_increase" and row["first_action_f"] == 66.0
        or row["scenario"] == "cold_start_near_deadline" and row["first_action_f"] == 68.0
        or row["scenario"] == "mild_day_should_coast" and row["first_action_f"] == 68.0
        for _, row in df.iterrows()
    ]
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
