from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc.model_interfaces import FirstPassagePredictor
from mpc.problem import MPCConfig, MPCInputs


@dataclass(frozen=True)
class TransitionBundle:
    next_temp_idx: np.ndarray
    next_running_flag: np.ndarray
    energy_kwh: np.ndarray
    comfort_violation_f: np.ndarray
    feasible: np.ndarray
    runtime_minutes: np.ndarray
    action_grid: np.ndarray


def build_transition_bundle(
    config: MPCConfig,
    inputs: MPCInputs,
    predictor: FirstPassagePredictor,
) -> TransitionBundle:
    temp_grid = config.temperature_grid
    horizon = inputs.horizon_steps
    n_temp = len(temp_grid)
    n_actions = len(config.action_grid)

    next_temp_idx = np.zeros((horizon, n_temp, 2, n_actions), dtype=np.int16)
    next_running_flag = np.zeros((horizon, n_temp, 2, n_actions), dtype=np.int8)
    energy_kwh = np.zeros((horizon, n_temp, 2, n_actions), dtype=float)
    comfort_violation_f = np.zeros((horizon, n_temp, 2, n_actions), dtype=float)
    feasible = np.ones((horizon, n_temp, 2, n_actions), dtype=bool)
    runtime_minutes = np.zeros((horizon, n_temp, 2, n_actions), dtype=float)

    dt = float(config.timestep_minutes)
    action_grid = config.action_grid

    for t in range(horizon):
        ts = inputs.timestamp_at(t)
        outdoor = inputs.outdoor_at(t)
        lower_comfort = inputs.lower_comfort_at(t)

        for temp_idx, current_temp in enumerate(temp_grid):
            for running_flag in (0, 1):
                for action_idx, action_temp in enumerate(action_grid):
                    if action_temp < lower_comfort - 1e-6:
                        feasible[t, temp_idx, running_flag, action_idx] = False
                        comfort_violation_f[t, temp_idx, running_flag, action_idx] = max(
                            lower_comfort - current_temp,
                            0.0,
                        )
                        next_temp_idx[t, temp_idx, running_flag, action_idx] = temp_idx
                        continue

                    if action_temp <= current_temp + config.target_reached_epsilon_f:
                        tau = predictor.predict_drift_time(
                            current_temp=current_temp,
                            boundary_temp=lower_comfort,
                            outdoor_temp=outdoor,
                            timestamp=ts,
                        )
                        progress = min(dt / max(tau, 1e-6), 1.0)
                        next_temp = current_temp + progress * (lower_comfort - current_temp)
                        runtime = 0.0
                        next_running = 1 if next_temp <= lower_comfort + config.target_reached_epsilon_f else 0
                    else:
                        tau = predictor.predict_heat_time(
                            current_temp=current_temp,
                            target_temp=action_temp,
                            outdoor_temp=outdoor,
                            timestamp=ts,
                            system_running=bool(running_flag),
                        )
                        progress = min(dt / max(tau, 1e-6), 1.0)
                        next_temp = current_temp + progress * (action_temp - current_temp)
                        runtime = min(tau, dt)
                        next_running = 1 if next_temp < action_temp - config.target_reached_epsilon_f else 0

                    snapped_idx = int(np.abs(temp_grid - next_temp).argmin())
                    snapped_temp = float(temp_grid[snapped_idx])
                    violation = max(lower_comfort - snapped_temp - config.comfort_slack_f, 0.0)

                    next_temp_idx[t, temp_idx, running_flag, action_idx] = snapped_idx
                    next_running_flag[t, temp_idx, running_flag, action_idx] = next_running
                    runtime_minutes[t, temp_idx, running_flag, action_idx] = runtime
                    energy_kwh[t, temp_idx, running_flag, action_idx] = inputs.hvac_power_kw * runtime / 60.0
                    comfort_violation_f[t, temp_idx, running_flag, action_idx] = violation

    return TransitionBundle(
        next_temp_idx=next_temp_idx,
        next_running_flag=next_running_flag,
        energy_kwh=energy_kwh,
        comfort_violation_f=comfort_violation_f,
        feasible=feasible,
        runtime_minutes=runtime_minutes,
        action_grid=action_grid,
    )
