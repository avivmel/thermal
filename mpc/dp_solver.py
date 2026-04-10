from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc.problem import MPCConfig, MPCInputs, snap_to_grid
from mpc.transitions import TransitionBundle


@dataclass(frozen=True)
class DpSolution:
    values: np.ndarray
    action_indices: np.ndarray
    first_action_index: int
    first_action_temp_f: float
    state_temperature_bins: np.ndarray
    open_loop_actions_f: list[float]


def solve_finite_horizon_dp(
    config: MPCConfig,
    inputs: MPCInputs,
    transitions: TransitionBundle,
) -> DpSolution:
    temp_grid = config.temperature_grid
    horizon = inputs.horizon_steps
    n_temp = len(temp_grid)
    n_actions = len(transitions.action_grid)

    values = np.full((horizon + 1, n_temp, 2), np.inf, dtype=float)
    action_indices = np.zeros((horizon, n_temp, 2), dtype=np.int16)

    final_lower_comfort = inputs.lower_comfort_at(horizon - 1)
    terminal_violation = np.maximum(final_lower_comfort - temp_grid - config.comfort_slack_f, 0.0)
    values[horizon, :, :] = config.terminal_penalty * (terminal_violation[:, None] ** 2)

    for t in range(horizon - 1, -1, -1):
        for temp_idx in range(n_temp):
            for running_flag in (0, 1):
                best_value = np.inf
                best_action = 0

                for action_idx in range(n_actions):
                    if not transitions.feasible[t, temp_idx, running_flag, action_idx]:
                        continue

                    next_temp_idx = transitions.next_temp_idx[t, temp_idx, running_flag, action_idx]
                    next_running = transitions.next_running_flag[t, temp_idx, running_flag, action_idx]
                    energy = transitions.energy_kwh[t, temp_idx, running_flag, action_idx]
                    comfort_violation = transitions.comfort_violation_f[t, temp_idx, running_flag, action_idx]
                    stage_cost = (
                        config.lambda_energy * (energy ** 2)
                        + config.lambda_comfort * (comfort_violation ** 2)
                    )
                    total_cost = stage_cost + values[t + 1, next_temp_idx, next_running]
                    if total_cost < best_value:
                        best_value = total_cost
                        best_action = action_idx

                values[t, temp_idx, running_flag] = best_value
                action_indices[t, temp_idx, running_flag] = best_action

    initial_temp_idx = snap_to_grid(inputs.initial_indoor_temp_f, temp_grid)
    initial_running_flag = 1 if inputs.initial_running else 0
    first_action_index = int(action_indices[0, initial_temp_idx, initial_running_flag])

    open_loop_actions_f: list[float] = []
    curr_temp_idx = initial_temp_idx
    curr_running = initial_running_flag
    for t in range(horizon):
        action_idx = int(action_indices[t, curr_temp_idx, curr_running])
        open_loop_actions_f.append(float(transitions.action_grid[action_idx]))
        next_temp_idx = int(transitions.next_temp_idx[t, curr_temp_idx, curr_running, action_idx])
        next_running = int(transitions.next_running_flag[t, curr_temp_idx, curr_running, action_idx])
        curr_temp_idx = next_temp_idx
        curr_running = next_running

    return DpSolution(
        values=values,
        action_indices=action_indices,
        first_action_index=first_action_index,
        first_action_temp_f=float(transitions.action_grid[first_action_index]),
        state_temperature_bins=temp_grid,
        open_loop_actions_f=open_loop_actions_f,
    )
