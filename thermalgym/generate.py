from __future__ import annotations

import multiprocessing
import random
from pathlib import Path
from typing import Callable, Optional, Union
import numpy as np
import pandas as pd

from thermalgym.env import ThermalEnv, HEAT_MIN, HEAT_MAX, COOL_MIN, COOL_MAX, _CA_TOU_PRICES
from thermalgym.buildings import BUILDINGS, get_building, Building
from thermalgym.policies import Baseline, PreCool, Setback, PriceResponse


def generate_episodes(
    output: Union[str, Path],
    n_episodes: int = 1000,
    buildings: Union[str, list[str]] = "all",
    modes: list[str] = ["heat_increase", "cool_decrease"],
    gap_range: tuple[float, float] = (1.0, 5.0),
    outdoor_temp_range: Optional[tuple[float, float]] = None,
    timestep_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate setpoint-response training episodes.

    Args:
        output: Path to write Parquet file.
        n_episodes: Total episodes across all buildings (distributed proportionally).
        buildings: "all" or list of building IDs.
        modes: ["heat_increase"], ["cool_decrease"], or both.
        gap_range: (min_gap, max_gap) in °F for setpoint-to-temp gap.
        outdoor_temp_range: Filter episodes by outdoor temp. None = no filter.
        timestep_minutes: Simulation resolution.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with setpoint_responses schema. Also writes Parquet to output.
    """
    output = Path(output)
    building_ids = list(BUILDINGS.keys()) if buildings == "all" else list(buildings)

    if not building_ids:
        raise ValueError("No buildings specified.")
    for bid in building_ids:
        if bid not in BUILDINGS:
            from thermalgym.buildings import get_building
            get_building(bid)  # raises KeyError with helpful message

    # Distribute episodes proportionally across buildings
    n_buildings = len(building_ids)
    base_count = n_episodes // n_buildings
    remainder = n_episodes % n_buildings
    counts = [base_count + (1 if i < remainder else 0) for i in range(n_buildings)]

    # Build per-worker args
    ctx = multiprocessing.get_context("spawn")
    rng = random.Random(seed)
    worker_args = [
        (bid, cnt, rng.randint(0, 2**31), modes, gap_range, outdoor_temp_range, timestep_minutes)
        for bid, cnt in zip(building_ids, counts)
        if cnt > 0
    ]

    all_frames: list[pd.DataFrame] = []
    with ctx.Pool(processes=min(len(worker_args), multiprocessing.cpu_count())) as pool:
        results = pool.starmap(_run_building_episodes, worker_args)
    for df in results:
        if df is not None and len(df) > 0:
            all_frames.append(df)

    if not all_frames:
        combined = pd.DataFrame(columns=[
            "home_id", "episode_id", "mode", "timestep", "timestamp",
            "indoor_temp", "outdoor_temp", "setpoint", "hvac_power_kw",
            "hvac_on", "hour", "month", "time_to_target_min",
        ])
    else:
        combined = pd.concat(all_frames, ignore_index=True)
        # Assign global monotonically-increasing episode_ids
        episode_map = {old: new for new, old in enumerate(combined["episode_id"].unique())}
        combined["episode_id"] = combined["episode_id"].map(episode_map)

    combined.to_parquet(output, index=False)
    return combined


def _run_building_episodes(
    building_id: str,
    n_episodes: int,
    seed: int,
    modes: list[str],
    gap_range: tuple[float, float],
    outdoor_temp_range: Optional[tuple[float, float]],
    timestep_minutes: int,
) -> pd.DataFrame:
    """Worker function: generate episodes for one building."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    episode_id = 0

    try:
        env = ThermalEnv(building=building_id, timestep_minutes=timestep_minutes)
    except (FileNotFoundError, ImportError):
        # IDF/EPW not available or EnergyPlus not installed — return empty
        return pd.DataFrame()

    attempts = 0
    max_attempts = n_episodes * 10

    while episode_id < n_episodes and attempts < max_attempts:
        attempts += 1

        # Random date in 2017 (Jan 1 - Nov 30 to allow run_period_days buffer)
        day_of_year = int(rng.integers(1, 335))
        date = (pd.Timestamp("2017-01-01") + pd.Timedelta(days=day_of_year - 1)).strftime("%Y-%m-%d")

        mode = modes[int(rng.integers(0, len(modes)))]

        try:
            obs = env.reset(date=date)
        except Exception:
            continue

        # Check outdoor temp filter
        if outdoor_temp_range is not None:
            t_out = obs["outdoor_temp"]
            if not (outdoor_temp_range[0] <= t_out <= outdoor_temp_range[1]):
                continue

        indoor_temp = obs["indoor_temp"]
        gap = float(rng.uniform(gap_range[0], gap_range[1]))

        if mode == "heat_increase":
            new_setpoint = float(np.clip(indoor_temp + gap, HEAT_MIN, HEAT_MAX))
            target_reached = lambda obs: obs["indoor_temp"] >= new_setpoint
            active_setpoint = new_setpoint
            action_base = {"heat_setpoint": new_setpoint, "cool_setpoint": 76.0}
        else:  # cool_decrease
            new_setpoint = float(np.clip(indoor_temp - gap, COOL_MIN, COOL_MAX))
            target_reached = lambda obs: obs["indoor_temp"] <= new_setpoint
            active_setpoint = new_setpoint
            action_base = {"heat_setpoint": 68.0, "cool_setpoint": new_setpoint}

        max_steps = int(4 * 60 / timestep_minutes)  # 4-hour timeout
        ep_rows: list[dict] = []
        reached = False
        reach_time_min: Optional[float] = None

        for step_idx in range(max_steps):
            ep_rows.append({
                "home_id": building_id,
                "episode_id": episode_id,
                "mode": mode,
                "timestep": step_idx,
                "timestamp": obs["timestamp"],
                "indoor_temp": obs["indoor_temp"],
                "outdoor_temp": obs["outdoor_temp"],
                "setpoint": active_setpoint,
                "hvac_power_kw": obs["hvac_power_kw"],
                "hvac_on": obs["hvac_mode"] != "off",
                "hour": obs["hour"],
                "month": obs["month"],
                "time_to_target_min": float("nan"),
            })

            if target_reached(obs):
                reached = True
                reach_time_min = step_idx * timestep_minutes
                break

            if env.done:
                break

            try:
                obs = env.step(action_base)
            except Exception:
                break

        if ep_rows:
            if reached and reach_time_min is not None:
                ep_rows[0]["time_to_target_min"] = reach_time_min
            rows.extend(ep_rows)
            episode_id += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hvac_on"] = df["hvac_on"].astype(bool)
    return df


def evaluate(
    policy: Union[Callable, dict[str, Callable]],
    buildings: Union[str, list[str]] = "all",
    scenarios: list[str] = ["baseline", "precool", "setback"],
    n_days: int = 7,
    start_date: str = "2017-07-01",
    price_signal: Optional[np.ndarray] = None,
    timestep_minutes: int = 15,
) -> pd.DataFrame:
    """
    Evaluate one or more policies across buildings and DR scenarios.

    Args:
        policy: Single callable OR dict mapping name→callable.
                Baseline() is always included as "baseline".
        buildings: "all" or list of building IDs.
        scenarios: Subset of ["baseline", "precool", "setback", "price_response"].
        n_days: Number of days per evaluation run.
        start_date: ISO date string for start of evaluation.
        price_signal: 24-element array ($/kWh). None = CA TOU default.
        timestep_minutes: Resolution.

    Returns:
        Tidy DataFrame. One row per (policy, building, scenario).
    """
    if price_signal is None:
        price_signal = _CA_TOU_PRICES.copy()
    price_signal = np.asarray(price_signal)

    # Normalize policy arg
    if callable(policy):
        policies: dict[str, Callable] = {"policy": policy}
    else:
        policies = dict(policy)
    if "baseline" not in policies:
        policies["baseline"] = Baseline()

    # Resolve building list
    building_ids = list(BUILDINGS.keys()) if buildings == "all" else list(buildings)

    # Scenario → policy factory
    _scenario_policies = {
        "baseline": Baseline,
        "precool": PreCool,
        "setback": Setback,
        "price_response": PriceResponse,
    }

    results: list[dict] = []

    # First pass: collect baseline peak_kwh and cost per (building, scenario) for delta metrics
    baseline_stats: dict[tuple[str, str], tuple[float, float]] = {}

    for building_id in building_ids:
        for scenario in scenarios:
            scenario_policy = _scenario_policies.get(scenario, Baseline)()
            try:
                env = ThermalEnv(
                    building=building_id,
                    timestep_minutes=timestep_minutes,
                    run_period_days=n_days,
                    price_signal=price_signal,
                )
                obs = env.reset(date=start_date)
                while not env.done:
                    action = scenario_policy(obs)
                    obs = env.step(action)
                metrics = _compute_metrics(
                    env.history, price_signal,
                    baseline_peak_kwh=0.0, baseline_cost=0.0,
                )
                baseline_stats[(building_id, scenario)] = (metrics["peak_kwh"], metrics["cost_usd"])
            except (FileNotFoundError, ImportError):
                baseline_stats[(building_id, scenario)] = (0.0, 0.0)

    # Second pass: run each named policy
    for policy_name, pol in policies.items():
        for building_id in building_ids:
            for scenario in scenarios:
                baseline_peak_kwh, baseline_cost = baseline_stats.get((building_id, scenario), (0.0, 0.0))
                try:
                    env = ThermalEnv(
                        building=building_id,
                        timestep_minutes=timestep_minutes,
                        run_period_days=n_days,
                        price_signal=price_signal,
                    )
                    obs = env.reset(date=start_date)
                    while not env.done:
                        action = pol(obs)
                        obs = env.step(action)
                    metrics = _compute_metrics(
                        env.history, price_signal,
                        baseline_peak_kwh=baseline_peak_kwh,
                        baseline_cost=baseline_cost,
                    )
                except (FileNotFoundError, ImportError) as e:
                    metrics = {
                        "total_kwh": float("nan"), "peak_kw": float("nan"),
                        "par": float("nan"), "peak_kwh": float("nan"),
                        "shifted_kwh": float("nan"), "comfort_violations_h": float("nan"),
                        "discomfort_degree_hours": float("nan"),
                        "max_temp_deviation_f": float("nan"),
                        "cost_usd": float("nan"), "cost_savings_usd": float("nan"),
                    }

                row = {"policy": policy_name, "building": building_id, "scenario": scenario}
                row.update(metrics)
                results.append(row)

    return pd.DataFrame(results)


def _compute_metrics(
    history: pd.DataFrame,
    price_signal: np.ndarray,
    baseline_peak_kwh: float,
    baseline_cost: float,
    peak_hours: tuple[int, int] = (17, 20),
) -> dict:
    """
    Compute all evaluation metrics from a completed episode history.

    Args:
        history: env.history DataFrame.
        price_signal: 24-element price array.
        baseline_peak_kwh: peak_kwh from the baseline scenario (for shifted_kwh).
        baseline_cost: cost_usd from the baseline policy run (for cost_savings_usd).
        peak_hours: (start_inclusive, end_exclusive) hours of peak period.

    Returns dict with keys matching evaluate() output columns (excluding policy/building/scenario).
    """
    if len(history) == 0:
        return {
            "total_kwh": 0.0, "peak_kw": 0.0, "par": 0.0, "peak_kwh": 0.0,
            "shifted_kwh": 0.0, "comfort_violations_h": 0.0,
            "discomfort_degree_hours": 0.0, "max_temp_deviation_f": 0.0,
            "cost_usd": 0.0, "cost_savings_usd": 0.0,
        }

    h = history.copy()
    # Infer timestep duration in hours from timestamp spacing
    if len(h) > 1:
        dt_hours = (h["timestamp"].iloc[1] - h["timestamp"].iloc[0]).total_seconds() / 3600.0
    else:
        dt_hours = 1.0

    # Energy
    energy_kwh = h["hvac_power_kw"] * dt_hours
    total_kwh = float(energy_kwh.sum())
    peak_kw = float(h["hvac_power_kw"].max())
    mean_kw = float(h["hvac_power_kw"].mean())
    par = peak_kw / mean_kw if mean_kw > 0 else 0.0

    # Peak period energy
    peak_start, peak_end = peak_hours
    in_peak = (h["hour"] >= peak_start) & (h["hour"] < peak_end)
    peak_kwh = float(energy_kwh[in_peak].sum())
    shifted_kwh = max(0.0, baseline_peak_kwh - peak_kwh)

    # Comfort violations: outside [heat_sp - 1, cool_sp + 1]
    lower_bound = h["heat_setpoint"] - 1.0
    upper_bound = h["cool_setpoint"] + 1.0
    too_cold = h["indoor_temp"] < lower_bound
    too_hot = h["indoor_temp"] > upper_bound
    oob = too_cold | too_hot
    comfort_violations_h = float(oob.sum() * dt_hours)

    deviation = pd.Series(np.zeros(len(h)), index=h.index)
    deviation[too_cold] = lower_bound[too_cold] - h["indoor_temp"][too_cold]
    deviation[too_hot] = h["indoor_temp"][too_hot] - upper_bound[too_hot]
    discomfort_degree_hours = float((deviation * dt_hours).sum())
    max_temp_deviation_f = float(deviation.max())

    # Cost
    price_per_step = h["hour"].map(lambda hr: float(price_signal[hr % 24]))
    cost_usd = float((energy_kwh * price_per_step).sum())
    cost_savings_usd = max(0.0, baseline_cost - cost_usd)

    return {
        "total_kwh": total_kwh,
        "peak_kw": peak_kw,
        "par": par,
        "peak_kwh": peak_kwh,
        "shifted_kwh": shifted_kwh,
        "comfort_violations_h": comfort_violations_h,
        "discomfort_degree_hours": discomfort_degree_hours,
        "max_temp_deviation_f": max_temp_deviation_f,
        "cost_usd": cost_usd,
        "cost_savings_usd": cost_savings_usd,
    }
