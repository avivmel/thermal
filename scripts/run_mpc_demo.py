from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd

from mpc import FittedFirstPassagePredictor, MPCConfig
from mpc.controller import make_daily_heating_mpc
from thermalgym import Baseline, PreHeat, ThermalEnv, get_building
from thermalgym.generate import _compute_metrics


def run_policy(env: ThermalEnv, policy, start_date: str) -> pd.DataFrame:
    obs = env.reset(date=start_date)
    while not env.done:
        action = policy(obs)
        obs = env.step(action)
    return env.history


def summarize(name: str, history: pd.DataFrame) -> dict:
    metrics = _compute_metrics(
        history=history,
        price_signal=np.array([0.09] * 24, dtype=float),
        baseline_peak_kwh=0.0,
        baseline_cost=0.0,
    )
    metrics["policy"] = name
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a heating-day MPC demo in ThermalGym.")
    parser.add_argument("--building", default="medium_cold_heatpump")
    parser.add_argument("--date", default="2017-01-15")
    parser.add_argument("--baseline-heat", type=float, default=68.0)
    parser.add_argument("--peak-heat", type=float, default=66.0)
    parser.add_argument("--peak-start", type=int, default=17)
    parser.add_argument("--peak-end", type=int, default=20)
    parser.add_argument("--preheat-hours", type=float, default=2.0)
    parser.add_argument("--max-active-rows", type=int, default=60000)
    parser.add_argument("--max-drift-rows", type=int, default=60000)
    args = parser.parse_args()

    predictor = FittedFirstPassagePredictor.from_local_data(
        max_active_rows=args.max_active_rows,
        max_drift_rows=args.max_drift_rows,
    )

    building = get_building(args.building)
    config = MPCConfig(
        action_min_f=args.peak_heat,
        action_max_f=74.0,
        lambda_energy=8.0,
        lambda_comfort=250.0,
        terminal_penalty=1200.0,
    )
    mpc_policy = make_daily_heating_mpc(
        predictor=predictor,
        hvac_power_kw=building.hvac_capacity_kw,
        config=config,
        peak_start_hour=args.peak_start,
        peak_end_hour=args.peak_end,
        default_heat_f=args.baseline_heat,
        peak_heat_f=args.peak_heat,
    )

    policies = {
        "baseline": Baseline(heat_setpoint=args.baseline_heat, cool_setpoint=76.0),
        "preheat": PreHeat(
            preheat_offset=args.baseline_heat - args.peak_heat,
            preheat_hours=args.preheat_hours,
            peak_start=args.peak_start,
            peak_end=args.peak_end,
            setback=args.baseline_heat - args.peak_heat,
            base_heat=args.baseline_heat,
            base_cool=76.0,
        ),
        "mpc": mpc_policy,
    }

    summaries: list[dict] = []
    for name, policy in policies.items():
        try:
            env = ThermalEnv(
                building=args.building,
                timestep_minutes=config.timestep_minutes,
                run_period_days=1,
            )
            history = run_policy(env, policy, args.date)
            summaries.append(summarize(name, history))
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            print(f"{name}: unable to run ThermalEnv demo: {exc}")
            return

    result = pd.DataFrame(summaries).set_index("policy")
    print("Predictor metadata:")
    print(asdict(predictor.metadata))
    print("\nHeating-day metrics:")
    print(
        result[
            [
                "total_kwh",
                "peak_kw",
                "par",
                "peak_kwh",
                "comfort_violations_h",
                "discomfort_degree_hours",
            ]
        ].round(3)
    )


if __name__ == "__main__":
    main()
