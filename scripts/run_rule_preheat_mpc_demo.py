from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpc import FittedFirstPassagePredictor, MPCConfig
from mpc.controller import make_daily_heating_mpc
from rule_mpc import RulePreheatMPCConfig, RulePreheatMPCController
from thermalgym import Baseline, PreHeat, ThermalEnv, get_building


def run_policy(env: ThermalEnv, policy, start_date: str, policy_name: str) -> tuple[pd.DataFrame, list[dict]]:
    obs = env.reset(date=start_date)
    plans: list[dict] = []
    while not env.done:
        action = policy(obs)
        plan = getattr(policy, "last_plan", None)
        if plan is not None:
            plan_row = plan.to_dict()
            plan_row["policy"] = policy_name
            plan_row["timestamp"] = pd.Timestamp(obs["timestamp"]).isoformat()
            plans.append(plan_row)
        obs = env.step(action)
    history = env.history
    history["policy"] = policy_name
    return history, plans


def summarize(
    name: str,
    history: pd.DataFrame,
    baseline_peak_kwh: float,
    baseline_cost: float,
    baseline_heat_f: float,
    peak_heat_f: float,
    peak_start_hour: int,
    peak_end_hour: int,
    timestep_minutes: int,
    price_signal: np.ndarray,
) -> dict:
    if len(history) == 0:
        return {"policy": name}

    h = history.copy()
    if len(h) > 1:
        dt_hours = (h["timestamp"].iloc[1] - h["timestamp"].iloc[0]).total_seconds() / 3600.0
        if dt_hours <= 0:
            dt_hours = timestep_minutes / 60.0
    else:
        dt_hours = timestep_minutes / 60.0

    energy_kwh = h["hvac_power_kw"] * dt_hours
    in_peak = (h["hour"] >= peak_start_hour) & (h["hour"] < peak_end_hour)
    lower_comfort = np.where(in_peak, peak_heat_f, baseline_heat_f)
    upper_comfort = np.full(len(h), 77.0, dtype=float)

    too_cold = h["indoor_temp"].to_numpy() < lower_comfort - 1.0
    too_hot = h["indoor_temp"].to_numpy() > upper_comfort
    deviation = np.zeros(len(h), dtype=float)
    indoor = h["indoor_temp"].to_numpy(dtype=float)
    deviation[too_cold] = lower_comfort[too_cold] - 1.0 - indoor[too_cold]
    deviation[too_hot] = indoor[too_hot] - upper_comfort[too_hot]

    peak_kwh = float(energy_kwh[in_peak].sum())
    peak_window_peak_kw = float(h.loc[in_peak, "hvac_power_kw"].max()) if in_peak.any() else 0.0
    price_per_step = h["hour"].map(lambda hr: float(price_signal[hr % 24]))
    cost_usd = float((energy_kwh * price_per_step).sum())
    mean_kw = float(h["hvac_power_kw"].mean())

    return {
        "policy": name,
        "total_kwh": float(energy_kwh.sum()),
        "peak_kw": float(h["hvac_power_kw"].max()),
        "peak_window_peak_kw": peak_window_peak_kw,
        "par": float(h["hvac_power_kw"].max() / mean_kw) if mean_kw > 0 else 0.0,
        "peak_kwh": peak_kwh,
        "shifted_kwh": baseline_peak_kwh - peak_kwh,
        "comfort_violations_h": float((too_cold | too_hot).sum() * dt_hours),
        "discomfort_degree_hours": float((deviation * dt_hours).sum()),
        "max_temp_deviation_f": float(deviation.max()),
        "max_indoor_temp_f": float(h["indoor_temp"].max()),
        "cost_usd": cost_usd,
        "cost_savings_usd": baseline_cost - cost_usd,
    }


def write_plan_jsonl(path: Path, plans: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for plan in plans:
            f.write(json.dumps(plan) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based preheat MPC heating-day demo.")
    parser.add_argument("--building", default="medium_cold_heatpump")
    parser.add_argument("--date", default="2017-01-15")
    parser.add_argument("--baseline-heat", type=float, default=68.0)
    parser.add_argument("--peak-heat", type=float, default=66.0)
    parser.add_argument("--peak-start", type=int, default=17)
    parser.add_argument("--peak-end", type=int, default=20)
    parser.add_argument("--preheat-hours", type=float, default=2.0)
    parser.add_argument("--max-active-rows", type=int, default=60000)
    parser.add_argument("--max-drift-rows", type=int, default=60000)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = FittedFirstPassagePredictor.from_local_data(
        max_active_rows=args.max_active_rows,
        max_drift_rows=args.max_drift_rows,
    )

    building = get_building(args.building)
    dp_config = MPCConfig(
        action_min_f=args.peak_heat,
        action_max_f=74.0,
        lambda_energy=8.0,
        lambda_comfort=250.0,
        terminal_penalty=1200.0,
    )
    rule_config = RulePreheatMPCConfig(
        timestep_minutes=dp_config.timestep_minutes,
        normal_heat_setpoint=args.baseline_heat,
        peak_heat_setpoint=args.peak_heat,
        peak_lower_comfort=args.peak_heat,
        peak_start_hour=args.peak_start,
        peak_end_hour=args.peak_end,
    )
    mpc_policy = make_daily_heating_mpc(
        predictor=predictor,
        hvac_power_kw=building.hvac_capacity_kw,
        config=dp_config,
        peak_start_hour=args.peak_start,
        peak_end_hour=args.peak_end,
        default_heat_f=args.baseline_heat,
        peak_heat_f=args.peak_heat,
    )
    rule_policy = RulePreheatMPCController(predictor=predictor, config=rule_config)

    policies = {
        "baseline": Baseline(heat_setpoint=args.baseline_heat, cool_setpoint=76.0),
        "fixed_preheat": PreHeat(
            preheat_offset=args.baseline_heat - args.peak_heat,
            preheat_hours=args.preheat_hours,
            peak_start=args.peak_start,
            peak_end=args.peak_end,
            setback=args.baseline_heat - args.peak_heat,
            base_heat=args.baseline_heat,
            base_cool=76.0,
        ),
        "dp_mpc": mpc_policy,
        "rule_preheat_mpc": rule_policy,
    }

    price_signal = np.array([0.09] * 24, dtype=float)
    histories: list[pd.DataFrame] = []
    all_plans: list[dict] = []
    summaries: list[dict] = []
    baseline_peak_kwh = 0.0
    baseline_cost = 0.0

    for name, policy in policies.items():
        try:
            env = ThermalEnv(
                building=args.building,
                timestep_minutes=dp_config.timestep_minutes,
                run_period_days=1,
                price_signal=price_signal,
            )
            history, plans = run_policy(env, policy, args.date, name)
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            print(f"{name}: unable to run ThermalEnv demo: {exc}")
            return

        summary = summarize(
            name=name,
            history=history,
            baseline_peak_kwh=baseline_peak_kwh,
            baseline_cost=baseline_cost,
            baseline_heat_f=args.baseline_heat,
            peak_heat_f=args.peak_heat,
            peak_start_hour=args.peak_start,
            peak_end_hour=args.peak_end,
            timestep_minutes=dp_config.timestep_minutes,
            price_signal=price_signal,
        )
        if name == "baseline":
            baseline_peak_kwh = summary["peak_kwh"]
            baseline_cost = summary["cost_usd"]
            summary["shifted_kwh"] = 0.0
            summary["cost_savings_usd"] = 0.0

        histories.append(history)
        all_plans.extend(plans)
        summaries.append(summary)

    history_path = output_dir / "rule_preheat_mpc_histories.csv"
    summary_path = output_dir / "rule_preheat_mpc_summary.csv"
    plan_path = output_dir / "rule_preheat_mpc_plans.jsonl"

    pd.concat(histories, ignore_index=True).to_csv(history_path, index=False)
    summary_df = pd.DataFrame(summaries).set_index("policy")
    summary_df.to_csv(summary_path)
    write_plan_jsonl(plan_path, all_plans)

    print("Predictor metadata:")
    print(asdict(predictor.metadata))
    print("\nRule-preheat config:")
    print(rule_config.to_dict())
    print("\nHeating-day metrics:")
    print(
        summary_df[
            [
                "total_kwh",
                "peak_kw",
                "peak_window_peak_kw",
                "par",
                "peak_kwh",
                "comfort_violations_h",
                "discomfort_degree_hours",
                "max_indoor_temp_f",
            ]
        ].round(3)
    )
    print(f"\nWrote {history_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plan_path}")


if __name__ == "__main__":
    main()
