#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.forecast import EPWForecast, PersistenceForecast
from mpc.model_interfaces import XGBFirstPassagePredictor
from mpc.peak_mpc import MPCConfig, PeakAwareMPCController, PeakWindow
from mpc.thermalgym_adapter import ThermalGymMPCAdapter
from thermalgym import Baseline, PreCool, PreHeat, Setback, ThermalEnv, get_building


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "metrics.parquet"

    if args.recompute_relative_only:
        recompute_relative_metrics_file(
            metrics_path,
            recompute_runtime_from_raw_files=args.recompute_runtime_from_raw,
            recompute_metrics_from_raw_files=args.recompute_metrics_from_raw,
        )
        print(f"Updated relative metrics: {metrics_path}")
        return

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cell = build_cell(args)
    cell_id = cell_id_from_args(args)
    raw_path = raw_dir / f"{cell_id}.parquet"
    commands_path = raw_dir / f"{cell_id}_commands.parquet"
    done_path = raw_dir / f"{cell_id}.done"
    manifest_path = out_dir / "manifest.json"

    if done_path.exists() and not args.overwrite:
        print(f"Skipping existing cell: {cell_id}")
        return

    started = time.time()
    building = get_building(args.building)
    runner = build_controller_runner(args, building)

    env = ThermalEnv(
        building=building,
        timestep_minutes=args.timestep_minutes,
        run_period_days=args.run_period_days,
    )
    obs = env.reset(date=args.start_date)

    command_rows: list[dict[str, Any]] = []
    while not env.done:
        action, command_row = runner.action_for_obs(obs)
        command_rows.append(command_row)
        obs = env.step(action)

    history = env.history
    commands = pd.DataFrame(command_rows)
    history.to_parquet(raw_path, index=False)
    commands.to_parquet(commands_path, index=False)

    metrics = compute_metrics(
        history=history,
        commands=commands,
        cell=cell,
        comfort_lower_f=args.comfort_lower,
        comfort_upper_f=args.comfort_upper,
        peak_start=args.peak_start,
        peak_end=args.peak_end,
        runtime_power_threshold_kw=args.runtime_power_threshold_kw,
    )
    metrics["wall_time_s"] = time.time() - started
    write_metrics(metrics_path, metrics, overwrite_cell_id=cell_id)

    manifest = {
        "git_sha": git_sha(),
        "cell": cell,
        "predictor_metadata": metadata_to_dict(runner.predictor.metadata)
        if runner.predictor is not None
        else None,
        "active_model": str(args.active_model) if runner.predictor is not None else None,
        "drift_model": str(args.drift_model) if runner.predictor is not None else None,
        "forecast_kind": args.forecast_kind,
        "raw_path": str(raw_path),
        "commands_path": str(commands_path),
        "metrics_path": str(metrics_path),
        "wall_time_s": metrics["wall_time_s"],
    }
    write_manifest(manifest_path, cell_id, manifest)
    done_path.write_text("ok\n")

    print(f"Completed cell: {cell_id}")
    print(f"Raw history: {raw_path}")
    print(f"Metrics: {metrics_path}")
    print(pd.DataFrame([metrics]).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one controller evaluation cell in ThermalGym.")
    parser.add_argument("--building")
    parser.add_argument("--start-date")
    parser.add_argument("--mode", choices=["heating", "cooling"])
    parser.add_argument(
        "--controller",
        choices=["mpc", "baseline", "setback", "precool", "preheat"],
        default="mpc",
    )
    parser.add_argument("--peak-start", help="Peak start clock time, e.g. 17:00")
    parser.add_argument("--peak-end", help="Peak end clock time, e.g. 20:00")
    parser.add_argument("--run-period-days", type=int, default=1)
    parser.add_argument("--timestep-minutes", type=int, default=15, choices=[5, 15, 60])
    parser.add_argument("--out-dir", default="results/mpc_eval/smoke")
    parser.add_argument("--forecast-kind", choices=["epw", "persistence", "none"], default="epw")
    parser.add_argument("--active-model", default="models/active_time_xgb.pkl")
    parser.add_argument("--drift-model", default="models/drift_time_xgb.pkl")
    parser.add_argument("--comfort-lower", type=float)
    parser.add_argument("--comfort-upper", type=float)
    parser.add_argument("--normal-heat-setpoint", type=float)
    parser.add_argument("--normal-cool-setpoint", type=float)
    parser.add_argument("--horizon-minutes", type=int, default=360)
    parser.add_argument("--precondition-margin-minutes", type=float, default=10.0)
    parser.add_argument("--drift-safety-margin-minutes", type=float, default=10.0)
    parser.add_argument("--rule-offset", type=float, default=2.0)
    parser.add_argument("--rule-setback", type=float, default=2.0)
    parser.add_argument("--rule-setback-magnitude", type=float, default=4.0)
    parser.add_argument("--rule-precondition-hours", type=float, default=2.0)
    parser.add_argument("--runtime-power-threshold-kw", type=float, default=0.05)
    parser.add_argument(
        "--recompute-relative-only",
        action="store_true",
        help="Update relative metrics in --out-dir/metrics.parquet without running EnergyPlus.",
    )
    parser.add_argument(
        "--recompute-runtime-from-raw",
        action="store_true",
        help="When recomputing, refresh runtime columns from raw history parquet files.",
    )
    parser.add_argument(
        "--recompute-metrics-from-raw",
        action="store_true",
        help="When recomputing, refresh all absolute metric columns from raw parquet files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    apply_mode_defaults(args)
    validate_args(args)
    return args


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "cooling":
        args.comfort_lower = 72.0 if args.comfort_lower is None else args.comfort_lower
        args.comfort_upper = 76.0 if args.comfort_upper is None else args.comfort_upper
        args.normal_heat_setpoint = (
            68.0 if args.normal_heat_setpoint is None else args.normal_heat_setpoint
        )
        args.normal_cool_setpoint = (
            75.0 if args.normal_cool_setpoint is None else args.normal_cool_setpoint
        )
    else:
        args.comfort_lower = 68.0 if args.comfort_lower is None else args.comfort_lower
        args.comfort_upper = 72.0 if args.comfort_upper is None else args.comfort_upper
        args.normal_heat_setpoint = (
            69.0 if args.normal_heat_setpoint is None else args.normal_heat_setpoint
        )
        args.normal_cool_setpoint = (
            76.0 if args.normal_cool_setpoint is None else args.normal_cool_setpoint
        )


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "recompute_relative_only", False):
        return
    for name in ("building", "start_date", "mode", "peak_start", "peak_end"):
        if getattr(args, name) is None:
            option = "--" + name.replace("_", "-")
            raise SystemExit(f"{option} is required unless --recompute-relative-only is set")
    if args.run_period_days <= 0:
        raise SystemExit("--run-period-days must be positive")
    if args.comfort_lower >= args.comfort_upper:
        raise SystemExit("--comfort-lower must be less than --comfort-upper")
    _, peak_start_minute = parse_clock_time(args.peak_start)
    _, peak_end_minute = parse_clock_time(args.peak_end)
    if args.controller != "mpc" and (peak_start_minute != 0 or peak_end_minute != 0):
        raise SystemExit("rule-based controllers require whole-hour --peak-start and --peak-end")


def build_mpc_config(args: argparse.Namespace) -> MPCConfig:
    start_date = pd.Timestamp(args.start_date).normalize()
    windows = []
    for day in range(args.run_period_days):
        day_start = start_date + pd.Timedelta(days=day)
        windows.append(
            PeakWindow(
                start=timestamp_for_time(day_start, args.peak_start),
                end=timestamp_for_time(day_start, args.peak_end),
            )
        )
    return MPCConfig(
        peak_windows=windows,
        comfort_lower_f=args.comfort_lower,
        comfort_upper_f=args.comfort_upper,
        normal_heat_setpoint_f=args.normal_heat_setpoint,
        normal_cool_setpoint_f=args.normal_cool_setpoint,
        control_step_minutes=args.timestep_minutes,
        horizon_minutes=args.horizon_minutes,
        precondition_margin_minutes=args.precondition_margin_minutes,
        drift_safety_margin_minutes=args.drift_safety_margin_minutes,
    )


def build_forecast(kind: str, epw_path: Path, start_date: pd.Timestamp):
    if kind == "epw":
        return EPWForecast.from_epw(epw_path, year=start_date.year)
    if kind in ("none", "persistence"):
        return None
    raise ValueError(f"unsupported forecast kind: {kind}")


class ControllerRunner:
    def __init__(
        self,
        controller_name: str,
        policy,
        adapter: ThermalGymMPCAdapter | None = None,
        forecast_kind: str = "none",
        predictor: XGBFirstPassagePredictor | None = None,
    ) -> None:
        self.controller_name = controller_name
        self.policy = policy
        self.adapter = adapter
        self.forecast_kind = forecast_kind
        self.predictor = predictor

    def action_for_obs(self, obs: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        if self.controller_name == "mpc":
            if self.adapter is None:
                raise RuntimeError("MPC runner requires an adapter")
            state = self.adapter.state_from_obs(obs)
            step_forecast = (
                PersistenceForecast.from_state(state) if self.forecast_kind == "persistence" else None
            )
            command = self.policy.decide(state, forecast=step_forecast)
            action = self.adapter.action_from_command(command)
            return action, {
                "timestamp": state.timestamp,
                "phase": command.phase,
                "reason": command.reason,
                "heat_setpoint": command.heat_setpoint_f,
                "cool_setpoint": command.cool_setpoint_f,
                "action_heat_setpoint": action["heat_setpoint"],
                "action_cool_setpoint": action["cool_setpoint"],
            }

        action = self.policy(obs)
        return action, {
            "timestamp": pd.Timestamp(obs["timestamp"]),
            "phase": "non_mpc",
            "reason": self.controller_name,
            "heat_setpoint": None,
            "cool_setpoint": None,
            "action_heat_setpoint": action["heat_setpoint"],
            "action_cool_setpoint": action["cool_setpoint"],
        }


def build_controller_runner(args: argparse.Namespace, building) -> ControllerRunner:
    if args.controller == "mpc":
        predictor = XGBFirstPassagePredictor.from_model_files(
            active_model_path=args.active_model,
            drift_model_path=args.drift_model,
        )
        forecast = build_forecast(args.forecast_kind, building.epw_path, pd.Timestamp(args.start_date))
        config = build_mpc_config(args)
        controller = PeakAwareMPCController(config, predictor, forecast=forecast)
        adapter = ThermalGymMPCAdapter.for_building(building, mode=args.mode)
        return ControllerRunner(
            controller_name="mpc",
            policy=controller,
            adapter=adapter,
            forecast_kind=args.forecast_kind,
            predictor=predictor,
        )

    peak_start_hour, _ = parse_clock_time(args.peak_start)
    peak_end_hour, _ = parse_clock_time(args.peak_end)
    policy = build_rule_policy(
        controller=args.controller,
        mode=args.mode,
        peak_start_hour=peak_start_hour,
        peak_end_hour=peak_end_hour,
        base_heat=args.normal_heat_setpoint,
        base_cool=args.normal_cool_setpoint,
        rule_offset=args.rule_offset,
        rule_setback=args.rule_setback,
        rule_setback_magnitude=args.rule_setback_magnitude,
        rule_precondition_hours=args.rule_precondition_hours,
    )
    return ControllerRunner(controller_name=args.controller, policy=policy)


def build_rule_policy(
    controller: str,
    mode: str,
    peak_start_hour: int,
    peak_end_hour: int,
    base_heat: float,
    base_cool: float,
    rule_offset: float,
    rule_setback: float,
    rule_setback_magnitude: float,
    rule_precondition_hours: float,
):
    if controller == "baseline":
        return Baseline(heat_setpoint=base_heat, cool_setpoint=base_cool)
    if controller == "setback":
        return Setback(
            magnitude=rule_setback_magnitude,
            peak_start=peak_start_hour,
            peak_end=peak_end_hour,
            mode=mode,
            base_heat=base_heat,
            base_cool=base_cool,
        )
    if controller == "precool":
        return PreCool(
            precool_offset=rule_offset,
            precool_hours=rule_precondition_hours,
            peak_start=peak_start_hour,
            peak_end=peak_end_hour,
            setback=rule_setback,
            base_heat=base_heat,
            base_cool=base_cool,
        )
    if controller == "preheat":
        return PreHeat(
            preheat_offset=rule_offset,
            preheat_hours=rule_precondition_hours,
            peak_start=peak_start_hour,
            peak_end=peak_end_hour,
            setback=rule_setback,
            base_heat=base_heat,
            base_cool=base_cool,
        )
    raise ValueError(f"unsupported controller: {controller}")


def compute_metrics(
    history: pd.DataFrame,
    commands: pd.DataFrame,
    cell: dict[str, Any],
    comfort_lower_f: float,
    comfort_upper_f: float,
    peak_start: str,
    peak_end: str,
    runtime_power_threshold_kw: float = 0.05,
) -> dict[str, Any]:
    if history.empty:
        metrics = {
            "peak_runtime_min": 0.0,
            "peak_energy_kwh": 0.0,
            "total_energy_kwh": 0.0,
            "pre_peak_runtime_min": 0.0,
            "comfort_violation_min": 0.0,
            "comfort_violation_degree_min": 0.0,
            "comfort_band_violation_min": 0.0,
            "comfort_band_violation_degree_min": 0.0,
            "comfort_too_cold_min": 0.0,
            "comfort_too_hot_min": 0.0,
            "comfort_too_cold_degree_min": 0.0,
            "comfort_too_hot_degree_min": 0.0,
            "max_too_cold_f": 0.0,
            "max_too_hot_f": 0.0,
            "max_violation_f": 0.0,
            "cost_usd": 0.0,
            "peak_cost_usd": 0.0,
        }
    else:
        h = history.copy()
        h["timestamp"] = pd.to_datetime(h["timestamp"])
        dt_min = infer_timestep_minutes(h)
        dt_h = dt_min / 60.0
        energy_kwh = h["hvac_power_kw"].astype(float) * dt_h
        in_peak = mask_time_window(h["timestamp"], peak_start, peak_end)
        in_pre_peak = mask_pre_peak_window(h["timestamp"], peak_start, hours=2)
        running = (h["hvac_mode"] != "off") | (
            h["hvac_power_kw"].astype(float) > runtime_power_threshold_kw
        )

        low_deviation = (comfort_lower_f - h["indoor_temp"]).clip(lower=0.0)
        high_deviation = (h["indoor_temp"] - comfort_upper_f).clip(lower=0.0)
        band_deviation = low_deviation + high_deviation
        mode_deviation = comfort_deviation_for_mode(cell.get("mode"), low_deviation, high_deviation)
        band_violating = band_deviation > 0.0
        mode_violating = mode_deviation > 0.0

        prices = h["electricity_price"].astype(float)
        metrics = {
            "peak_runtime_min": float((running & in_peak).sum() * dt_min),
            "peak_energy_kwh": float(energy_kwh[in_peak].sum()),
            "total_energy_kwh": float(energy_kwh.sum()),
            "pre_peak_runtime_min": float((running & in_pre_peak).sum() * dt_min),
            "comfort_violation_min": float(mode_violating.sum() * dt_min),
            "comfort_violation_degree_min": float((mode_deviation * dt_min).sum()),
            "comfort_band_violation_min": float(band_violating.sum() * dt_min),
            "comfort_band_violation_degree_min": float((band_deviation * dt_min).sum()),
            "comfort_too_cold_min": float((low_deviation > 0.0).sum() * dt_min),
            "comfort_too_hot_min": float((high_deviation > 0.0).sum() * dt_min),
            "comfort_too_cold_degree_min": float((low_deviation * dt_min).sum()),
            "comfort_too_hot_degree_min": float((high_deviation * dt_min).sum()),
            "max_too_cold_f": float(low_deviation.max()),
            "max_too_hot_f": float(high_deviation.max()),
            "max_violation_f": float(mode_deviation.max()),
            "cost_usd": float((energy_kwh * prices).sum()),
            "peak_cost_usd": float((energy_kwh[in_peak] * prices[in_peak]).sum()),
        }

    phase_counts = commands["phase"].value_counts().to_dict() if not commands.empty else {}
    for phase in ("normal", "precondition", "peak_coast", "peak_maintain"):
        metrics[f"decisions_phase_{phase}_count"] = int(phase_counts.get(phase, 0))
    metrics["decisions_phase_non_mpc_count"] = int(phase_counts.get("non_mpc", 0))

    metrics.update(cell)
    return metrics


def comfort_deviation_for_mode(
    mode: Any,
    low_deviation: pd.Series,
    high_deviation: pd.Series,
) -> pd.Series:
    if mode == "heating":
        return low_deviation
    if mode == "cooling":
        return high_deviation
    return low_deviation + high_deviation


def infer_timestep_minutes(history: pd.DataFrame) -> float:
    if len(history) < 2:
        return 0.0
    return float(
        (history["timestamp"].iloc[1] - history["timestamp"].iloc[0]).total_seconds() / 60.0
    )


def mask_time_window(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    start_hour, start_minute = parse_clock_time(start_time)
    end_hour, end_minute = parse_clock_time(end_time)
    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute
    if end > start:
        return (minute_of_day >= start) & (minute_of_day < end)
    return (minute_of_day >= start) | (minute_of_day < end)


def mask_pre_peak_window(timestamps: pd.Series, peak_start_time: str, hours: int) -> pd.Series:
    start_hour, start_minute = parse_clock_time(peak_start_time)
    peak_start_minute = start_hour * 60 + start_minute
    pre_start = (peak_start_minute - hours * 60) % (24 * 60)
    return mask_time_window(
        timestamps,
        format_clock_time(pre_start),
        format_clock_time(peak_start_minute),
    )


def parse_clock_time(value: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise SystemExit(f"invalid clock time {value!r}; expected HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise SystemExit(f"invalid clock time {value!r}; expected HH:MM")
    return hour, minute


def format_clock_time(minute_of_day: int) -> str:
    minute_of_day = minute_of_day % (24 * 60)
    hour, minute = divmod(minute_of_day, 60)
    return f"{hour:02d}:{minute:02d}"


def timestamp_for_time(date: pd.Timestamp, clock_time: str) -> pd.Timestamp:
    hour, minute = parse_clock_time(clock_time)
    return date.normalize() + pd.Timedelta(hours=hour, minutes=minute)


def build_cell(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cell_id": cell_id_from_args(args),
        "controller": args.controller,
        "building": args.building,
        "start_date": args.start_date,
        "run_period_days": args.run_period_days,
        "timestep_minutes": args.timestep_minutes,
        "mode": args.mode,
        "peak_start": args.peak_start,
        "peak_end": args.peak_end,
        "comfort_lower_f": args.comfort_lower,
        "comfort_upper_f": args.comfort_upper,
        "normal_heat_setpoint_f": args.normal_heat_setpoint,
        "normal_cool_setpoint_f": args.normal_cool_setpoint,
        "forecast_kind": args.forecast_kind,
        "rule_offset": args.rule_offset,
        "rule_setback": args.rule_setback,
        "rule_setback_magnitude": args.rule_setback_magnitude,
        "rule_precondition_hours": args.rule_precondition_hours,
        "runtime_power_threshold_kw": args.runtime_power_threshold_kw,
    }


def cell_id_from_args(args: argparse.Namespace) -> str:
    return "_".join(
        [
            args.building,
            args.start_date,
            args.mode,
            args.controller,
            f"peak{args.peak_start.replace(':', '')}-{args.peak_end.replace(':', '')}",
            args.forecast_kind,
        ]
    )


def write_metrics(path: Path, metrics: dict[str, Any], overwrite_cell_id: str) -> None:
    row = pd.DataFrame([metrics])
    if path.exists():
        existing = pd.read_parquet(path)
        if "cell_id" in existing.columns:
            existing = existing[existing["cell_id"] != overwrite_cell_id]
        row = pd.concat([existing, row], ignore_index=True)
    row = add_relative_metrics(row)
    row.to_parquet(path, index=False)


RELATIVE_METRIC_COLUMNS = [
    "peak_runtime_reduction_pct",
    "peak_energy_reduction_pct",
    "energy_overhead_pct",
    "cost_savings_pct",
    "peak_cost_savings_pct",
    "comfort_violation_delta_min",
    "comfort_degree_min_delta",
]

BASELINE_MATCH_COLUMNS = [
    "building",
    "start_date",
    "run_period_days",
    "timestep_minutes",
    "mode",
    "peak_start",
    "peak_end",
    "comfort_lower_f",
    "comfort_upper_f",
    "normal_heat_setpoint_f",
    "normal_cool_setpoint_f",
    "forecast_kind",
]


def recompute_relative_metrics_file(
    path: Path,
    recompute_runtime_from_raw_files: bool = False,
    recompute_metrics_from_raw_files: bool = False,
) -> None:
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")
    metrics = pd.read_parquet(path)
    if recompute_metrics_from_raw_files:
        metrics = recompute_metrics_from_raw(path.parent, metrics)
    elif recompute_runtime_from_raw_files:
        metrics = recompute_runtime_from_raw(path.parent, metrics)
    add_relative_metrics(metrics).to_parquet(path, index=False)


METRIC_COLUMNS = [
    "peak_runtime_min",
    "peak_energy_kwh",
    "total_energy_kwh",
    "pre_peak_runtime_min",
    "comfort_violation_min",
    "comfort_violation_degree_min",
    "comfort_band_violation_min",
    "comfort_band_violation_degree_min",
    "comfort_too_cold_min",
    "comfort_too_hot_min",
    "comfort_too_cold_degree_min",
    "comfort_too_hot_degree_min",
    "max_too_cold_f",
    "max_too_hot_f",
    "max_violation_f",
    "cost_usd",
    "peak_cost_usd",
    "decisions_phase_normal_count",
    "decisions_phase_precondition_count",
    "decisions_phase_peak_coast_count",
    "decisions_phase_peak_maintain_count",
    "decisions_phase_non_mpc_count",
]


def recompute_metrics_from_raw(out_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "cell_id" not in metrics.columns:
        return metrics
    result = metrics.copy()
    for index, row in result.iterrows():
        raw_path = out_dir / "raw" / f"{row['cell_id']}.parquet"
        commands_path = out_dir / "raw" / f"{row['cell_id']}_commands.parquet"
        if not raw_path.exists() or not commands_path.exists():
            continue
        recomputed = compute_metrics(
            history=pd.read_parquet(raw_path),
            commands=pd.read_parquet(commands_path),
            cell=cell_metadata_from_metric_row(row),
            comfort_lower_f=float(row["comfort_lower_f"]),
            comfort_upper_f=float(row["comfort_upper_f"]),
            peak_start=row["peak_start"],
            peak_end=row["peak_end"],
            runtime_power_threshold_kw=float(row.get("runtime_power_threshold_kw", 0.05)),
        )
        for column in METRIC_COLUMNS:
            if column in recomputed:
                result.loc[index, column] = recomputed[column]
    return result


def cell_metadata_from_metric_row(row: pd.Series) -> dict[str, Any]:
    excluded = set(METRIC_COLUMNS) | set(RELATIVE_METRIC_COLUMNS)
    return {key: value for key, value in row.to_dict().items() if key not in excluded}


def recompute_runtime_from_raw(out_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "cell_id" not in metrics.columns:
        return metrics
    result = metrics.copy()
    for index, row in result.iterrows():
        raw_path = out_dir / "raw" / f"{row['cell_id']}.parquet"
        if not raw_path.exists():
            continue
        history = pd.read_parquet(raw_path)
        if history.empty:
            continue
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        dt_min = infer_timestep_minutes(history)
        running = (history["hvac_mode"] != "off") | (
            history["hvac_power_kw"].astype(float) > float(row.get("runtime_power_threshold_kw", 0.05))
        )
        in_peak = mask_time_window(history["timestamp"], row["peak_start"], row["peak_end"])
        in_pre_peak = mask_pre_peak_window(history["timestamp"], row["peak_start"], hours=2)
        result.loc[index, "peak_runtime_min"] = float((running & in_peak).sum() * dt_min)
        result.loc[index, "pre_peak_runtime_min"] = float((running & in_pre_peak).sum() * dt_min)
    return result


def add_relative_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    result = metrics.copy()
    for column in RELATIVE_METRIC_COLUMNS:
        result[column] = np.nan

    missing_columns = [column for column in BASELINE_MATCH_COLUMNS if column not in result.columns]
    if missing_columns:
        return result

    grouped = result.groupby(BASELINE_MATCH_COLUMNS, dropna=False, sort=False)
    for _, index in grouped.groups.items():
        group = result.loc[index]
        baselines = group[group["controller"] == "baseline"]
        if baselines.empty:
            continue
        baseline = baselines.iloc[0]
        for row_index in group.index:
            row = result.loc[row_index]
            result.loc[row_index, "peak_runtime_reduction_pct"] = reduction_pct(
                baseline["peak_runtime_min"], row["peak_runtime_min"]
            )
            result.loc[row_index, "peak_energy_reduction_pct"] = reduction_pct(
                baseline["peak_energy_kwh"], row["peak_energy_kwh"]
            )
            result.loc[row_index, "energy_overhead_pct"] = increase_pct(
                baseline["total_energy_kwh"], row["total_energy_kwh"]
            )
            result.loc[row_index, "cost_savings_pct"] = reduction_pct(
                baseline["cost_usd"], row["cost_usd"]
            )
            result.loc[row_index, "peak_cost_savings_pct"] = reduction_pct(
                baseline["peak_cost_usd"], row["peak_cost_usd"]
            )
            result.loc[row_index, "comfort_violation_delta_min"] = (
                float(row["comfort_violation_min"]) - float(baseline["comfort_violation_min"])
            )
            result.loc[row_index, "comfort_degree_min_delta"] = (
                float(row["comfort_violation_degree_min"])
                - float(baseline["comfort_violation_degree_min"])
            )
    return result


def reduction_pct(baseline: Any, value: Any) -> float:
    baseline = float(baseline)
    value = float(value)
    if baseline == 0.0:
        return 0.0 if value == 0.0 else np.nan
    return (baseline - value) / baseline * 100.0


def increase_pct(baseline: Any, value: Any) -> float:
    baseline = float(baseline)
    value = float(value)
    if baseline == 0.0:
        return 0.0 if value == 0.0 else np.nan
    return (value - baseline) / baseline * 100.0


def write_manifest(path: Path, cell_id: str, manifest: dict[str, Any]) -> None:
    if path.exists():
        all_manifests = json.loads(path.read_text())
    else:
        all_manifests = {"cells": {}}
    all_manifests["cells"][cell_id] = manifest
    path.write_text(json.dumps(all_manifests, indent=2, sort_keys=True, default=json_default) + "\n")


def metadata_to_dict(metadata) -> dict[str, Any]:
    if is_dataclass(metadata):
        return asdict(metadata)
    return dict(metadata)


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
