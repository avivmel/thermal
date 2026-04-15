#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermalgym import get_building


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for building in args.buildings:
        for date in args.dates:
            run_baseline_cell(args, building, date)

    metrics_path = out_dir / "metrics.parquet"
    summary = build_screening_summary(
        metrics_path=metrics_path,
        buildings=args.buildings,
        dates=args.dates,
        mode=args.mode,
        peak_start=args.peak_start,
        peak_end=args.peak_end,
    )
    summary_path = out_dir / "screening_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Screening summary: {summary_path}")
    print(
        summary[
            [
                "building",
                "start_date",
                "peak_energy_kwh",
                "peak_runtime_min",
                "total_energy_kwh",
                "comfort_violation_min",
                "outdoor_peak_mean_f",
                "outdoor_daily_max_f",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen ThermalGym cells for useful baseline load.")
    parser.add_argument("--buildings", nargs="+", required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--mode", required=True, choices=["heating", "cooling"])
    parser.add_argument("--peak-start", default="17:00")
    parser.add_argument("--peak-end", default="20:00")
    parser.add_argument("--run-period-days", type=int, default=1)
    parser.add_argument("--timestep-minutes", type=int, choices=[5, 15, 60], default=15)
    parser.add_argument("--out-dir", default="results/mpc_eval/cell_screen")
    parser.add_argument("--forecast-kind", choices=["epw", "persistence", "none"], default="epw")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_baseline_cell(args: argparse.Namespace, building: str, date: str) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_mpc_evaluation.py"),
        "--controller",
        "baseline",
        "--building",
        building,
        "--start-date",
        date,
        "--mode",
        args.mode,
        "--peak-start",
        args.peak_start,
        "--peak-end",
        args.peak_end,
        "--run-period-days",
        str(args.run_period_days),
        "--timestep-minutes",
        str(args.timestep_minutes),
        "--out-dir",
        str(args.out_dir),
        "--forecast-kind",
        args.forecast_kind,
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def build_screening_summary(
    metrics_path: Path,
    buildings: list[str],
    dates: list[str],
    mode: str,
    peak_start: str,
    peak_end: str,
) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    metrics = pd.read_parquet(metrics_path)
    selected = metrics[
        (metrics["controller"] == "baseline")
        & (metrics["building"].isin(buildings))
        & (metrics["start_date"].isin(dates))
        & (metrics["mode"] == mode)
        & (metrics["peak_start"] == peak_start)
        & (metrics["peak_end"] == peak_end)
    ].copy()
    if selected.empty:
        return selected

    weather_rows = [
        weather_summary(
            building=row["building"],
            date=row["start_date"],
            peak_start=peak_start,
            peak_end=peak_end,
        )
        for _, row in selected.iterrows()
    ]
    weather = pd.DataFrame(weather_rows)
    selected = selected.merge(weather, on=["building", "start_date"], how="left")
    selected = selected.sort_values(
        by=[
            "peak_energy_kwh",
            "peak_runtime_min",
            "total_energy_kwh",
            "outdoor_peak_mean_f",
        ],
        ascending=[False, False, False, False],
    )
    return selected.reset_index(drop=True)


def weather_summary(
    building: str,
    date: str,
    peak_start: str,
    peak_end: str,
) -> dict[str, Any]:
    weather = load_epw_weather(get_building(building).epw_path, year=pd.Timestamp(date).year)
    day = weather[weather["timestamp"].dt.date == pd.Timestamp(date).date()]
    in_peak = mask_time_window(day["timestamp"], peak_start, peak_end)
    return {
        "building": building,
        "start_date": date,
        "outdoor_daily_max_f": float(day["outdoor_temp_f"].max()),
        "outdoor_daily_mean_f": float(day["outdoor_temp_f"].mean()),
        "outdoor_daily_min_f": float(day["outdoor_temp_f"].min()),
        "outdoor_peak_mean_f": float(day.loc[in_peak, "outdoor_temp_f"].mean()),
        "outdoor_peak_max_f": float(day.loc[in_peak, "outdoor_temp_f"].max()),
    }


def load_epw_weather(path: Path, year: int) -> pd.DataFrame:
    rows = pd.read_csv(
        path,
        skiprows=8,
        header=None,
        usecols=[1, 2, 3, 6],
        names=["month", "day", "hour", "dry_bulb_c"],
    )
    rows["timestamp"] = [
        pd.Timestamp(year, int(month), int(day)) + pd.Timedelta(hours=int(hour) - 1)
        for month, day, hour in zip(rows["month"], rows["day"], rows["hour"])
    ]
    rows["outdoor_temp_f"] = rows["dry_bulb_c"].astype(float) * 9.0 / 5.0 + 32.0
    return rows[["timestamp", "outdoor_temp_f"]]


def mask_time_window(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    start_minute = clock_minute(start_time)
    end_minute = clock_minute(end_time)
    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    if end_minute > start_minute:
        return (minute_of_day >= start_minute) & (minute_of_day < end_minute)
    return (minute_of_day >= start_minute) | (minute_of_day < end_minute)


def clock_minute(value: str) -> int:
    hour_text, minute_text = value.split(":", maxsplit=1)
    hour = int(hour_text)
    minute = int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid clock time: {value!r}")
    return hour * 60 + minute


if __name__ == "__main__":
    main()

