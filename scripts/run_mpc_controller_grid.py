#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ControllerCommand:
    argv: list[str]
    metrics_path: Path


def main() -> None:
    args = parse_args()
    grid = pd.read_csv(args.controller_grid)
    out_dir = Path(args.out_dir)
    commands = build_commands(grid, args, out_dir)

    if args.dry_run:
        for command in commands:
            print(format_command(command.argv))
        print(f"\nDry run: {len(commands)} controller cells")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        for command in commands:
            run_command(command.argv)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_command, command.argv): command for command in commands}
            for future in as_completed(futures):
                command = futures[future]
                try:
                    future.result()
                except subprocess.CalledProcessError as exc:
                    print(f"FAILED: {format_command(command.argv)}", file=sys.stderr)
                    raise SystemExit(exc.returncode) from exc

    merge_isolated_metrics(commands, out_dir)
    run_analysis(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a selected MPC controller grid from controller_grid.csv."
    )
    parser.add_argument("controller_grid")
    parser.add_argument("--out-dir", default="results/mpc_eval/selected_grid")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--active-model", default="models/active_time_xgb.pkl")
    parser.add_argument("--drift-model", default="models/drift_time_xgb.pkl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(
    grid: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[ControllerCommand]:
    required = [
        "controller",
        "building",
        "start_date",
        "mode",
        "peak_start",
        "peak_end",
        "run_period_days",
        "timestep_minutes",
        "forecast_kind",
        "comfort_lower_f",
        "comfort_upper_f",
        "normal_heat_setpoint_f",
        "normal_cool_setpoint_f",
    ]
    missing = [column for column in required if column not in grid.columns]
    if missing:
        raise SystemExit(f"controller grid missing required columns: {', '.join(missing)}")

    commands: list[ControllerCommand] = []
    for _, row in grid.iterrows():
        command_out_dir = out_dir / "isolated" / cell_slug(row)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_mpc_evaluation.py"),
            "--controller",
            row["controller"],
            "--building",
            row["building"],
            "--start-date",
            row["start_date"],
            "--mode",
            row["mode"],
            "--peak-start",
            row["peak_start"],
            "--peak-end",
            row["peak_end"],
            "--run-period-days",
            str(row["run_period_days"]),
            "--timestep-minutes",
            str(row["timestep_minutes"]),
            "--out-dir",
            str(command_out_dir),
            "--forecast-kind",
            row["forecast_kind"],
            "--comfort-lower",
            str(row["comfort_lower_f"]),
            "--comfort-upper",
            str(row["comfort_upper_f"]),
            "--normal-heat-setpoint",
            str(row["normal_heat_setpoint_f"]),
            "--normal-cool-setpoint",
            str(row["normal_cool_setpoint_f"]),
        ]
        if row["controller"] == "mpc":
            command.extend(
                [
                    "--active-model",
                    args.active_model,
                    "--drift-model",
                    args.drift_model,
                ]
            )
        if args.overwrite:
            command.append("--overwrite")
        commands.append(ControllerCommand(command, command_out_dir / "metrics.parquet"))
    return commands


def run_command(command: list[str]) -> None:
    print(f"RUN: {format_command(command)}", flush=True)
    subprocess.run(command, check=True)


def merge_isolated_metrics(commands: list[ControllerCommand], out_dir: Path) -> None:
    frames = []
    missing = []
    for command in commands:
        if command.metrics_path.exists():
            frames.append(pd.read_parquet(command.metrics_path))
        else:
            missing.append(str(command.metrics_path))
    if missing:
        raise SystemExit("missing isolated metrics files:\n" + "\n".join(missing))
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_parquet(out_dir / "metrics.parquet", index=False)
    print(f"Merged isolated metrics: {len(metrics)} rows -> {out_dir / 'metrics.parquet'}")


def run_analysis(args: argparse.Namespace, out_dir: Path) -> None:
    if args.skip_analysis:
        return
    command = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_mpc_eval.py"),
        str(out_dir),
    ]
    print(f"RUN: {format_command(command)}", flush=True)
    subprocess.run(command, check=True)


def cell_slug(row: pd.Series) -> str:
    peak = f"{str(row['peak_start']).replace(':', '')}-{str(row['peak_end']).replace(':', '')}"
    return (
        f"{row['building']}_{row['start_date']}_{row['mode']}_{row['controller']}"
        f"_peak{peak}_{row['forecast_kind']}"
    )


def format_command(command: list[Any]) -> str:
    return " ".join(shell_quote(str(part)) for part in command)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
