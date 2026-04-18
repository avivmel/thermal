#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


BUILDINGS = [
    "small_cold_heatpump",
    "medium_cold_heatpump",
    "large_cold_resistance",
    "small_mixed_heatpump",
    "medium_mixed_heatpump",
    "large_mixed_ac",
    "small_hot_heatpump",
    "medium_hot_heatpump",
    "large_hot_ac",
]


@dataclass(frozen=True)
class BaselineCase:
    start_date: str
    mode: str


DEFAULT_CASES = [
    BaselineCase("2017-01-15", "heating"),
    BaselineCase("2017-02-15", "heating"),
    BaselineCase("2017-12-15", "heating"),
    BaselineCase("2017-06-15", "cooling"),
    BaselineCase("2017-07-15", "cooling"),
    BaselineCase("2017-08-15", "cooling"),
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    commands = build_commands(args, out_dir)

    if args.dry_run:
        for command in commands:
            print(format_command(command.argv))
        print(f"\nDry run: {len(commands)} baseline cells")
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

    run_selector(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline-only MPC evaluation candidates for cell screening."
    )
    parser.add_argument("--out-dir", default="results/mpc_eval/baseline_grid")
    parser.add_argument(
        "--selection-out-dir",
        help="Defaults to <out_dir>/selection.",
    )
    parser.add_argument(
        "--evaluation-out-dir",
        default="results/mpc_eval/selected_grid",
        help="Out directory embedded in generated selected-cell run commands.",
    )
    parser.add_argument("--run-period-days", type=int, default=7)
    parser.add_argument("--timestep-minutes", type=int, choices=[5, 15, 60], default=15)
    parser.add_argument("--peak-start", default="17:00")
    parser.add_argument("--peak-end", default="20:00")
    parser.add_argument("--forecast-kind", choices=["epw", "persistence", "none"], default="epw")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--top-per-mode",
        type=int,
        default=0,
        help="Passed to select_mpc_eval_cells.py; 0 keeps all selected cells.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class BaselineCommand:
    argv: list[str]
    metrics_path: Path


def build_commands(args: argparse.Namespace, out_dir: Path) -> list[BaselineCommand]:
    commands: list[BaselineCommand] = []
    for building in BUILDINGS:
        for case in DEFAULT_CASES:
            command_out_dir = out_dir
            if args.workers > 1:
                command_out_dir = out_dir / "isolated" / cell_slug(
                    building=building,
                    start_date=case.start_date,
                    mode=case.mode,
                    peak_start=args.peak_start,
                    peak_end=args.peak_end,
                    forecast_kind=args.forecast_kind,
                )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_mpc_evaluation.py"),
                "--controller",
                "baseline",
                "--building",
                building,
                "--start-date",
                case.start_date,
                "--mode",
                case.mode,
                "--peak-start",
                args.peak_start,
                "--peak-end",
                args.peak_end,
                "--run-period-days",
                str(args.run_period_days),
                "--timestep-minutes",
                str(args.timestep_minutes),
                "--out-dir",
                str(command_out_dir),
                "--forecast-kind",
                args.forecast_kind,
            ]
            if args.overwrite:
                command.append("--overwrite")
            commands.append(BaselineCommand(command, command_out_dir / "metrics.parquet"))
    return commands


def run_command(command: list[str]) -> None:
    print(f"RUN: {format_command(command)}", flush=True)
    subprocess.run(command, check=True)


def merge_isolated_metrics(commands: list[BaselineCommand], out_dir: Path) -> None:
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


def run_selector(args: argparse.Namespace, out_dir: Path) -> None:
    if args.skip_selection:
        return
    command = [
        sys.executable,
        str(ROOT / "scripts" / "select_mpc_eval_cells.py"),
        str(out_dir),
        "--evaluation-out-dir",
        args.evaluation_out_dir,
        "--top-per-mode",
        str(args.top_per_mode),
    ]
    if args.selection_out_dir:
        command.extend(["--out-dir", args.selection_out_dir])
    print(f"RUN: {format_command(command)}", flush=True)
    subprocess.run(command, check=True)


def format_command(command: list[str]) -> str:
    return " ".join(shell_quote(part) for part in command)


def cell_slug(
    *,
    building: str,
    start_date: str,
    mode: str,
    peak_start: str,
    peak_end: str,
    forecast_kind: str,
) -> str:
    peak = f"{peak_start.replace(':', '')}-{peak_end.replace(':', '')}"
    return f"{building}_{start_date}_{mode}_baseline_peak{peak}_{forecast_kind}"


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
