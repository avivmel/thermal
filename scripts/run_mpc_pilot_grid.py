#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PilotCase:
    building: str
    start_date: str
    mode: str


CONTROLLERS_BY_MODE = {
    "heating": ["baseline", "mpc", "preheat", "setback"],
    "cooling": ["baseline", "mpc", "precool", "setback"],
}

DEFAULT_CASES = [
    PilotCase("small_cold_heatpump", "2017-01-15", "heating"),
    PilotCase("small_cold_heatpump", "2017-07-15", "cooling"),
    PilotCase("medium_mixed_heatpump", "2017-01-15", "heating"),
    PilotCase("medium_mixed_heatpump", "2017-07-15", "cooling"),
    PilotCase("large_hot_ac", "2017-01-15", "heating"),
    PilotCase("large_hot_ac", "2017-07-15", "cooling"),
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    commands = build_commands(args, out_dir)

    if args.dry_run:
        for command in commands:
            print(format_command(command))
        print(f"\nDry run: {len(commands)} cells")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        for command in commands:
            run_command(command)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_command, command): command for command in commands}
            for future in as_completed(futures):
                command = futures[future]
                try:
                    future.result()
                except subprocess.CalledProcessError as exc:
                    print(f"FAILED: {format_command(command)}", file=sys.stderr)
                    raise SystemExit(exc.returncode) from exc

    run_analysis(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact 7-day MPC pilot grid before the full paper sweep."
    )
    parser.add_argument("--out-dir", default="results/mpc_eval/pilot_grid")
    parser.add_argument("--run-period-days", type=int, default=7)
    parser.add_argument("--timestep-minutes", type=int, choices=[5, 15, 60], default=15)
    parser.add_argument("--peak-start", default="17:00")
    parser.add_argument("--peak-end", default="20:00")
    parser.add_argument("--forecast-kind", choices=["epw", "persistence", "none"], default="epw")
    parser.add_argument("--active-model", default="models/active_time_xgb.pkl")
    parser.add_argument("--drift-model", default="models/drift_time_xgb.pkl")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace, out_dir: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    for case in DEFAULT_CASES:
        for controller in CONTROLLERS_BY_MODE[case.mode]:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_mpc_evaluation.py"),
                "--controller",
                controller,
                "--building",
                case.building,
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
                str(out_dir),
                "--forecast-kind",
                args.forecast_kind,
            ]
            if controller == "mpc":
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
            commands.append(command)
    return commands


def run_command(command: list[str]) -> None:
    print(f"RUN: {format_command(command)}", flush=True)
    subprocess.run(command, check=True)


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


def format_command(command: list[str]) -> str:
    return " ".join(shell_quote(part) for part in command)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
