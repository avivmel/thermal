#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


CELL_COLUMNS = [
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

BASELINE_COLUMNS = [
    "cell_id",
    *CELL_COLUMNS,
    "peak_energy_kwh",
    "peak_runtime_min",
    "total_energy_kwh",
    "comfort_violation_min",
    "comfort_violation_degree_min",
    "comfort_band_violation_min",
    "comfort_band_violation_degree_min",
    "comfort_too_cold_min",
    "comfort_too_hot_min",
    "max_violation_f",
    "max_too_cold_f",
    "max_too_hot_f",
]

CONTROLLERS_BY_MODE = {
    "heating": ["baseline", "mpc", "preheat", "setback"],
    "cooling": ["baseline", "mpc", "precool", "setback"],
}


def main() -> None:
    args = parse_args()
    metrics_path = resolve_metrics_path(args.metrics)
    metrics = pd.read_parquet(metrics_path)
    baselines = baseline_rows(metrics)
    selected, rejected = select_cells(baselines, args)
    controller_grid = expand_controller_grid(selected)

    out_dir = Path(args.out_dir) if args.out_dir else metrics_path.parent / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "selected_cells.csv", index=False)
    rejected.to_csv(out_dir / "rejected_cells.csv", index=False)
    controller_grid.to_csv(out_dir / "controller_grid.csv", index=False)
    write_command_file(controller_grid, out_dir / "run_commands.sh", args.evaluation_out_dir)

    print(f"Read baseline cells: {len(baselines)}")
    print(f"Selected cells: {len(selected)}")
    print(f"Rejected cells: {len(rejected)}")
    print(f"Controller cells: {len(controller_grid)}")
    print(f"Selection output: {out_dir}")
    if not selected.empty:
        print("\nSelected:")
        print(
            selected[
                [
                    "building",
                    "start_date",
                    "mode",
                    "peak_energy_kwh",
                    "peak_runtime_min",
                    "comfort_violation_min",
                    "comfort_band_violation_min",
                ]
            ].to_string(index=False)
        )
    if not rejected.empty:
        print("\nRejected:")
        print(
            rejected[
                [
                    "building",
                    "start_date",
                    "mode",
                    "peak_energy_kwh",
                    "peak_runtime_min",
                    "comfort_violation_min",
                    "comfort_band_violation_min",
                    "reject_reason",
                ]
            ].to_string(index=False)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select paper-suitable MPC evaluation cells from baseline metrics."
    )
    parser.add_argument(
        "metrics",
        help="Evaluation directory or metrics.parquet file containing baseline rows.",
    )
    parser.add_argument("--out-dir", help="Defaults to <eval_dir>/selection.")
    parser.add_argument(
        "--evaluation-out-dir",
        default="results/mpc_eval/selected_grid",
        help="Out directory embedded in generated run_commands.sh.",
    )
    parser.add_argument("--min-baseline-peak-energy-kwh", type=float, default=0.25)
    parser.add_argument("--min-baseline-peak-runtime-min", type=float, default=180.0)
    parser.add_argument("--max-baseline-mode-violation-min", type=float, default=0.0)
    parser.add_argument("--max-baseline-mode-violation-degree-min", type=float, default=0.0)
    parser.add_argument(
        "--max-baseline-band-violation-frac",
        type=float,
        default=0.20,
        help="Reject if full-band violation minutes exceed this fraction of episode minutes.",
    )
    parser.add_argument(
        "--top-per-mode",
        type=int,
        default=0,
        help="Keep only the highest-load N selected cells per mode; 0 keeps all.",
    )
    return parser.parse_args()


def resolve_metrics_path(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "metrics.parquet"
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")
    return path


def baseline_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in BASELINE_COLUMNS if column not in metrics.columns]
    if missing:
        raise SystemExit(f"metrics file missing required columns: {', '.join(missing)}")

    baselines = metrics[metrics["controller"] == "baseline"].copy()
    if baselines.empty:
        raise SystemExit("metrics file has no baseline rows")
    baselines["episode_minutes"] = (
        baselines["run_period_days"].astype(float) * 24.0 * 60.0
    )
    baselines["baseline_band_violation_frac"] = (
        baselines["comfort_band_violation_min"].astype(float)
        / baselines["episode_minutes"].clip(lower=1.0)
    )
    return baselines[BASELINE_COLUMNS + ["episode_minutes", "baseline_band_violation_frac"]]


def select_cells(
    baselines: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in baselines.iterrows():
        reasons = rejection_reasons(row, args)
        data = row.to_dict()
        data["reject_reason"] = "; ".join(reasons)
        data["selected"] = not reasons
        rows.append(data)

    scored = pd.DataFrame(rows)
    selected = scored[scored["selected"]].copy()
    rejected = scored[~scored["selected"]].copy()

    selected = selected.sort_values(
        ["mode", "peak_energy_kwh", "peak_runtime_min"],
        ascending=[True, False, False],
    )
    if args.top_per_mode > 0 and not selected.empty:
        selected = (
            selected.groupby("mode", group_keys=False, sort=False)
            .head(args.top_per_mode)
            .reset_index(drop=True)
        )
    else:
        selected = selected.reset_index(drop=True)

    rejected = rejected.sort_values(["mode", "building", "start_date"]).reset_index(drop=True)
    return selected, rejected


def rejection_reasons(row: pd.Series, args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if float(row["peak_energy_kwh"]) < args.min_baseline_peak_energy_kwh:
        reasons.append(
            f"peak_energy_kwh<{args.min_baseline_peak_energy_kwh:g}"
        )
    if float(row["peak_runtime_min"]) < args.min_baseline_peak_runtime_min:
        reasons.append(
            f"peak_runtime_min<{args.min_baseline_peak_runtime_min:g}"
        )
    if float(row["comfort_violation_min"]) > args.max_baseline_mode_violation_min:
        reasons.append(
            f"mode_comfort_min>{args.max_baseline_mode_violation_min:g}"
        )
    if (
        float(row["comfort_violation_degree_min"])
        > args.max_baseline_mode_violation_degree_min
    ):
        reasons.append(
            "mode_comfort_degree_min"
            f">{args.max_baseline_mode_violation_degree_min:g}"
        )
    if (
        float(row["baseline_band_violation_frac"])
        > args.max_baseline_band_violation_frac
    ):
        reasons.append(
            "band_violation_frac"
            f">{args.max_baseline_band_violation_frac:g}"
        )
    return reasons


def expand_controller_grid(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        for controller in CONTROLLERS_BY_MODE[row["mode"]]:
            data = {column: row[column] for column in CELL_COLUMNS}
            data["controller"] = controller
            rows.append(data)
    if not rows:
        return pd.DataFrame(columns=[*CELL_COLUMNS, "controller"])
    return pd.DataFrame(rows)


def write_command_file(
    controller_grid: pd.DataFrame,
    path: Path,
    evaluation_out_dir: str,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for _, row in controller_grid.iterrows():
        command = [
            "python",
            "scripts/run_mpc_evaluation.py",
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
            evaluation_out_dir,
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
        lines.append(format_command(command))
    lines.append("")
    lines.append(f"python scripts/analyze_mpc_eval.py {evaluation_out_dir}")
    lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(0o755)


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
