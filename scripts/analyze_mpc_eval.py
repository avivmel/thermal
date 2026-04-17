#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

RELATIVE_METRIC_COLUMNS = [
    "peak_runtime_reduction_pct",
    "peak_energy_reduction_pct",
    "energy_overhead_pct",
    "cost_savings_pct",
    "peak_cost_savings_pct",
    "comfort_violation_delta_min",
    "comfort_degree_min_delta",
    "comfort_band_violation_delta_min",
    "comfort_band_degree_min_delta",
]

SUMMARY_METRICS = [
    "peak_runtime_reduction_pct",
    "peak_energy_reduction_pct",
    "energy_overhead_pct",
    "cost_savings_pct",
    "comfort_violation_delta_min",
    "comfort_band_violation_delta_min",
    "max_violation_f",
    "max_too_cold_f",
    "max_too_hot_f",
]

RULE_CONTROLLERS = {"setback", "precool", "preheat"}


def main() -> None:
    args = parse_args()
    metrics_paths = resolve_metrics_paths(args.inputs)
    metrics = read_metrics(metrics_paths)
    metrics = add_source_columns(metrics, metrics_paths)
    metrics = add_relative_metrics(metrics)

    out_dir = resolve_output_dir(args, metrics_paths)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cell_metrics = make_cell_controller_table(metrics)
    controller_summary = summarize_controllers(metrics, seed=args.seed, n_boot=args.bootstrap)
    best_rule = compare_mpc_to_rules(metrics)
    main_results = make_main_results(controller_summary)

    cell_metrics.to_csv(tables_dir / "cell_controller_metrics.csv", index=False)
    controller_summary.to_csv(tables_dir / "controller_summary.csv", index=False)
    main_results.to_csv(tables_dir / "main_results.csv", index=False)
    best_rule.to_csv(tables_dir / "mpc_vs_rule_comparison.csv", index=False)

    report = render_report(
        metrics=metrics,
        cell_metrics=cell_metrics,
        controller_summary=controller_summary,
        best_rule=best_rule,
        metrics_paths=metrics_paths,
    )
    (out_dir / "report.md").write_text(report)

    print(f"Read {len(metrics)} metric rows from {len(metrics_paths)} file(s)")
    print(f"Wrote analysis: {out_dir}")
    print(controller_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MPC evaluation metrics and emit paper-oriented tables."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Evaluation directories or metrics.parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        help="Output analysis directory. Defaults to <eval_dir>/analysis for one input.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap samples for mean confidence intervals. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_metrics_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            path = path / "metrics.parquet"
        if not path.exists():
            raise SystemExit(f"metrics file not found: {path}")
        paths.append(path)
    return paths


def read_metrics(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["_metrics_path"] = str(path)
        frame["_eval_dir"] = str(path.parent)
        frames.append(frame)
    if not frames:
        raise SystemExit("no metrics files provided")
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    if metrics.empty:
        raise SystemExit("metrics files contain no rows")
    if "controller" not in metrics.columns:
        raise SystemExit("metrics file is missing required column: controller")
    return metrics


def add_source_columns(metrics: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    result = metrics.copy()
    if len(paths) == 1:
        result["eval_name"] = paths[0].parent.name
    else:
        result["eval_name"] = result["_eval_dir"].map(lambda value: Path(value).name)
    return result


def resolve_output_dir(args: argparse.Namespace, paths: list[Path]) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    if len(paths) == 1:
        return paths[0].parent / "analysis"
    return Path("results/mpc_eval/analysis")


def make_cell_controller_table(metrics: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "eval_name",
        "cell_id",
        "controller",
        "building",
        "start_date",
        "run_period_days",
        "mode",
        "peak_start",
        "peak_end",
        "forecast_kind",
        "peak_runtime_min",
        "peak_runtime_reduction_pct",
        "peak_energy_kwh",
        "peak_energy_reduction_pct",
        "total_energy_kwh",
        "energy_overhead_pct",
        "cost_usd",
        "cost_savings_pct",
        "comfort_violation_min",
        "comfort_violation_delta_min",
        "comfort_violation_degree_min",
        "comfort_band_violation_min",
        "comfort_band_violation_delta_min",
        "comfort_band_violation_degree_min",
        "comfort_too_cold_min",
        "comfort_too_hot_min",
        "max_violation_f",
        "max_too_cold_f",
        "max_too_hot_f",
        "decisions_phase_precondition_count",
        "decisions_phase_peak_coast_count",
        "decisions_phase_peak_maintain_count",
    ]
    columns = [column for column in preferred if column in metrics.columns]
    sort_columns = [column for column in ["eval_name", "building", "start_date", "mode", "controller"] if column in columns]
    return metrics[columns].sort_values(sort_columns).reset_index(drop=True)


def summarize_controllers(metrics: pd.DataFrame, seed: int, n_boot: int) -> pd.DataFrame:
    group_columns = [column for column in ["eval_name", "mode", "controller"] if column in metrics.columns]
    rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row["n_cells"] = int(len(group))
        row["n_baseline_matched_cells"] = int(group["peak_energy_reduction_pct"].notna().sum())
        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{metric}_iqr"] = iqr(values)
            if n_boot > 0:
                low, high = bootstrap_mean_ci(values, seed=seed, n_boot=n_boot)
                row[f"{metric}_mean_ci_low"] = low
                row[f"{metric}_mean_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def make_main_results(controller_summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "eval_name",
        "mode",
        "controller",
        "n_cells",
        "peak_energy_reduction_pct_median",
        "peak_energy_reduction_pct_mean",
        "peak_energy_reduction_pct_mean_ci_low",
        "peak_energy_reduction_pct_mean_ci_high",
        "peak_runtime_reduction_pct_median",
        "energy_overhead_pct_median",
        "comfort_violation_delta_min_median",
        "comfort_band_violation_delta_min_median",
        "max_violation_f_median",
    ]
    columns = [column for column in keep if column in controller_summary.columns]
    return controller_summary[columns].copy()


def compare_mpc_to_rules(metrics: pd.DataFrame) -> pd.DataFrame:
    match_columns = available_match_columns(metrics)
    if not match_columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, group in metrics.groupby(match_columns, dropna=False, sort=True):
        mpc = group[group["controller"] == "mpc"]
        rules = group[group["controller"].isin(RULE_CONTROLLERS)]
        if mpc.empty or rules.empty:
            continue
        mpc_row = mpc.iloc[0]
        feasible_rules = rules[rules["comfort_violation_delta_min"].fillna(np.inf) <= 0.0]
        best_feasible = pick_best_rule(feasible_rules)
        best_any = pick_best_rule(rules)

        row = {column: mpc_row[column] for column in match_columns}
        row["mpc_cell_id"] = mpc_row.get("cell_id")
        row["mpc_peak_energy_reduction_pct"] = mpc_row.get("peak_energy_reduction_pct")
        row["mpc_peak_runtime_reduction_pct"] = mpc_row.get("peak_runtime_reduction_pct")
        row["mpc_energy_overhead_pct"] = mpc_row.get("energy_overhead_pct")
        row["mpc_comfort_violation_delta_min"] = mpc_row.get("comfort_violation_delta_min")
        row["mpc_comfort_band_violation_delta_min"] = mpc_row.get(
            "comfort_band_violation_delta_min"
        )
        add_rule_comparison(row, "best_feasible_rule", best_feasible, mpc_row)
        add_rule_comparison(row, "best_any_rule", best_any, mpc_row)
        rows.append(row)
    return pd.DataFrame(rows)


def pick_best_rule(rules: pd.DataFrame) -> pd.Series | None:
    if rules.empty:
        return None
    sort_columns = [column for column in ["peak_energy_reduction_pct", "peak_runtime_reduction_pct"] if column in rules.columns]
    if not sort_columns:
        return rules.iloc[0]
    return rules.sort_values(sort_columns, ascending=False).iloc[0]


def add_rule_comparison(
    row: dict[str, Any],
    prefix: str,
    rule: pd.Series | None,
    mpc: pd.Series,
) -> None:
    if rule is None:
        row[f"{prefix}_controller"] = None
        return
    row[f"{prefix}_controller"] = rule.get("controller")
    row[f"{prefix}_peak_energy_reduction_pct"] = rule.get("peak_energy_reduction_pct")
    row[f"{prefix}_peak_runtime_reduction_pct"] = rule.get("peak_runtime_reduction_pct")
    row[f"{prefix}_energy_overhead_pct"] = rule.get("energy_overhead_pct")
    row[f"{prefix}_comfort_violation_delta_min"] = rule.get("comfort_violation_delta_min")
    row[f"{prefix}_comfort_band_violation_delta_min"] = rule.get(
        "comfort_band_violation_delta_min"
    )
    row[f"mpc_minus_{prefix}_peak_energy_reduction_pct"] = safe_float(
        mpc.get("peak_energy_reduction_pct")
    ) - safe_float(rule.get("peak_energy_reduction_pct"))
    row[f"mpc_minus_{prefix}_energy_overhead_pct"] = safe_float(
        mpc.get("energy_overhead_pct")
    ) - safe_float(rule.get("energy_overhead_pct"))
    row[f"mpc_minus_{prefix}_comfort_violation_delta_min"] = safe_float(
        mpc.get("comfort_violation_delta_min")
    ) - safe_float(rule.get("comfort_violation_delta_min"))


def add_relative_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    for column in RELATIVE_METRIC_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    match_columns = available_match_columns(result)
    if not match_columns:
        return result

    grouped = result.groupby(match_columns, dropna=False, sort=False)
    for _, index in grouped.groups.items():
        group = result.loc[index]
        baselines = group[group["controller"] == "baseline"]
        if baselines.empty:
            continue
        baseline = baselines.iloc[0]
        for row_index in group.index:
            row = result.loc[row_index]
            result.loc[row_index, "peak_runtime_reduction_pct"] = reduction_pct(
                baseline.get("peak_runtime_min"), row.get("peak_runtime_min")
            )
            result.loc[row_index, "peak_energy_reduction_pct"] = reduction_pct(
                baseline.get("peak_energy_kwh"), row.get("peak_energy_kwh")
            )
            result.loc[row_index, "energy_overhead_pct"] = increase_pct(
                baseline.get("total_energy_kwh"), row.get("total_energy_kwh")
            )
            result.loc[row_index, "cost_savings_pct"] = reduction_pct(
                baseline.get("cost_usd"), row.get("cost_usd")
            )
            result.loc[row_index, "peak_cost_savings_pct"] = reduction_pct(
                baseline.get("peak_cost_usd"), row.get("peak_cost_usd")
            )
            result.loc[row_index, "comfort_violation_delta_min"] = difference(
                row.get("comfort_violation_min"), baseline.get("comfort_violation_min")
            )
            result.loc[row_index, "comfort_degree_min_delta"] = difference(
                row.get("comfort_violation_degree_min"),
                baseline.get("comfort_violation_degree_min"),
            )
            result.loc[row_index, "comfort_band_violation_delta_min"] = difference(
                row.get("comfort_band_violation_min"),
                baseline.get("comfort_band_violation_min"),
            )
            result.loc[row_index, "comfort_band_degree_min_delta"] = difference(
                row.get("comfort_band_violation_degree_min"),
                baseline.get("comfort_band_violation_degree_min"),
            )
    return result


def available_match_columns(metrics: pd.DataFrame) -> list[str]:
    return [column for column in BASELINE_MATCH_COLUMNS if column in metrics.columns]


def reduction_pct(baseline: Any, value: Any) -> float:
    baseline_value = safe_float(baseline)
    actual_value = safe_float(value)
    if np.isnan(baseline_value) or np.isnan(actual_value):
        return np.nan
    if baseline_value == 0.0:
        return 0.0 if actual_value == 0.0 else np.nan
    return (baseline_value - actual_value) / baseline_value * 100.0


def increase_pct(baseline: Any, value: Any) -> float:
    baseline_value = safe_float(baseline)
    actual_value = safe_float(value)
    if np.isnan(baseline_value) or np.isnan(actual_value):
        return np.nan
    if baseline_value == 0.0:
        return 0.0 if actual_value == 0.0 else np.nan
    return (actual_value - baseline_value) / baseline_value * 100.0


def difference(value: Any, baseline: Any) -> float:
    actual_value = safe_float(value)
    baseline_value = safe_float(baseline)
    if np.isnan(actual_value) or np.isnan(baseline_value):
        return np.nan
    return actual_value - baseline_value


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result


def iqr(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def render_report(
    metrics: pd.DataFrame,
    cell_metrics: pd.DataFrame,
    controller_summary: pd.DataFrame,
    best_rule: pd.DataFrame,
    metrics_paths: list[Path],
) -> str:
    lines = [
        "# MPC Evaluation Analysis",
        "",
        "Generated from:",
    ]
    for path in metrics_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            f"Rows analyzed: {len(metrics)}",
            f"Unique evaluation groups: {count_eval_groups(metrics)}",
            "",
            "## Controller Summary",
            "",
            markdown_table(
                make_main_results(controller_summary),
                max_rows=40,
            ),
            "",
            "## Cell-Level Metrics",
            "",
            markdown_table(cell_metrics, max_rows=40),
        ]
    )
    if not best_rule.empty:
        lines.extend(
            [
                "",
                "## MPC vs Rule Baselines",
                "",
                markdown_table(best_rule, max_rows=40),
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `tables/cell_controller_metrics.csv`",
            "- `tables/controller_summary.csv`",
            "- `tables/main_results.csv`",
            "- `tables/mpc_vs_rule_comparison.csv`",
            "",
            "Notes:",
            "- `comfort_violation_*` is mode-aware: too-cold minutes for heating, too-hot minutes for cooling.",
            "- `comfort_band_violation_*` counts either side of the comfort band and is useful as a diagnostic.",
            "- Confidence intervals are bootstrap intervals over available cells; single-cell summaries have zero-width intervals.",
        ]
    )
    return "\n".join(lines) + "\n"


def count_eval_groups(metrics: pd.DataFrame) -> int:
    match_columns = available_match_columns(metrics)
    if not match_columns:
        return 0
    return int(metrics[match_columns].drop_duplicates().shape[0])


def markdown_table(frame: pd.DataFrame, max_rows: int) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(format_float)
    return render_markdown_table(display)


def render_markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rows.append([markdown_cell(row[column]) for column in frame.columns])

    widths = [len(column) for column in columns]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header = "| " + " | ".join(pad(value, widths[index]) for index, value in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(pad(value, widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def pad(value: str, width: int) -> str:
    return value + " " * (width - len(value))


def format_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
