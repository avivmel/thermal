from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.screen_mpc_cells import build_screening_summary, clock_minute, mask_time_window


def test_clock_minute_parses_hh_mm() -> None:
    assert clock_minute("00:00") == 0
    assert clock_minute("17:30") == 1050


def test_mask_time_window_handles_overnight_window() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2017-01-01 21:59",
                "2017-01-01 22:00",
                "2017-01-01 23:30",
                "2017-01-02 01:59",
                "2017-01-02 02:00",
            ]
        )
    )

    assert mask_time_window(timestamps, "22:00", "02:00").tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_build_screening_summary_ranks_by_baseline_peak_load(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "controller": "baseline",
                "building": "large_hot_ac",
                "start_date": "2017-07-28",
                "mode": "cooling",
                "peak_start": "17:00",
                "peak_end": "20:00",
                "peak_energy_kwh": 1.0,
                "peak_runtime_min": 90.0,
                "total_energy_kwh": 2.0,
            },
            {
                "controller": "baseline",
                "building": "large_hot_ac",
                "start_date": "2017-07-29",
                "mode": "cooling",
                "peak_start": "17:00",
                "peak_end": "20:00",
                "peak_energy_kwh": 2.0,
                "peak_runtime_min": 180.0,
                "total_energy_kwh": 4.0,
            },
            {
                "controller": "mpc",
                "building": "large_hot_ac",
                "start_date": "2017-07-29",
                "mode": "cooling",
                "peak_start": "17:00",
                "peak_end": "20:00",
                "peak_energy_kwh": 0.5,
                "peak_runtime_min": 15.0,
                "total_energy_kwh": 4.2,
            },
        ]
    )
    path = tmp_path / "metrics.parquet"
    metrics.to_parquet(path, index=False)

    summary = build_screening_summary(
        metrics_path=path,
        buildings=["large_hot_ac"],
        dates=["2017-07-28", "2017-07-29"],
        mode="cooling",
        peak_start="17:00",
        peak_end="20:00",
    )

    assert summary["start_date"].tolist() == ["2017-07-29", "2017-07-28"]
    assert summary["controller"].tolist() == ["baseline", "baseline"]

