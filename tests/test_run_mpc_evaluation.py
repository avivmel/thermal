from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_mpc_evaluation import (
    compute_metrics,
    mask_time_window,
    parse_clock_time,
)


def test_parse_clock_time_validates_hh_mm() -> None:
    assert parse_clock_time("07:30") == (7, 30)
    with pytest.raises(SystemExit):
        parse_clock_time("24:00")


def test_mask_time_window_handles_same_day_and_overnight_windows() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2017-01-01 06:59",
                "2017-01-01 07:00",
                "2017-01-01 09:59",
                "2017-01-01 10:00",
                "2017-01-01 23:00",
            ]
        )
    )

    assert mask_time_window(timestamps, "07:00", "10:00").tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert mask_time_window(timestamps, "22:00", "02:00").tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_compute_metrics_matches_mpc_plan_names() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2017-01-01 15:00",
                    "2017-01-01 15:15",
                    "2017-01-01 17:00",
                    "2017-01-01 17:15",
                ]
            ),
            "hvac_mode": ["heating", "off", "heating", "off"],
            "hvac_power_kw": [2.0, 0.0, 4.0, 0.0],
            "indoor_temp": [67.0, 68.5, 73.0, 70.0],
            "electricity_price": [0.15, 0.15, 0.45, 0.45],
        }
    )
    commands = pd.DataFrame({"phase": ["normal", "precondition", "peak_coast", "peak_coast"]})
    cell = {"cell_id": "cell-a", "building": "small_cold_heatpump"}

    metrics = compute_metrics(
        history=history,
        commands=commands,
        cell=cell,
        comfort_lower_f=68.0,
        comfort_upper_f=72.0,
        peak_start="17:00",
        peak_end="20:00",
    )

    assert metrics["cell_id"] == "cell-a"
    assert metrics["peak_runtime_min"] == 15.0
    assert metrics["pre_peak_runtime_min"] == 15.0
    assert metrics["peak_energy_kwh"] == 1.0
    assert metrics["total_energy_kwh"] == 1.5
    assert metrics["comfort_violation_min"] == 30.0
    assert metrics["comfort_violation_degree_min"] == 30.0
    assert metrics["max_violation_f"] == 1.0
    assert metrics["cost_usd"] == pytest.approx(0.525)
    assert metrics["peak_cost_usd"] == pytest.approx(0.45)
    assert metrics["decisions_phase_normal_count"] == 1
    assert metrics["decisions_phase_precondition_count"] == 1
    assert metrics["decisions_phase_peak_coast_count"] == 2
    assert metrics["decisions_phase_peak_maintain_count"] == 0

