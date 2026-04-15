from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mpc.forecast import EPWForecast, NoisyForecast, PerfectForecast, PersistenceForecast


def test_perfect_forecast_uses_nearest_prior_lookup() -> None:
    forecast = PerfectForecast(
        pd.Series(
            [40.0, 42.0, 45.0],
            index=pd.to_datetime(["2017-01-01 00:00", "2017-01-01 01:00", "2017-01-01 02:00"]),
        )
    )

    assert forecast.outdoor_temp_at(pd.Timestamp("2017-01-01 01:30")) == 42.0
    assert forecast.outdoor_temp_at(pd.Timestamp("2017-01-01 02:00")) == 45.0
    with pytest.raises(KeyError):
        forecast.outdoor_temp_at(pd.Timestamp("2016-12-31 23:59"))


def test_persistence_forecast_returns_constant_temperature() -> None:
    forecast = PersistenceForecast(51.5)

    assert forecast.outdoor_temp_at(pd.Timestamp("2017-01-01 00:00")) == 51.5
    assert forecast.outdoor_temp_at(pd.Timestamp("2017-07-01 12:00")) == 51.5


def test_epw_forecast_parses_dry_bulb_temperature_as_fahrenheit(tmp_path: Path) -> None:
    epw = tmp_path / "test.epw"
    epw.write_text(
        "\n".join(
            [
                "LOCATION,Test,NA,USA,TMY3,000000,0,0,0,0",
                "DESIGN CONDITIONS,0",
                "TYPICAL/EXTREME PERIODS,0",
                "GROUND TEMPERATURES,0",
                "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
                "COMMENTS 1,",
                "COMMENTS 2,",
                "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
                "1995,1,1,1,0,?9,10.0,5.0,50,101000",
                "1995,1,1,2,0,?9,12.0,5.0,50,101000",
            ]
        )
    )

    forecast = EPWForecast.from_epw(epw, year=2017)

    assert forecast.outdoor_temp_at(pd.Timestamp("2017-01-01 00:30")) == pytest.approx(50.0)
    assert forecast.outdoor_temp_at(pd.Timestamp("2017-01-01 01:00")) == pytest.approx(53.6)


def test_noisy_forecast_is_seeded_and_nearest_prior() -> None:
    base = PerfectForecast(
        pd.Series(
            [70.0, 71.0, 72.0],
            index=pd.to_datetime(["2017-01-01 00:00", "2017-01-01 01:00", "2017-01-01 02:00"]),
        )
    )
    first = NoisyForecast(base, ar1_phi=0.7, sigma_f=2.0, seed=42)
    second = NoisyForecast(base, ar1_phi=0.7, sigma_f=2.0, seed=42)
    different = NoisyForecast(base, ar1_phi=0.7, sigma_f=2.0, seed=7)

    ts = pd.Timestamp("2017-01-01 01:30")

    assert first.outdoor_temp_at(ts) == second.outdoor_temp_at(ts)
    assert first.outdoor_temp_at(ts) != different.outdoor_temp_at(ts)


def test_noisy_forecast_zero_sigma_matches_base() -> None:
    base = PerfectForecast(
        pd.Series([70.0], index=pd.to_datetime(["2017-01-01 00:00"]))
    )
    noisy = NoisyForecast(base, sigma_f=0.0, seed=42)

    assert noisy.outdoor_temp_at(pd.Timestamp("2017-01-01 00:30")) == 70.0

