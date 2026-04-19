from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.open_exogenous_macro_timeseries import (  # noqa: E402
    MacroSeriesSpec,
    build_effective_from,
    fetch_fred_macro_series_frame,
    fetch_macro_timeseries_frame,
)


class OpenExogenousMacroTimeseriesTests(unittest.TestCase):
    def test_build_effective_from_respects_frequency_specific_period_end(self) -> None:
        monthly = build_effective_from(
            pd.Timestamp("2024-03-01"),
            frequency="monthly",
            availability_lag_days=20,
        )
        quarterly = build_effective_from(
            pd.Timestamp("2024-01-01"),
            frequency="quarterly",
            availability_lag_days=45,
        )
        daily = build_effective_from(
            pd.Timestamp("2024-03-15"),
            frequency="daily",
            availability_lag_days=0,
        )

        self.assertEqual(monthly, pd.Timestamp("2024-04-20"))
        self.assertEqual(quarterly, pd.Timestamp("2024-05-15"))
        self.assertEqual(daily, pd.Timestamp("2024-03-15"))

    def test_fetch_fred_macro_series_frame_parses_public_csv_and_computes_effective_dates(self) -> None:
        spec = MacroSeriesSpec(
            country_code="US",
            metric_name="inflation_cpi_latest",
            source_series_id="CPIAUCSL",
            source_frequency="monthly",
            availability_lag_days=20,
            source_name="fred_stlouis_fed",
            metric_units="index",
        )
        csv_text = "\n".join(
            [
                "observation_date,CPIAUCSL",
                "2024-01-01,306.746",
                "2024-02-01,307.671",
                "2024-03-01,.",
            ]
        )

        frame = fetch_fred_macro_series_frame(
            spec,
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
            http_text_reader=lambda *_args, **_kwargs: csv_text,
        )

        self.assertEqual(frame["metric_name"].unique().tolist(), ["inflation_cpi_latest"])
        self.assertEqual(frame["observation_date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-01", "2024-02-01"])
        self.assertEqual(frame["period_end"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-31", "2024-02-29"])
        self.assertEqual(frame["effective_from"].dt.strftime("%Y-%m-%d").tolist(), ["2024-02-20", "2024-03-20"])
        self.assertEqual(frame["metric_value"].tolist(), [306.746, 307.671])

    def test_fetch_macro_timeseries_frame_filters_requested_countries(self) -> None:
        def fake_reader(url: str, **_kwargs: object) -> str:
            if "CPIAUCSL" in url:
                return "observation_date,CPIAUCSL\n2024-01-01,306.746\n"
            if "CP0000FRM086NEST" in url:
                return "observation_date,CP0000FRM086NEST\n2024-01-01,116.2\n"
            return "observation_date,series\n"

        frame = fetch_macro_timeseries_frame(
            country_codes=["FR"],
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
            http_text_reader=fake_reader,
        )

        self.assertTrue((frame["country_code"] == "FR").all())
        self.assertIn("inflation_cpi_latest", frame["metric_name"].unique().tolist())
        self.assertNotIn("US", frame["country_code"].unique().tolist())


if __name__ == "__main__":
    unittest.main()
