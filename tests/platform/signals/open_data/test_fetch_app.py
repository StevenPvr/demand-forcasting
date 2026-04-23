from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.signals.open_data.fetch_app import (  # noqa: E402
    DEFAULT_BAKERY_LATITUDE,
    DEFAULT_BAKERY_LONGITUDE,
    DEFAULT_BAKERY_SCHOOL_ZONE,
    build_location_metadata_frame,
    compute_country_date_bounds,
    compute_country_years,
    fetch_school_holidays_frame,
    fetch_weather_frame,
    parse_school_holiday_ics,
)


def _bakery_school_ics(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    del url, timeout_seconds, max_retries, retry_backoff_seconds
    return "\n".join(
        [
            "BEGIN:VCALENDAR",
            "BEGIN:VEVENT",
            "DTSTART;VALUE=DATE:20240101",
            "DTEND;VALUE=DATE:20240103",
            "SUMMARY:Vacances d'hiver",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
    )


def _weather_payload_reader(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, object]:
    del url, timeout_seconds, max_retries, retry_backoff_seconds
    return {
        "daily": {
            "time": ["2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"],
            "temperature_2m_mean": [10.0, 11.0, 12.0, 13.0],
            "temperature_2m_min": [5.0, 6.0, 7.0, 8.0],
            "temperature_2m_max": [15.0, 16.0, 17.0, 18.0],
            "precipitation_sum": [0.0, 1.0, 0.0, 2.0],
            "relative_humidity_2m_mean": [0.5, 0.6, 0.55, 0.65],
            "wind_speed_10m_mean": [1.0, 2.0, 1.5, 2.5],
        }
    }


class FetchOpenExogenousTests(unittest.TestCase):
    @staticmethod
    def _sample_silver_locations() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dataset_source": "bakery",
                    "location_id": "bakery_store_1",
                    "region_code": None,
                    "min_dt": "2023-12-30",
                    "max_dt": "2024-01-02",
                },
                {
                    "dataset_source": "freshretail",
                    "location_id": "42",
                    "region_code": "31",
                    "min_dt": "2024-03-28",
                    "max_dt": "2024-04-02",
                },
                {
                    "dataset_source": "freshretail_lt",
                    "location_id": "84",
                    "region_code": None,
                    "min_dt": "2024-02-10",
                    "max_dt": "2024-02-14",
                },
            ]
        )

    @staticmethod
    def _weather_payload() -> dict[str, object]:
        return _weather_payload_reader(
            "",
            timeout_seconds=0,
            max_retries=0,
            retry_backoff_seconds=0.0,
        )

    def test_build_location_metadata_frame_applies_current_dataset_assumptions(self) -> None:
        metadata = build_location_metadata_frame(
            self._sample_silver_locations(),
            bakery_city_name="Paris",
            bakery_school_zone=DEFAULT_BAKERY_SCHOOL_ZONE,
            bakery_latitude=DEFAULT_BAKERY_LATITUDE,
            bakery_longitude=DEFAULT_BAKERY_LONGITUDE,
        )

        self.assertEqual(sorted(metadata["dataset_source"].tolist()), ["bakery", "freshretail", "freshretail_lt"])

        bakery_row = metadata.loc[metadata["dataset_source"].eq("bakery")].iloc[0]
        self.assertEqual(bakery_row["country_code"], "FR")
        self.assertEqual(bakery_row["school_zone"], DEFAULT_BAKERY_SCHOOL_ZONE)
        self.assertAlmostEqual(float(bakery_row["latitude"]), DEFAULT_BAKERY_LATITUDE)
        self.assertEqual(bakery_row["site_format"], "bakery")
        self.assertEqual(bakery_row["service_model"], "counter_service")

        freshretail_row = metadata.loc[metadata["dataset_source"].eq("freshretail")].iloc[0]
        self.assertEqual(freshretail_row["country_code"], "CN")
        self.assertTrue(pd.isna(freshretail_row["latitude"]))
        self.assertEqual(freshretail_row["assumption_source"], "freshretail_city_id_without_public_geocoding")

        freshretail_lt_row = metadata.loc[metadata["dataset_source"].eq("freshretail_lt")].iloc[0]
        self.assertEqual(freshretail_lt_row["country_code"], "CN")
        self.assertEqual(freshretail_lt_row["site_format"], "grocery")
        self.assertEqual(freshretail_lt_row["service_model"], "store_pick_pack")

    def test_build_location_metadata_frame_returns_empty_contract_for_empty_input(
        self,
    ) -> None:
        metadata = build_location_metadata_frame(
            pd.DataFrame(columns=["dataset_source", "location_id", "region_code"]),
            bakery_city_name="Paris",
            bakery_school_zone=DEFAULT_BAKERY_SCHOOL_ZONE,
            bakery_latitude=DEFAULT_BAKERY_LATITUDE,
            bakery_longitude=DEFAULT_BAKERY_LONGITUDE,
        )

        self.assertTrue(metadata.empty)
        self.assertIn("dataset_source", metadata.columns)
        self.assertIn("location_id", metadata.columns)
        self.assertIn("country_code", metadata.columns)

    def test_compute_country_years_includes_target_day_spillover(self) -> None:
        silver_locations = pd.DataFrame(
            [
                {
                    "dataset_source": "bakery",
                    "location_id": "bakery_store_1",
                    "region_code": None,
                    "min_dt": "2023-12-31",
                    "max_dt": "2023-12-31",
                }
            ]
        )
        metadata = pd.DataFrame(
            [
                {
                    "dataset_source": "bakery",
                    "location_id": "bakery_store_1",
                    "country_code": "FR",
                }
            ]
        )

        self.assertEqual(compute_country_years(silver_locations, metadata), {"FR": [2023, 2024]})

    def test_compute_country_date_bounds_adds_lookback_window_per_country(self) -> None:
        silver_locations = self._sample_silver_locations()
        metadata = build_location_metadata_frame(
            silver_locations,
            bakery_city_name="Paris",
            bakery_school_zone=DEFAULT_BAKERY_SCHOOL_ZONE,
            bakery_latitude=DEFAULT_BAKERY_LATITUDE,
            bakery_longitude=DEFAULT_BAKERY_LONGITUDE,
        )

        bounds = compute_country_date_bounds(silver_locations, metadata, lookback_days=30)

        self.assertEqual(bounds["FR"], ("2023-11-30", "2024-01-03"))
        self.assertEqual(bounds["CN"], ("2024-01-11", "2024-04-03"))

    def test_parse_school_holiday_ics_expands_each_event_into_daily_rows(self) -> None:
        frame = parse_school_holiday_ics(
            _bakery_school_ics(
                "",
                timeout_seconds=0,
                max_retries=0,
                retry_backoff_seconds=0.0,
            ),
            dataset_source="bakery",
            location_id="bakery_store_1",
            school_zone="C",
        )

        self.assertEqual(frame["dt"].tolist(), ["2024-01-01", "2024-01-02"])
        self.assertTrue((frame["dataset_source"] == "bakery").all())

    def test_fetch_school_holidays_frame_returns_french_school_rows_only(self) -> None:
        silver_locations = self._sample_silver_locations()
        metadata = build_location_metadata_frame(
            silver_locations,
            bakery_city_name="Paris",
            bakery_school_zone=DEFAULT_BAKERY_SCHOOL_ZONE,
            bakery_latitude=DEFAULT_BAKERY_LATITUDE,
            bakery_longitude=DEFAULT_BAKERY_LONGITUDE,
        )

        frame = fetch_school_holidays_frame(
            metadata,
            silver_locations,
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
            http_text_reader=_bakery_school_ics,
        )

        self.assertEqual(frame["dataset_source"].unique().tolist(), ["bakery"])
        self.assertEqual(frame["dt"].tolist(), ["2024-01-01", "2024-01-02"])

    def test_fetch_weather_frame_returns_only_locations_with_coordinates(self) -> None:
        silver_locations = self._sample_silver_locations()
        location_metadata = pd.DataFrame(
            [
                {
                    "dataset_source": "bakery",
                    "location_id": "bakery_store_1",
                    "latitude": DEFAULT_BAKERY_LATITUDE,
                    "longitude": DEFAULT_BAKERY_LONGITUDE,
                },
                {
                    "dataset_source": "freshretail",
                    "location_id": "42",
                    "latitude": None,
                    "longitude": None,
                },
            ]
        )

        frame = fetch_weather_frame(
            location_metadata,
            silver_locations,
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
            max_locations_per_request=5,
            inter_batch_sleep_seconds=0.0,
            archive_url="https://archive-api.open-meteo.com/v1/archive",
            http_json_reader=_weather_payload_reader,
        )

        self.assertEqual(frame["dataset_source"].unique().tolist(), ["bakery"])
        self.assertEqual(len(frame), 4)
        self.assertIn("weather_temperature_mean", frame.columns)
        self.assertIn("weather_temperature_min", frame.columns)
        self.assertIn("weather_temperature_max", frame.columns)
        self.assertEqual(frame["weather_temperature_min"].tolist(), [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(frame["weather_temperature_max"].tolist(), [15.0, 16.0, 17.0, 18.0])


if __name__ == "__main__":
    unittest.main()
