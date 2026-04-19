from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_open_exogenous import (  # noqa: E402
    DEFAULT_BAKERY_LATITUDE,
    DEFAULT_BAKERY_LONGITUDE,
    DEFAULT_BAKERY_SCHOOL_ZONE,
    M5_STATE_PROXY_SCHOOL,
    M5_STATE_PROXY_WEATHER,
    build_location_metadata_frame,
    compute_country_date_bounds,
    compute_country_years,
    fetch_weather_frame,
    fetch_school_holidays_frame,
    parse_school_holiday_ics,
)


class FetchOpenExogenousTests(unittest.TestCase):
    @staticmethod
    def _sample_silver_locations() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"dataset_source": "m5", "location_id": "CA_1", "region_code": "CA", "min_dt": "2011-01-29", "max_dt": "2011-02-02"},
                {"dataset_source": "bakery", "location_id": "bakery_store_1", "region_code": None, "min_dt": "2021-01-01", "max_dt": "2021-01-03"},
                {"dataset_source": "freshretail", "location_id": "42", "region_code": "31", "min_dt": "2024-03-28", "max_dt": "2024-04-02"},
            ]
        )

    @staticmethod
    def _sample_two_m5_locations() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dataset_source": "m5",
                    "location_id": "CA_1",
                    "region_code": "CA",
                    "min_dt": "2011-01-29",
                    "max_dt": "2011-01-30",
                },
                {
                    "dataset_source": "m5",
                    "location_id": "TX_1",
                    "region_code": "TX",
                    "min_dt": "2011-01-29",
                    "max_dt": "2011-01-30",
                },
            ]
        )

    @staticmethod
    def _batched_weather_payload() -> list[dict[str, object]]:
        return [
            {
                "daily": {
                    "time": ["2011-01-29", "2011-01-30"],
                    "temperature_2m_mean": [10.0, 11.0],
                    "temperature_2m_min": [5.0, 6.0],
                    "temperature_2m_max": [15.0, 16.0],
                    "precipitation_sum": [0.0, 1.0],
                    "relative_humidity_2m_mean": [0.5, 0.6],
                    "wind_speed_10m_mean": [1.0, 2.0],
                }
            },
            {
                "daily": {
                    "time": ["2011-01-29", "2011-01-30"],
                    "temperature_2m_mean": [20.0, 21.0],
                    "temperature_2m_min": [15.0, 16.0],
                    "temperature_2m_max": [25.0, 26.0],
                    "precipitation_sum": [2.0, 3.0],
                    "relative_humidity_2m_mean": [0.7, 0.8],
                    "wind_speed_10m_mean": [3.0, 4.0],
                }
            },
        ]

    def test_build_location_metadata_frame_applies_dataset_specific_assumptions(self) -> None:
        silver_locations = self._sample_silver_locations()

        metadata = build_location_metadata_frame(
            silver_locations,
            bakery_city_name="Paris",
            bakery_school_zone=DEFAULT_BAKERY_SCHOOL_ZONE,
            bakery_latitude=DEFAULT_BAKERY_LATITUDE,
            bakery_longitude=DEFAULT_BAKERY_LONGITUDE,
        )

        self.assertEqual(len(metadata), 3)

        m5_row = metadata.loc[metadata["dataset_source"].eq("m5")].iloc[0]
        self.assertEqual(m5_row["country_code"], "US")
        self.assertEqual(m5_row["city_name"], M5_STATE_PROXY_WEATHER["CA"]["city_name"])
        self.assertEqual(m5_row["school_zone"], M5_STATE_PROXY_SCHOOL["CA"]["school_zone"])
        self.assertIn("m5_state_capital_proxy_school_calendar", m5_row["assumption_source"])

        bakery_row = metadata.loc[metadata["dataset_source"].eq("bakery")].iloc[0]
        self.assertEqual(bakery_row["country_code"], "FR")
        self.assertEqual(bakery_row["school_zone"], DEFAULT_BAKERY_SCHOOL_ZONE)
        self.assertAlmostEqual(float(bakery_row["latitude"]), DEFAULT_BAKERY_LATITUDE)

        fresh_row = metadata.loc[metadata["dataset_source"].eq("freshretail")].iloc[0]
        self.assertEqual(fresh_row["country_code"], "CN")
        self.assertTrue(pd.isna(fresh_row["latitude"]))
        self.assertEqual(fresh_row["assumption_source"], "freshretail_city_id_without_public_geocoding")

    def test_compute_country_years_includes_target_day_spillover(self) -> None:
        silver_locations = pd.DataFrame(
            [
                {
                    "dataset_source": "m5",
                    "location_id": "CA_1",
                    "region_code": "CA",
                    "min_dt": "2016-12-30",
                    "max_dt": "2016-12-31",
                }
            ]
        )
        metadata = pd.DataFrame(
            [
                {
                    "dataset_source": "m5",
                    "location_id": "CA_1",
                    "country_code": "US",
                }
            ]
        )

        country_years = compute_country_years(silver_locations, metadata)

        self.assertEqual(country_years, {"US": [2016, 2017]})

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

        self.assertEqual(bounds["US"], ("2010-12-30", "2011-02-03"))
        self.assertEqual(bounds["FR"], ("2020-12-02", "2021-01-04"))
        self.assertEqual(bounds["CN"], ("2024-02-27", "2024-04-03"))

    def test_parse_school_holiday_ics_expands_each_event_into_daily_rows(self) -> None:
        ics_text = "\n".join(
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

        frame = parse_school_holiday_ics(
            ics_text,
            dataset_source="bakery",
            location_id="bakery_store_1",
            school_zone="C",
        )

        self.assertEqual(frame["dt"].tolist(), ["2024-01-01", "2024-01-02"])
        self.assertTrue((frame["dataset_source"] == "bakery").all())
        self.assertTrue((frame["location_id"] == "bakery_store_1").all())

    def test_fetch_school_holidays_frame_includes_m5_proxy_school_calendar(self) -> None:
        silver_locations = pd.DataFrame(
            [
                {
                    "dataset_source": "m5",
                    "location_id": "CA_1",
                    "region_code": "CA",
                    "min_dt": "2013-03-20",
                    "max_dt": "2013-03-29",
                }
            ]
        )
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
            http_text_reader=lambda **_: "",
        )

        self.assertEqual(frame["dataset_source"].unique().tolist(), ["m5"])
        self.assertEqual(frame["location_id"].unique().tolist(), ["CA_1"])
        self.assertEqual(frame["dt"].tolist(), ["2013-03-25", "2013-03-26", "2013-03-27", "2013-03-28", "2013-03-29"])
        self.assertTrue((frame["school_holiday_name"] == "Spring Break").all())

    def test_fetch_weather_frame_batches_multiple_locations_in_one_request(self) -> None:
        silver_locations = self._sample_two_m5_locations()
        metadata = build_location_metadata_frame(
            silver_locations,
            bakery_city_name="Paris",
            bakery_school_zone=DEFAULT_BAKERY_SCHOOL_ZONE,
            bakery_latitude=DEFAULT_BAKERY_LATITUDE,
            bakery_longitude=DEFAULT_BAKERY_LONGITUDE,
        )
        requested_urls: list[str] = []

        def _http_json_reader(url: str, **_: object) -> object:
            requested_urls.append(url)
            return self._batched_weather_payload()

        frame = fetch_weather_frame(
            metadata,
            silver_locations,
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
            max_locations_per_request=10,
            inter_batch_sleep_seconds=0.0,
            archive_url="https://archive-api.open-meteo.com/v1/archive",
            http_json_reader=_http_json_reader,
        )

        self.assertEqual(len(requested_urls), 1)
        self.assertIn("latitude=38.5816%2C30.2672", requested_urls[0])
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame["location_id"].tolist(), ["CA_1", "CA_1", "TX_1", "TX_1"])
        self.assertEqual(frame["weather_temperature_mean"].tolist(), [10.0, 11.0, 20.0, 21.0])


if __name__ == "__main__":
    unittest.main()
