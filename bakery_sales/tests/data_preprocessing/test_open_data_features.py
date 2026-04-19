from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.open_data_features import build_open_data_features


def test_build_open_data_features_uses_target_holidays_and_origin_weather(
    tmp_path: Path,
) -> None:
    dataset_df = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2021-07-12", "2021-07-13", "2021-09-01"]),
            "target_date": pd.to_datetime(["2021-07-13", "2021-07-14", "2021-09-02"]),
        }
    )
    holiday_calls: list[str] = []
    weather_calls: list[str] = []

    def fake_download_text(url: str) -> str:
        holiday_calls.append(url)
        return "\n".join(
            [
                "date,vacances_zone_a,vacances_zone_b,vacances_zone_c,nom_vacances",
                "2021-07-13,False,False,False,",
                "2021-07-14,False,True,False,Vacances d'ete",
                "2021-09-02,False,False,False,",
            ]
        )

    def fake_download_json(url: str) -> dict[str, object]:
        if "jours-feries" in url:
            return {
                "2021-07-14": "14 juillet",
                "2021-11-11": "11 novembre",
            }
        weather_calls.append(url)
        return {
            "daily": {
                "time": ["2021-07-12", "2021-07-13", "2021-09-01"],
                "temperature_2m_mean": [18.0, 26.0, 8.0],
                "temperature_2m_max": [22.0, 31.0, 10.0],
                "temperature_2m_min": [14.0, 20.0, 4.0],
                "apparent_temperature_mean": [17.0, 28.0, 6.0],
                "precipitation_sum": [0.0, 8.0, 1.0],
                "precipitation_hours": [0.0, 4.0, 1.0],
                "rain_sum": [0.0, 7.5, 1.0],
                "snowfall_sum": [0.0, 0.0, 0.0],
                "weather_code": [1, 61, 3],
                "wind_speed_10m_max": [15.0, 35.0, 10.0],
                "wind_gusts_10m_max": [25.0, 50.0, 15.0],
                "sunshine_duration": [36000.0, 18000.0, 25200.0],
                "shortwave_radiation_sum": [25.0, 10.0, 18.0],
                "cloud_cover_mean": [20.0, 80.0, 40.0],
                "relative_humidity_2m_mean": [55.0, 70.0, 65.0],
            }
        }

    feature_df = build_open_data_features(
        dataset_df=dataset_df,
        holiday_cache_path=tmp_path / "school_holidays.csv",
        public_holiday_cache_path=tmp_path / "public_holidays.csv",
        weather_cache_path=tmp_path / "weather.csv",
        school_holidays_url="https://example.test/school-holidays.csv",
        public_holidays_url="https://etalab.github.io/jours-feries-france-data/json/metropole.json",
        weather_api_base_url="https://example.test/weather",
        school_zone="B",
        latitude=47.32829,
        longitude=-2.42934,
        timezone_name="Europe/Paris",
        download_text=fake_download_text,
        download_json=fake_download_json,
    )

    assert holiday_calls == ["https://example.test/school-holidays.csv"]
    assert len(weather_calls) == 1
    assert "start_date=2021-07-12" in weather_calls[0]
    assert "end_date=2021-09-01" in weather_calls[0]
    assert feature_df["exog_holiday_target_is_school_holiday"].tolist() == [0, 1, 0]
    assert feature_df["exog_holiday_target_is_summer_school_holiday"].tolist() == [0, 1, 0]
    assert feature_df["exog_holiday_target_is_day_before_school_holiday"].tolist() == [1, 0, 0]
    assert feature_df["exog_holiday_target_is_public_holiday"].tolist() == [0, 1, 0]
    assert feature_df["exog_holiday_target_is_day_before_public_holiday"].tolist() == [1, 0, 0]
    assert feature_df["exog_holiday_target_is_day_after_public_holiday"].tolist() == [0, 0, 0]
    assert feature_df["exog_holiday_target_days_until_next_public_holiday"].tolist() == [1.0, 0.0, 70.0]
    assert feature_df["exog_weather_origin_temperature_2m_mean"].tolist() == [18.0, 26.0, 8.0]
    assert feature_df["exog_weather_origin_temperature_range"].tolist() == [8.0, 11.0, 6.0]
    assert feature_df["exog_weather_origin_precipitation_sum"].tolist() == [0.0, 8.0, 1.0]
    assert feature_df["exog_weather_origin_weather_code"].tolist() == [1, 61, 3]
    assert feature_df["exog_weather_origin_wind_speed_10m_max"].tolist() == [15.0, 35.0, 10.0]
    assert feature_df["exog_weather_origin_sunshine_duration_hours"].tolist() == [10.0, 5.0, 7.0]
    assert feature_df["exog_weather_origin_is_rainy_day"].tolist() == [0, 1, 1]
    assert feature_df["exog_weather_origin_is_heavy_rain_day"].tolist() == [0, 1, 0]
    assert feature_df["exog_weather_origin_is_hot_day"].tolist() == [0, 1, 0]
    assert feature_df["exog_weather_origin_is_sunny_day"].tolist() == [1, 0, 0]
    assert feature_df["exog_weather_origin_is_windy_day"].tolist() == [0, 1, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
