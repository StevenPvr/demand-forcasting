from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data_preprocessing.temporal_features as temporal_features
from src.data_preprocessing.temporal_features import build_modeling_features


def test_build_modeling_features_creates_target_date_calendar_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_df = pd.DataFrame(
        {
            "origin_date": pd.date_range("2021-01-01", periods=40, freq="D").strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2021-01-02", periods=40, freq="D").strftime("%Y-%m-%d"),
            "exog_sales_1_lag1": [float(index) for index in range(40)],
            "exog_flag_origin_day_missing": [0] * 40,
            "exog_flag_target_day_missing": [0] * 40,
            "target_baguette_t_plus_1": [float(index + 10) for index in range(40)],
        }
    )

    monkeypatch.setattr(
        temporal_features,
        "build_open_data_features",
        lambda dataset_df: pd.DataFrame(
            {
                "exog_holiday_target_is_school_holiday": [0] * len(dataset_df),
                "exog_holiday_target_is_public_holiday": [0] * len(dataset_df),
                "exog_weather_origin_temperature_2m_mean": [8.5] * len(dataset_df),
                "exog_weather_origin_is_rainy_day": [1] * len(dataset_df),
                "exog_weather_origin_is_hot_day": [0] * len(dataset_df),
            },
            index=dataset_df.index,
        ),
    )

    modeling_df = build_modeling_features(dataset_df)

    assert len(modeling_df) == 10
    first_row = modeling_df.iloc[0]
    assert first_row["origin_date"] == "2021-01-31"
    assert first_row["target_date"] == "2021-02-01"
    assert first_row["exog_calendar_target_day_of_week"] == 0
    assert first_row["exog_calendar_target_is_month_end"] == 0
    assert first_row["exog_calendar_target_days_from_month_start"] == 0
    assert first_row["exog_calendar_target_days_to_month_end"] == 27
    assert first_row["exog_calendar_target_is_payday_window"] == 1
    assert first_row["exog_autoreg_target_lag_1"] == pytest.approx(39.0)
    assert first_row["exog_autoreg_target_lag_7"] == pytest.approx(33.0)
    assert first_row["exog_autoreg_target_lag_30"] == pytest.approx(10.0)
    assert first_row["exog_autoreg_target_roll_mean_7"] == pytest.approx(36.0)
    assert first_row["exog_autoreg_target_roll_mean_14"] == pytest.approx(32.5)
    assert first_row["exog_autoreg_target_same_weekday_roll_mean_4"] == pytest.approx(22.5)
    assert first_row["exog_autoreg_target_gap_lag_1_vs_roll_mean_7"] == pytest.approx(3.0)
    assert first_row["exog_autoreg_target_ratio_lag_1_vs_roll_mean_7"] == pytest.approx(39.0 / 36.0)
    assert "exog_calendar_target_month_sin" in modeling_df.columns
    assert "exog_calendar_target_week_of_year_cos" in modeling_df.columns
    assert "exog_calendar_target_day_of_year_sin_2" in modeling_df.columns
    assert "exog_calendar_target_day_of_year_cos_2" in modeling_df.columns
    assert "exog_calendar_target_day_of_year_sin_3" in modeling_df.columns
    assert "exog_calendar_target_day_of_year_cos_3" in modeling_df.columns
    assert "exog_holiday_target_is_school_holiday" in modeling_df.columns
    assert "exog_holiday_target_is_public_holiday" in modeling_df.columns
    assert "exog_weather_origin_temperature_2m_mean" in modeling_df.columns
    assert "exog_weather_origin_is_rainy_day_x_target_is_weekend" in modeling_df.columns
    assert "feature_calendar_day_of_week" not in modeling_df.columns


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
