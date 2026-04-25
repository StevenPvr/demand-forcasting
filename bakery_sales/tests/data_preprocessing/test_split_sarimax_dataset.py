from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.split_baguette_dataset import (
    split_baguette_dataframe,
    split_baguette_dataset,
)
import src.data_preprocessing.temporal_features as temporal_features


def test_split_baguette_dataframe_keeps_temporal_order() -> None:
    dataset_df = pd.DataFrame(
        {
            "origin_date": [f"2021-01-{index + 1:02d}" for index in range(10)],
            "target_date": [f"2021-01-{index + 2:02d}" for index in range(10)],
            "exog_sales_1_lag1": [float(index) for index in range(10)],
            "target_baguette_t_plus_1": [float(index + 1) for index in range(10)],
        }
    )

    train_df, val_df, test_df = split_baguette_dataframe(dataset_df)

    assert len(train_df) == 7
    assert len(val_df) == 1
    assert len(test_df) == 2
    assert list(train_df["origin_date"]) == [f"2021-01-{index + 1:02d}" for index in range(7)]
    assert list(val_df["origin_date"]) == ["2021-01-08"]
    assert list(test_df["origin_date"]) == ["2021-01-09", "2021-01-10"]


def test_split_baguette_dataset_writes_three_csv_files_bundle_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "daily.csv"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    stationarity_report_json = tmp_path / "stationarity_report.json"
    preprocessing_bundle_path = tmp_path / "bundle.joblib"
    pd.DataFrame(
        {
            "origin_date": pd.date_range("2021-01-01", periods=80, freq="D").strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2021-01-02", periods=80, freq="D").strftime("%Y-%m-%d"),
            "exog_sales_1_lag1": [float(index) for index in range(80)],
            "exog_flag_origin_day_missing": [0 for _ in range(80)],
            "exog_flag_target_day_missing": [0 for _ in range(80)],
            "target_baguette_t_plus_1": [float(index + 1) for index in range(80)],
        }
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr(
        temporal_features,
        "build_open_data_features",
        lambda dataset_df: pd.DataFrame(
            {
                "exog_holiday_target_is_school_holiday": [0] * len(dataset_df),
                "exog_weather_origin_temperature_2m_mean": [10.0] * len(dataset_df),
            },
            index=dataset_df.index,
        ),
    )

    summary = split_baguette_dataset(
        input_csv=input_csv,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        stationarity_report_json=stationarity_report_json,
        preprocessing_bundle_path=preprocessing_bundle_path,
    )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    bundle = joblib.load(preprocessing_bundle_path)

    assert summary == {
        "raw_row_count": 80,
        "row_count": 50,
        "train_rows": 35,
        "val_rows": 7,
        "test_rows": 8,
    }
    assert len(train_df) == 35
    assert len(val_df) == 7
    assert len(test_df) == 8
    assert train_df.iloc[0]["origin_date"] == "2021-01-31"
    assert stationarity_report_json.exists()
    assert preprocessing_bundle_path.exists()
    assert "exog_calendar_target_is_weekend" in train_df.columns
    assert "exog_autoreg_target_lag_30" in train_df.columns
    assert bundle["target_column"] == "target_baguette_t_plus_1"
    assert test_df.iloc[-1]["origin_date"] == "2021-03-21"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
