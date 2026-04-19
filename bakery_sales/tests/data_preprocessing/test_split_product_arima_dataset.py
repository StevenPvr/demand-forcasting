from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.split_product_arima_dataset import (
    split_product_arima_dataframe,
    split_product_arima_dataset,
)


def test_split_product_arima_dataframe_keeps_temporal_order_per_product() -> None:
    baguette_df = pd.DataFrame(
        {
            "date": [f"2021-01-{index + 1:02d}" for index in range(10)],
            "product": ["BAGUETTE"] * 10,
            "quantity": [float(index + 1) for index in range(10)],
            "is_missing_day": [0] * 10,
        }
    )
    croissant_df = pd.DataFrame(
        {
            "date": [f"2021-01-{index + 1:02d}" for index in range(10)],
            "product": ["CROISSANT"] * 10,
            "quantity": [float(index + 11) for index in range(10)],
            "is_missing_day": [0] * 10,
        }
    )
    dataset_df = pd.concat([baguette_df, croissant_df], ignore_index=True)

    train_df, val_df, test_df = split_product_arima_dataframe(dataset_df)

    assert len(train_df) == 14
    assert len(val_df) == 2
    assert len(test_df) == 4
    assert train_df.groupby("product").size().to_dict() == {"BAGUETTE": 7, "CROISSANT": 7}
    assert val_df.groupby("product").size().to_dict() == {"BAGUETTE": 1, "CROISSANT": 1}
    assert test_df.groupby("product").size().to_dict() == {"BAGUETTE": 2, "CROISSANT": 2}


def test_split_product_arima_dataset_writes_csv_files_bundle_and_report(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "daily.csv"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    stationarity_report_json = tmp_path / "stationarity_report.json"
    preprocessing_bundle_path = tmp_path / "bundle.joblib"

    baguette_df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=20, freq="D").strftime("%Y-%m-%d"),
            "product": ["BAGUETTE"] * 20,
            "quantity": [float(index + 1) for index in range(20)],
            "is_missing_day": [0] * 20,
        }
    )
    croissant_df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=20, freq="D").strftime("%Y-%m-%d"),
            "product": ["CROISSANT"] * 20,
            "quantity": [float(index + 21) for index in range(20)],
            "is_missing_day": [0] * 20,
        }
    )
    pd.concat([baguette_df, croissant_df], ignore_index=True).to_csv(input_csv, index=False)

    summary = split_product_arima_dataset(
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
    stationarity_payload = json.loads(stationarity_report_json.read_text(encoding="utf-8"))
    bundle = joblib.load(preprocessing_bundle_path)

    assert summary == {
        "raw_row_count": 40,
        "row_count": 40,
        "product_count": 2,
        "train_rows": 28,
        "val_rows": 6,
        "test_rows": 6,
    }
    assert train_df.groupby("product").size().to_dict() == {"BAGUETTE": 14, "CROISSANT": 14}
    assert val_df.groupby("product").size().to_dict() == {"BAGUETTE": 3, "CROISSANT": 3}
    assert test_df.groupby("product").size().to_dict() == {"BAGUETTE": 3, "CROISSANT": 3}
    assert bundle["target_column"] == "quantity"
    assert bundle["product_column"] == "product"
    assert stationarity_payload["product_count"] == 2
    assert set(stationarity_payload["product_reports"]) == {"BAGUETTE", "CROISSANT"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
