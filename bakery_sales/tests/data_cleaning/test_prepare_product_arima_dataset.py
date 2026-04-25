from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning.prepare_product_arima_dataset import (
    build_daily_product_training_frame,
    prepare_daily_product_arima_dataset,
)


def test_build_daily_product_training_frame_regularizes_grid_for_each_product() -> None:
    sales_df = pd.DataFrame(
        {
            "date": [
                "2021-01-02",
                "2021-01-02",
                "2021-01-04",
                "2021-01-04",
            ],
            "time": ["08:01", "08:04", "09:00", "09:05"],
            "article": ["CROISSANT", "BAGUETTE", "CROISSANT", "BAGUETTE"],
            "Quantity": [4.0, 3.0, 2.0, 5.0],
        }
    )

    training_df, metadata = build_daily_product_training_frame(sales_df)

    assert len(training_df) == 6
    assert list(training_df.columns) == ["date", "product", "quantity", "is_missing_day"]
    assert training_df.loc[
        (training_df["date"] == "2021-01-03") & (training_df["product"] == "BAGUETTE"),
        "quantity",
    ].item() == pytest.approx(0.0)
    assert training_df.loc[
        (training_df["date"] == "2021-01-03") & (training_df["product"] == "CROISSANT"),
        "is_missing_day",
    ].item() == 1
    assert metadata["product_column"] == "product"
    assert metadata["target_column"] == "quantity"
    assert metadata["products"] == ["BAGUETTE", "CROISSANT"]


def test_prepare_daily_product_arima_dataset_writes_csv_and_json(tmp_path: Path) -> None:
    input_csv = tmp_path / "bakery.csv"
    output_csv = tmp_path / "daily.csv"
    output_json = tmp_path / "summary.json"
    input_csv.write_text(
        "\n".join(
            [
                "index,date,time,ticket_number,article,Quantity,unit_price",
                '0,2021-01-02,08:01,150040.0,CROISSANT,1.0,"1,10 €"',
                '1,2021-01-02,08:04,150040.0,BAGUETTE,2.0,"0,90 €"',
                '2,2021-01-04,08:13,150041.0,CROISSANT,3.0,"1,10 €"',
                '3,2021-01-04,08:16,150041.0,BAGUETTE,4.0,"0,90 €"',
            ]
        ),
        encoding="utf-8",
    )

    summary = prepare_daily_product_arima_dataset(
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
    )

    training_df = pd.read_csv(output_csv)
    metadata = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary["row_count"] == 6
    assert summary["product_count"] == 2
    assert set(training_df["product"]) == {"BAGUETTE", "CROISSANT"}
    assert metadata["aggregation_level"] == "daily"
    assert metadata["target_column"] == "quantity"
    assert metadata["missing_flag_column"] == "is_missing_day"


def test_build_daily_product_training_frame_excludes_products_with_more_than_one_third_zero_days() -> None:
    sales_df = pd.DataFrame(
        {
            "date": [
                "2021-01-01",
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
            ],
            "time": ["08:00", "08:05", "08:10", "08:15"],
            "article": ["BAGUETTE", "CROISSANT", "BAGUETTE", "BAGUETTE"],
            "Quantity": [3.0, 1.0, 4.0, 5.0],
        }
    )

    training_df, metadata = build_daily_product_training_frame(sales_df)

    assert set(training_df["product"]) == {"BAGUETTE"}
    assert metadata["products"] == ["BAGUETTE"]
    assert metadata["product_count"] == 1
    assert metadata["excluded_products_due_to_zero_ratio"] == ["CROISSANT"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
