from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning.prepare_baguette_dataset import (
    _remove_negative_cancellations,
    build_daily_baguette_training_frame,
    prepare_daily_baguette_dataset,
)


def test_build_daily_baguette_training_frame_regularizes_daily_grid_and_aligns_j_plus_1() -> None:
    sales_df = pd.DataFrame(
        {
            "date": [
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-04",
                "2021-01-04",
                "2021-01-04",
            ],
            "time": ["08:01", "08:04", "08:13", "09:00", "09:05", "09:15"],
            "article": [
                "CROISSANT",
                "BAGUETTE",
                "CROISSANT",
                "CROISSANT",
                "BAGUETTE",
                "CROISSANT",
            ],
            "Quantity": [1.0, 3.0, 3.0, 2.0, 5.0, 1.0],
        }
    )

    training_df, mapping = build_daily_baguette_training_frame(sales_df)

    assert list(training_df["origin_date"]) == ["2021-01-02", "2021-01-03"]
    assert list(training_df["target_date"]) == ["2021-01-03", "2021-01-04"]
    assert list(training_df["exog_sales_1_lag1"]) == [4.0, 0.0]
    assert list(training_df["target_baguette_t_plus_1"]) == [0.0, 5.0]
    assert list(training_df["exog_flag_origin_day_missing"]) == [0, 1]
    assert list(training_df["exog_flag_target_day_missing"]) == [1, 0]
    assert mapping == {"exog_sales_1_lag1": "CROISSANT"}


def test_build_daily_baguette_training_frame_filters_placeholder_article() -> None:
    sales_df = pd.DataFrame(
        {
            "date": ["2021-01-02", "2021-01-02", "2021-01-03"],
            "time": ["08:01", "08:04", "08:08"],
            "ticket_number": ["1001", "1001", "1002"],
            "article": ["BAGUETTE", ".", "BAGUETTE"],
            "Quantity": [1.0, 7.0, 2.0],
            "unit_price": [0.9, 3.0, 0.9],
        }
    )

    training_df, mapping = build_daily_baguette_training_frame(sales_df)

    assert "." not in mapping.values()
    first_row = training_df.iloc[0]
    assert first_row["origin_date"] == "2021-01-02"
    assert first_row["target_date"] == "2021-01-03"
    assert first_row["exog_sales_total_quantity_lag1"] == pytest.approx(1.0)
    assert first_row["exog_sales_ticket_count_lag1"] == pytest.approx(1.0)
    assert first_row["exog_sales_total_revenue_lag1"] == pytest.approx(0.9)
    assert first_row["exog_flag_origin_day_missing"] == 0
    assert first_row["exog_flag_target_day_missing"] == 0
    assert first_row["target_baguette_t_plus_1"] == pytest.approx(2.0)
    assert mapping == {}


def test_remove_negative_cancellations_offsets_nearest_previous_positive_rows() -> None:
    sales_df = pd.DataFrame(
        {
            "date": [
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
            ],
            "time": ["08:00", "08:05", "08:10", "08:12", "08:15"],
            "article": [
                "CROISSANT",
                "CROISSANT",
                "CROISSANT",
                "BAGUETTE",
                "CROISSANT",
            ],
            "Quantity": [1.0, 2.0, -2.0, 3.0, -5.0],
        }
    )

    cleaned_df = _remove_negative_cancellations(sales_df)

    assert cleaned_df.to_dict("records") == [
        {
            "date": "2021-01-02",
            "time": "08:12",
            "article": "BAGUETTE",
            "Quantity": 3.0,
        }
    ]


def test_prepare_daily_baguette_dataset_writes_csv_and_json(tmp_path: Path) -> None:
    input_csv = tmp_path / "bakery.csv"
    output_csv = tmp_path / "daily.csv"
    output_json = tmp_path / "mapping.json"
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

    summary = prepare_daily_baguette_dataset(
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
    )

    training_df = pd.read_csv(output_csv)
    mapping_payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary["row_count"] == 2
    assert int(summary["feature_count"]) > 1
    first_row = training_df.iloc[0]
    assert first_row["origin_date"] == "2021-01-02"
    assert first_row["target_date"] == "2021-01-03"
    assert first_row["exog_sales_1_lag1"] == pytest.approx(1.0)
    assert first_row["exog_sales_total_quantity_lag1"] == pytest.approx(3.0)
    assert first_row["exog_sales_total_revenue_lag1"] == pytest.approx(2.9)
    assert first_row["exog_sales_ticket_count_lag1"] == pytest.approx(1.0)
    assert first_row["exog_flag_origin_day_missing"] == 0
    assert first_row["exog_flag_target_day_missing"] == 1
    assert first_row["target_baguette_t_plus_1"] == pytest.approx(0.0)
    assert mapping_payload["aggregation_level"] == "daily"
    assert mapping_payload["forecast_horizon_days"] == 1
    assert mapping_payload["target_article"] == "BAGUETTE"
    assert mapping_payload["target_column"] == "target_baguette_t_plus_1"
    assert mapping_payload["sales_exog_mapping"] == {"exog_sales_1_lag1": "CROISSANT"}


def test_build_daily_baguette_training_frame_creates_transaction_and_category_features() -> None:
    sales_df = pd.DataFrame(
        {
            "date": [
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-03",
            ],
            "time": ["08:00", "08:05", "08:10", "12:30", "12:32", "15:00", "09:00"],
            "ticket_number": ["1001", "1001", "1001", "1002", "1002", "1003", "1004"],
            "article": [
                "BAGUETTE",
                "CROISSANT",
                "CAFE OU EAU",
                "FORMULE SANDWICH",
                "BOISSON 33CL",
                "TARTELETTE",
                "BAGUETTE",
            ],
            "Quantity": [3.0, 2.0, 1.0, 1.0, 1.0, 2.0, 4.0],
            "unit_price": [0.9, 1.1, 1.5, 4.0, 1.2, 2.5, 0.9],
        }
    )

    training_df, _ = build_daily_baguette_training_frame(sales_df)

    first_row = training_df.iloc[0]
    assert first_row["exog_sales_total_quantity_lag1"] == pytest.approx(10.0)
    assert first_row["exog_sales_total_revenue_lag1"] == pytest.approx(16.6)
    assert first_row["exog_sales_ticket_count_lag1"] == pytest.approx(3.0)
    assert first_row["exog_sales_unique_article_count_lag1"] == pytest.approx(6.0)
    assert first_row["exog_sales_avg_items_per_ticket_lag1"] == pytest.approx(10.0 / 3.0)
    assert first_row["exog_sales_avg_revenue_per_ticket_lag1"] == pytest.approx(16.6 / 3.0)
    assert first_row["exog_sales_weighted_unit_price_lag1"] == pytest.approx(1.66)
    assert first_row["exog_sales_first_sale_minute_lag1"] == pytest.approx(480.0)
    assert first_row["exog_sales_last_sale_minute_lag1"] == pytest.approx(900.0)
    assert first_row["exog_sales_sales_span_minutes_lag1"] == pytest.approx(420.0)
    assert first_row["exog_sales_morning_quantity_share_lag1"] == pytest.approx(0.6)
    assert first_row["exog_sales_lunch_quantity_share_lag1"] == pytest.approx(0.2)
    assert first_row["exog_sales_afternoon_quantity_share_lag1"] == pytest.approx(0.2)
    assert first_row["exog_sales_evening_quantity_share_lag1"] == pytest.approx(0.0)
    assert first_row["exog_sales_category_bread_quantity_lag1"] == pytest.approx(3.0)
    assert first_row["exog_sales_category_viennoiserie_quantity_lag1"] == pytest.approx(2.0)
    assert first_row["exog_sales_category_pastry_quantity_lag1"] == pytest.approx(2.0)
    assert first_row["exog_sales_category_sandwich_quantity_lag1"] == pytest.approx(1.0)
    assert first_row["exog_sales_category_beverage_quantity_lag1"] == pytest.approx(2.0)
    assert first_row["exog_sales_category_bread_revenue_lag1"] == pytest.approx(2.7)
    assert first_row["exog_sales_category_beverage_revenue_lag1"] == pytest.approx(2.7)
    assert first_row["exog_sales_category_pastry_quantity_share_lag1"] == pytest.approx(0.2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
