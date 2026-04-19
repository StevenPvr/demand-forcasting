from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.standardization import standardize_splits


def test_standardize_splits_uses_train_statistics_only_and_returns_bundle() -> None:
    train_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "target_date": ["2021-01-02", "2021-01-03", "2021-01-04"],
            "exog_sales_1_lag1": [1.0, 2.0, 3.0],
            "exog_sales_2_lag1": [10.0, 10.0, 10.0],
            "exog_autoreg_target_lag_1": [4.0, 5.0, 6.0],
            "exog_flag_origin_day_missing": [0, 1, 0],
            "exog_calendar_target_day_of_week": [5, 6, 0],
            "target_baguette_t_plus_1": [4.0, 5.0, 6.0],
        }
    )
    val_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-04"],
            "target_date": ["2021-01-05"],
            "exog_sales_1_lag1": [4.0],
            "exog_sales_2_lag1": [10.0],
            "exog_autoreg_target_lag_1": [7.0],
            "exog_flag_origin_day_missing": [0],
            "exog_calendar_target_day_of_week": [1],
            "target_baguette_t_plus_1": [7.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-05"],
            "target_date": ["2021-01-06"],
            "exog_sales_1_lag1": [5.0],
            "exog_sales_2_lag1": [10.0],
            "exog_autoreg_target_lag_1": [8.0],
            "exog_flag_origin_day_missing": [1],
            "exog_calendar_target_day_of_week": [2],
            "target_baguette_t_plus_1": [8.0],
        }
    )

    scaled_train, scaled_val, scaled_test, bundle = standardize_splits(train_df, val_df, test_df)

    assert scaled_train["exog_sales_1_lag1"].mean() == pytest.approx(0.0)
    assert scaled_train["exog_sales_1_lag1"].std(ddof=0) == pytest.approx(1.0)
    assert scaled_val.loc[0, "exog_sales_1_lag1"] == pytest.approx(2.449489742783178)
    assert scaled_test.loc[0, "exog_sales_1_lag1"] == pytest.approx(3.6742346141747673)
    assert scaled_train["exog_sales_2_lag1"].tolist() == [0.0, 0.0, 0.0]
    assert scaled_train["exog_autoreg_target_lag_1"].mean() == pytest.approx(0.0)
    assert scaled_val.loc[0, "exog_autoreg_target_lag_1"] == pytest.approx(2.449489742783178)
    assert scaled_train["exog_flag_origin_day_missing"].tolist() == [0, 1, 0]
    assert scaled_train["exog_calendar_target_day_of_week"].mean() == pytest.approx(0.0)
    assert scaled_train["exog_calendar_target_day_of_week"].std(ddof=0) == pytest.approx(1.0)
    assert scaled_val.loc[0, "exog_calendar_target_day_of_week"] == pytest.approx(-1.016001016001524)
    assert scaled_test.loc[0, "exog_calendar_target_day_of_week"] == pytest.approx(-0.6350006350009525)
    assert bundle["scaled_sales_columns"] == [
        "exog_sales_1_lag1",
        "exog_sales_2_lag1",
        "exog_autoreg_target_lag_1",
        "exog_calendar_target_day_of_week",
    ]
    assert bundle["passthrough_columns"] == [
        "origin_date",
        "target_date",
        "exog_flag_origin_day_missing",
        "target_baguette_t_plus_1",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
