from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.stationarity import build_stationarity_report


def test_build_stationarity_report_describes_target_without_transforming_dataset() -> None:
    row_count = 160
    random_walk = np.cumsum(np.random.default_rng(11).normal(size=row_count))
    train_df = pd.DataFrame(
        {
            "origin_date": pd.date_range("2021-01-01", periods=row_count, freq="D").strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2021-01-02", periods=row_count, freq="D").strftime("%Y-%m-%d"),
            "exog_sales_1_lag1": np.random.default_rng(7).normal(size=row_count),
            "target_baguette_t_plus_1": random_walk + 10.0,
        }
    )

    report = build_stationarity_report(train_df, target_column="target_baguette_t_plus_1")

    assert report["target_column"] == "target_baguette_t_plus_1"
    assert report["row_count"] == row_count
    assert report["target"]["level"]["stationary"] is False
    assert report["target"]["difference_1"]["stationary"] is True
    assert report["candidate_d_values"] == [0, 1]


def test_build_stationarity_report_handles_constant_target() -> None:
    row_count = 30
    train_df = pd.DataFrame(
        {
            "origin_date": pd.date_range("2021-01-01", periods=row_count, freq="D").strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2021-01-02", periods=row_count, freq="D").strftime("%Y-%m-%d"),
            "target_baguette_t_plus_1": np.ones(row_count, dtype=float),
        }
    )

    report = build_stationarity_report(train_df, target_column="target_baguette_t_plus_1")

    assert report["target"]["level"]["status"] == "constant"
    assert report["candidate_d_values"] == [0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
