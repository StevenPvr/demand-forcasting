from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.select_features_sarimax import run_feature_selection


def test_run_feature_selection_writes_sarimax_artifacts(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    train_out = tmp_path / "train_out.csv"
    val_out = tmp_path / "val_out.csv"
    test_out = tmp_path / "test_out.csv"
    summary_json = tmp_path / "summary.json"
    importances_csv = tmp_path / "importances.csv"
    tuning_trials_csv = tmp_path / "trials.csv"
    train_df = pd.DataFrame(
        {
            "origin_date": pd.date_range("2021-01-01", periods=16, freq="D").strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2021-01-02", periods=16, freq="D").strftime("%Y-%m-%d"),
            "exog_sales_1_lag1": [float(index + 1) for index in range(16)],
            "exog_sales_2_lag1": [float(index + 1) for index in range(16)],
            "exog_sales_3_lag1": [10.0] * 16,
            "exog_calendar_target_day_of_week": [index % 7 for index in range(16)],
            "exog_flag_target_day_missing": [0] * 16,
            "target_baguette_t_plus_1": [float(index + 1) for index in range(16)],
        }
    )
    val_df = pd.DataFrame(
        {
            "origin_date": pd.date_range("2021-01-17", periods=6, freq="D").strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2021-01-18", periods=6, freq="D").strftime("%Y-%m-%d"),
            "exog_sales_1_lag1": [17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            "exog_sales_2_lag1": [17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            "exog_sales_3_lag1": [10.0] * 6,
            "exog_calendar_target_day_of_week": [index % 7 for index in range(6)],
            "exog_flag_target_day_missing": [0] * 6,
            "target_baguette_t_plus_1": [17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "origin_date": pd.date_range("2021-01-23", periods=3, freq="D").strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2021-01-24", periods=3, freq="D").strftime("%Y-%m-%d"),
            "exog_sales_1_lag1": [23.0, 24.0, 25.0],
            "exog_sales_2_lag1": [23.0, 24.0, 25.0],
            "exog_sales_3_lag1": [10.0] * 3,
            "exog_calendar_target_day_of_week": [index % 7 for index in range(3)],
            "exog_flag_target_day_missing": [0] * 3,
            "target_baguette_t_plus_1": [23.0, 24.0, 25.0],
        }
    )
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    summary = run_feature_selection(
        train_input_csv=train_csv,
        val_input_csv=val_csv,
        test_input_csv=test_csv,
        train_output_csv=train_out,
        val_output_csv=val_out,
        test_output_csv=test_out,
        summary_json=summary_json,
        importances_csv=importances_csv,
        tuning_trials_csv=tuning_trials_csv,
        n_trials=2,
        n_folds=2,
        n_jobs=1,
    )
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    selected_train_df = pd.read_csv(train_out)

    assert summary["selector_trial_count"] == 2
    assert payload["target_model_family"] == "SARIMAX"
    assert payload["selector_model_family"] == "SARIMAX"
    assert payload["selected_lag_feature_columns"] == []
    assert payload["max_selected_exogenous_features"] == 5
    assert payload["selected_feature_count"] <= 5
    assert "exog_sales_2_lag1" not in selected_train_df.columns
    assert "exog_sales_3_lag1" not in selected_train_df.columns
    assert train_out.exists()
    assert val_out.exists()
    assert test_out.exists()
    assert importances_csv.exists()
    assert tuning_trials_csv.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
