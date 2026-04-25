from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.feature_selection.select_features_linear as select_features_linear
from src.feature_selection.select_features_linear import (
    _drop_correlated_candidate_features,
    _evaluate_folds_in_parallel,
    _log_importance_progress,
    _mandatory_feature_columns,
    _should_log_importance_progress,
    run_feature_selection,
)


def test_mandatory_feature_columns_include_calendar_flags_holidays_and_weather() -> None:
    dataset_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-01"],
            "exog_calendar_target_day_of_week": [4],
            "exog_flag_target_day_missing": [0],
            "exog_holiday_target_is_school_holiday": [1],
            "exog_weather_origin_temperature_2m_mean": [7.5],
            "exog_autoreg_target_lag_1": [12.0],
            "exog_sales_1_lag1": [12.0],
        }
    )

    assert _mandatory_feature_columns(dataset_df) == [
        "exog_calendar_target_day_of_week",
        "exog_flag_target_day_missing",
        "exog_holiday_target_is_school_holiday",
        "exog_weather_origin_temperature_2m_mean",
    ]


def test_evaluate_folds_in_parallel_prefers_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeParallel:
        def __init__(self, *, n_jobs: int, prefer: str) -> None:
            captured["n_jobs"] = n_jobs
            captured["prefer"] = prefer

        def __call__(self, tasks: Any) -> list[tuple[float, float]]:
            _ = tasks
            return [(1.0, 0.5), (2.0, 0.25)]

    def fake_delayed(func: Any) -> Any:
        return lambda *args, **kwargs: (func, args, kwargs)

    monkeypatch.setattr(select_features_linear, "Parallel", FakeParallel)
    monkeypatch.setattr(select_features_linear, "delayed", fake_delayed)

    scores = _evaluate_folds_in_parallel(
        params={"alpha": 0.1, "l1_ratio": 0.8, "fit_intercept": True, "max_iter": 5000},
        folds=[
            (
                pd.DataFrame({"origin_date": ["2021-01-01"]}),
                pd.DataFrame({"origin_date": ["2021-01-02"]}),
            ),
            (
                pd.DataFrame({"origin_date": ["2021-01-03"]}),
                pd.DataFrame({"origin_date": ["2021-01-04"]}),
            ),
        ],
        feature_columns=["exog_sales_1_lag1"],
        n_jobs=-1,
    )

    assert scores == [(1.0, 0.5), (2.0, 0.25)]
    assert captured["n_jobs"] == -1
    assert captured["prefer"] == "processes"


def test_importance_progress_logs_at_start_interval_and_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        select_features_linear.LOGGER,
        "info",
        lambda message, *args: calls.append(message % args),
    )

    assert _should_log_importance_progress(index=0, total_features=10, every_n=5) is True
    assert _should_log_importance_progress(index=3, total_features=10, every_n=5) is False
    assert _should_log_importance_progress(index=4, total_features=10, every_n=5) is True
    assert _should_log_importance_progress(index=9, total_features=10, every_n=5) is True

    _log_importance_progress(index=0, total_features=10, feature_name="exog_sales_1_lag1", every_n=5)
    _log_importance_progress(index=4, total_features=10, feature_name="exog_sales_2_lag1", every_n=5)
    _log_importance_progress(index=9, total_features=10, feature_name="exog_sales_3_lag1", every_n=5)

    assert len(calls) == 3
    assert "1/10" in calls[0]
    assert "5/10" in calls[1]
    assert "10/10" in calls[2]


def test_drop_correlated_candidate_features_keeps_most_target_correlated_on_train_only() -> None:
    train_df = pd.DataFrame(
        {
            "exog_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "exog_b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "exog_c": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target_baguette_t_plus_1": [1.0, 2.0, 3.0, 4.0, 10.0],
        }
    )

    kept_columns, dropped_columns = _drop_correlated_candidate_features(
        train_df=train_df,
        candidate_columns=["exog_a", "exog_b", "exog_c"],
        target_column="target_baguette_t_plus_1",
        correlation_threshold=0.95,
    )

    assert kept_columns == ["exog_a"]
    assert dropped_columns == ["exog_b", "exog_c"]


def test_run_feature_selection_keeps_mandatory_core_and_selected_sales_features(
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    train_out = tmp_path / "train_selected.csv"
    val_out = tmp_path / "val_selected.csv"
    test_out = tmp_path / "test_selected.csv"
    summary_json = tmp_path / "summary.json"
    importances_csv = tmp_path / "importances.csv"
    tuning_trials_csv = tmp_path / "selector_trials.csv"
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
        n_trials=3,
        n_folds=3,
        n_jobs=1,
    )

    selected_train_df = pd.read_csv(train_out)
    selected_val_df = pd.read_csv(val_out)
    selected_test_df = pd.read_csv(test_out)
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    importance_df = pd.read_csv(importances_csv)
    selector_trials_df = pd.read_csv(tuning_trials_csv)

    assert summary["selector_trial_count"] == 3
    assert summary["selected_feature_count"] >= 3
    assert list(selected_train_df.columns[:4]) == [
        "origin_date",
        "target_date",
        "exog_calendar_target_day_of_week",
        "exog_flag_target_day_missing",
    ]
    assert "exog_sales_1_lag1" in selected_train_df.columns
    assert "exog_sales_2_lag1" not in selected_train_df.columns
    assert "exog_sales_3_lag1" not in selected_train_df.columns
    assert list(selected_val_df.columns) == list(selected_train_df.columns)
    assert list(selected_test_df.columns) == list(selected_train_df.columns)
    assert payload["mandatory_feature_columns"] == [
        "exog_calendar_target_day_of_week",
        "exog_flag_target_day_missing",
    ]
    assert payload["dropped_constant_features"] == ["exog_sales_3_lag1"]
    assert payload["dropped_correlated_features"] == ["exog_sales_2_lag1"]
    assert payload["stage_one_surviving_sales_feature_columns"] == ["exog_sales_1_lag1"]
    assert payload["stage_two_candidate_feature_columns"] == ["exog_sales_1_lag1"]
    assert payload["selected_sales_feature_columns"] == ["exog_sales_1_lag1"]
    assert payload["selected_lag_feature_columns"] == []
    assert payload["selector_model_family"] == "ElasticNet"
    assert "alpha" in payload["selector_best_params"]
    assert "l1_ratio" in payload["selector_best_params"]
    assert "fit_intercept" in payload["selector_best_params"]
    assert "max_iter" in payload["selector_best_params"]
    assert "selector_train_mae_from_tuning" in payload
    assert "selector_overfit_gap_penalty" in payload
    assert "selector_variance_gap_penalty" in payload
    assert "selector_train_mae" in payload
    assert "selector_val_mae" in payload
    assert len(selector_trials_df) == 3
    assert list(importance_df.columns) == [
        "feature",
        "importance",
        "normalized_importance",
        "selected",
        "selection_stage",
    ]
    assert set(importance_df["feature"]) == {"exog_sales_1_lag1"}
    assert set(importance_df["selection_stage"]) == {"non_lag", "with_lags"}
    assert importance_df["selected"].tolist() == [True, True]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
