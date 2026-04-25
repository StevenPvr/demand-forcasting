from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import optuna
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimisation_model.optimize_elasticnet_model import (
    _evaluate_folds_in_parallel,
    _resolved_fold_jobs,
    build_walk_forward_folds,
    mae_score,
    suggest_optimization_params,
    run_optimization,
)


def test_build_walk_forward_folds_expands_training_window() -> None:
    train_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "target_baguette_t_plus_1": [1, 2, 3, 4],
        }
    )
    val_df = pd.DataFrame(
        {
            "origin_date": [
                "2021-01-05",
                "2021-01-06",
                "2021-01-07",
                "2021-01-08",
                "2021-01-09",
                "2021-01-10",
            ],
            "target_baguette_t_plus_1": [5, 6, 7, 8, 9, 10],
        }
    )

    folds = build_walk_forward_folds(train_df=train_df, val_df=val_df, n_folds=3)

    assert [(len(train_fold), len(val_fold)) for train_fold, val_fold in folds] == [
        (4, 2),
        (6, 2),
        (8, 2),
    ]
    assert folds[1][0]["origin_date"].tolist() == [
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        "2021-01-05",
        "2021-01-06",
    ]


def test_mae_score_uses_original_scale_predictions() -> None:
    y_true = pd.Series([0.0, 2.0, 3.0, 5.0])
    y_pred = pd.Series([0.5, 1.5, 3.5, 4.5])

    assert mae_score(y_true=y_true, y_pred=y_pred) == pytest.approx(0.5)


def test_suggest_optimization_params_uses_expected_search_ranges() -> None:
    calls: list[tuple[str, str, object, object, object | None]] = []

    class FakeTrial:
        def suggest_categorical(self, name: str, choices: list[object]) -> object:
            calls.append(("categorical", name, choices[0], choices[-1], None))
            return choices[-1]

        def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
            calls.append(("float", name, low, high, log))
            return low

    params = suggest_optimization_params(cast(optuna.Trial, FakeTrial()))

    assert params["alpha"] > 0.0
    assert 0.0 <= params["l1_ratio"] <= 1.0
    assert params["fit_intercept"] is False
    assert params["max_iter"] == 5000
    assert ("float", "alpha", 0.0001, 10.0, True) in calls
    assert ("float", "l1_ratio", 0.05, 1.0, False) in calls
    assert ("categorical", "fit_intercept", True, False, None) in calls


def test_resolved_fold_jobs_caps_to_fold_count() -> None:
    assert _resolved_fold_jobs(n_jobs_folds=-1, fold_count=5, cpu_count=12) == 5
    assert _resolved_fold_jobs(n_jobs_folds=3, fold_count=5, cpu_count=12) == 3
    assert _resolved_fold_jobs(n_jobs_folds=99, fold_count=4, cpu_count=12) == 4


def test_evaluate_folds_in_parallel_prefers_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    delayed_calls: list[tuple[object, ...]] = []

    class FakeParallel:
        def __init__(self, *, n_jobs: int, prefer: str) -> None:
            captured["n_jobs"] = n_jobs
            captured["prefer"] = prefer

        def __call__(self, tasks: object) -> list[tuple[float, float, float, float]]:
            list(cast(list[object], tasks))
            return [(1.0, 2.0, 3.0, 4.0)]

    def fake_delayed(func: Any) -> Any:
        def wrapper(*args: object, **kwargs: object) -> tuple[object, tuple[object, ...], dict[str, object]]:
            delayed_calls.append(args)
            return func, args, kwargs

        return wrapper

    monkeypatch.setattr("src.optimisation_model.optimize_elasticnet_model.Parallel", FakeParallel)
    monkeypatch.setattr("src.optimisation_model.optimize_elasticnet_model.delayed", fake_delayed)

    results = _evaluate_folds_in_parallel(
        params={"alpha": 1.0, "l1_ratio": 0.5, "fit_intercept": True, "max_iter": 5000},
        folds=[
            (
                pd.DataFrame({"origin_date": ["2021-01-01"]}),
                pd.DataFrame({"origin_date": ["2021-01-02"]}),
            )
        ]
        * 2,
        feature_columns=["exog_sales_1_lag1"],
        n_jobs_folds=2,
    )

    assert results == [(1.0, 2.0, 3.0, 4.0)]
    assert captured["n_jobs"] == 2
    assert captured["prefer"] == "processes"
    assert len(delayed_calls) == 2


def test_run_optimization_writes_best_params_trials_and_intervals(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    best_params_json = tmp_path / "best_params.json"
    trials_csv = tmp_path / "trials.csv"
    best_model_pkl = tmp_path / "best_model.pkl"
    best_predictions_csv = tmp_path / "best_predictions.csv"

    train_df = pd.DataFrame(
        {
            "origin_date": [f"2021-01-{index + 1:02d}" for index in range(20)],
            "target_date": [f"2021-01-{index + 2:02d}" for index in range(20)],
            "exog_sales_1_lag1": [float(index) for index in range(20)],
            "exog_calendar_target_day_of_week": [float(index % 7) for index in range(20)],
            "target_baguette_t_plus_1": [float(index // 2) for index in range(20)],
        }
    )
    val_df = pd.DataFrame(
        {
            "origin_date": [f"2021-02-{index + 1:02d}" for index in range(10)],
            "target_date": [f"2021-02-{index + 2:02d}" for index in range(10)],
            "exog_sales_1_lag1": [float(index + 20) for index in range(10)],
            "exog_calendar_target_day_of_week": [float((index + 1) % 7) for index in range(10)],
            "target_baguette_t_plus_1": [float((index + 20) // 2) for index in range(10)],
        }
    )
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    summary = run_optimization(
        train_csv=train_csv,
        val_csv=val_csv,
        best_params_json=best_params_json,
        trials_csv=trials_csv,
        best_model_pkl=best_model_pkl,
        best_predictions_csv=best_predictions_csv,
        n_trials=2,
        n_folds=2,
        n_jobs_folds=1,
    )

    best_payload = json.loads(best_params_json.read_text(encoding="utf-8"))
    trials_df = pd.read_csv(trials_csv)
    predictions_df = pd.read_csv(best_predictions_csv)

    assert summary["trial_count"] == 2
    assert summary["fold_count"] == len(val_df)
    assert summary["feature_count"] == 2
    assert summary["model_family"] == "ElasticNet"
    assert best_payload["trial_count"] == 2
    assert best_payload["fold_count"] == len(val_df)
    assert best_payload["model_family"] == "ElasticNet"
    assert best_payload["uses_exogenous_features"] is True
    assert best_payload["feature_columns"] == [
        "exog_sales_1_lag1",
        "exog_calendar_target_day_of_week",
    ]
    assert 0.0001 <= best_payload["best_params"]["alpha"] <= 10.0
    assert 0.05 <= best_payload["best_params"]["l1_ratio"] <= 0.95
    assert "best_score_mae" in best_payload
    assert "best_score_mase" in best_payload
    assert "complexity_penalty" in best_payload
    assert "variance_gap_penalty" in best_payload
    assert len(trials_df) == 2
    assert best_model_pkl.exists()
    assert len(predictions_df) == len(val_df)
    assert predictions_df["fold"].tolist() == list(range(1, len(val_df) + 1))
    assert predictions_df["train_rows_used"].tolist() == list(range(len(train_df), len(train_df) + len(val_df)))
    assert list(predictions_df.columns) == [
        "fold",
        "origin_date",
        "target_date",
        "actual",
        "prediction_raw",
        "prediction_rounded",
        "lower_80",
        "upper_80",
        "lower_95",
        "upper_95",
        "train_rows_used",
        "used_alpha",
        "used_l1_ratio",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
