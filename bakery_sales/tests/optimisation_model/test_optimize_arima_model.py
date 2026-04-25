from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import optuna
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimisation_model.optimize_arima_model import (
    _NonConvergedFoldError,
    _build_walk_forward_folds,
    _evaluate_folds_in_parallel,
    _objective,
    _resolved_fold_jobs,
    run_optimization,
    suggest_optimization_params,
)


def _product_frame(product_name: str, start_date: str, periods: int) -> pd.DataFrame:
    date_index = pd.date_range(start_date, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": date_index.strftime("%Y-%m-%d"),
            "product": [product_name] * periods,
            "quantity": [20.0 + float(index % 7) for index in range(periods)],
            "is_missing_day": [0] * periods,
        }
    )


def test_suggest_optimization_params_exposes_expected_sarima_space() -> None:
    calls: list[tuple[str, str, object, object]] = []

    class FakeTrial:
        def suggest_categorical(self, name: str, choices: list[object]) -> object:
            calls.append(("categorical", name, choices[0], choices[-1]))
            return choices[0]

        def suggest_int(self, name: str, low: int, high: int) -> int:
            calls.append(("int", name, low, high))
            return low

    params = suggest_optimization_params(cast(optuna.Trial, FakeTrial()))

    assert params["order"] == [0, 0, 0]
    assert params["seasonal_order"] == [0, 0, 0, 0]
    assert params["trend"] == "n"
    assert ("categorical", "seasonal_period", 0, 7) in calls
    assert ("int", "p", 0, 2) in calls
    assert ("int", "q", 0, 2) in calls


def test_run_optimization_writes_per_product_arima_artifacts(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    best_params_json = tmp_path / "best_params.json"
    trials_csv = tmp_path / "trials.csv"
    best_model_pkl = tmp_path / "best_model.pkl"
    best_predictions_csv = tmp_path / "best_predictions.csv"

    train_df = pd.concat(
        [
            _product_frame("BAGUETTE", "2021-01-01", 28),
            _product_frame("CROISSANT", "2021-01-01", 28),
        ],
        ignore_index=True,
    )
    val_df = pd.concat(
        [
            _product_frame("BAGUETTE", "2021-01-29", 6),
            _product_frame("CROISSANT", "2021-01-29", 6),
        ],
        ignore_index=True,
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
        n_trials=1,
        n_folds=1,
        n_jobs_folds=1,
    )

    best_payload = json.loads(best_params_json.read_text(encoding="utf-8"))
    predictions_df = pd.read_csv(best_predictions_csv)

    assert summary["model_family"] == "ARIMA_BY_PRODUCT"
    assert summary["feature_count"] == 0
    assert summary["product_count"] == 2
    assert best_payload["model_family"] == "ARIMA_BY_PRODUCT"
    assert best_payload["uses_exogenous_features"] is False
    assert best_payload["feature_columns"] == []
    assert best_payload["feature_count"] == 0
    assert set(best_payload["product_models"]) == {"BAGUETTE", "CROISSANT"}
    assert len(predictions_df) == len(val_df)
    assert set(predictions_df["product"]) == {"BAGUETTE", "CROISSANT"}
    assert best_model_pkl.exists()
    assert trials_csv.exists()


def test_resolved_fold_jobs_caps_to_fold_count() -> None:
    assert _resolved_fold_jobs(n_jobs_folds=-1, fold_count=5, cpu_count=12) == 5
    assert _resolved_fold_jobs(n_jobs_folds=3, fold_count=5, cpu_count=12) == 3
    assert _resolved_fold_jobs(n_jobs_folds=99, fold_count=4, cpu_count=12) == 4


def test_build_walk_forward_folds_uses_linewise_validation_even_if_requested_folds_is_one() -> None:
    train_df = _product_frame("BAGUETTE", "2021-01-01", 4)
    val_df = _product_frame("BAGUETTE", "2021-01-05", 3)

    folds = _build_walk_forward_folds(train_df=train_df, val_df=val_df, n_folds=1)

    assert len(folds) == 3
    assert [len(fold_val) for _, fold_val in folds] == [1, 1, 1]
    assert [len(fold_train) for fold_train, _ in folds] == [4, 5, 6]


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

    def fake_delayed(func: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> tuple[object, tuple[object, ...], dict[str, object]]:
            delayed_calls.append(args)
            return func, args, kwargs

        return wrapper

    monkeypatch.setattr("src.optimisation_model.optimize_arima_model.Parallel", FakeParallel)
    monkeypatch.setattr("src.optimisation_model.optimize_arima_model.delayed", fake_delayed)

    results = _evaluate_folds_in_parallel(
        params={"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "trend": "n"},
        folds=[
            (
                pd.DataFrame({"date": ["2021-01-01"], "product": ["BAGUETTE"], "quantity": [1.0]}),
                pd.DataFrame({"date": ["2021-01-02"], "product": ["BAGUETTE"], "quantity": [2.0]}),
            )
        ]
        * 2,
        n_jobs_folds=2,
    )

    assert results == [(1.0, 2.0, 3.0, 4.0)]
    assert captured["n_jobs"] == 2
    assert captured["prefer"] == "processes"
    assert len(delayed_calls) == 2


def test_objective_prunes_trial_when_a_fold_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_df = pd.DataFrame(
        {
            "date": ["2021-01-01"],
            "product": ["BAGUETTE"],
            "quantity": [1.0],
        }
    )
    val_df = pd.DataFrame(
        {
            "date": ["2021-01-02"],
            "product": ["BAGUETTE"],
            "quantity": [2.0],
        }
    )
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    monkeypatch.setattr(
        "src.optimisation_model.optimize_arima_model._evaluate_folds_in_parallel",
        lambda params, folds, n_jobs_folds: (_ for _ in ()).throw(
            _NonConvergedFoldError("did not converge")
        ),
    )

    with pytest.raises(optuna.TrialPruned):
        _objective(
            trial=trial,
            train_df=train_df,
            val_df=val_df,
            n_jobs_folds=1,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
