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

from src.optimisation_model.optimize_sarimax_model import (
    _evaluate_folds_in_parallel,
    _objective,
    _resolved_fold_jobs,
    _NonConvergedFoldError,
    run_optimization,
    suggest_optimization_params,
)


def test_suggest_optimization_params_exposes_expected_sarimax_space() -> None:
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


def test_run_optimization_writes_sarimax_artifacts(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    best_params_json = tmp_path / "best_params.json"
    trials_csv = tmp_path / "trials.csv"
    best_model_pkl = tmp_path / "best_model.pkl"
    best_predictions_csv = tmp_path / "best_predictions.csv"

    train_df = pd.DataFrame(
        {
            "origin_date": [f"2021-01-{index + 1:02d}" for index in range(28)],
            "target_date": [f"2021-01-{index + 2:02d}" for index in range(28)],
            "exog_sales_1_lag1": [20.0 + float(index % 7) for index in range(28)],
            "exog_calendar_target_day_of_week": [float(index % 7) for index in range(28)],
            "target_baguette_t_plus_1": [20.0 + float(index % 7) for index in range(28)],
        }
    )
    val_df = pd.DataFrame(
        {
            "origin_date": [f"2021-02-{index + 1:02d}" for index in range(6)],
            "target_date": [f"2021-02-{index + 2:02d}" for index in range(6)],
            "exog_sales_1_lag1": [22.0 + float(index % 7) for index in range(6)],
            "exog_calendar_target_day_of_week": [float(index % 7) for index in range(6)],
            "target_baguette_t_plus_1": [22.0 + float(index % 7) for index in range(6)],
        }
    )
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    class DummyResult:
        aic = 1.0
        mle_retvals = {"converged": True}

    def fake_fit_sarimax_model(
        history_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        params: dict[str, object],
    ) -> DummyResult:
        del history_df, feature_columns, target_column, params
        return DummyResult()

    def fake_batch_forecast(
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        params: dict[str, object],
    ) -> pd.DataFrame:
        train_rows_used = len(history_df)
        del feature_columns, target_column, params
        records = future_df.to_dict(orient="records")
        return pd.DataFrame(
            [
                {
                    "origin_date": row["origin_date"],
                    "target_date": row["target_date"],
                    "actual": row["target_baguette_t_plus_1"],
                    "prediction_raw": row["target_baguette_t_plus_1"],
                    "prediction_rounded": row["target_baguette_t_plus_1"],
                    "lower_80": row["target_baguette_t_plus_1"] - 1.0,
                    "upper_80": row["target_baguette_t_plus_1"] + 1.0,
                    "lower_95": row["target_baguette_t_plus_1"] - 2.0,
                    "upper_95": row["target_baguette_t_plus_1"] + 2.0,
                    "train_rows_used": train_rows_used,
                    "used_order": "(0, 0, 0)",
                    "used_seasonal_order": "(0, 0, 0, 0)",
                    "used_trend": "c",
                }
                for row in records
            ]
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("src.optimisation_model.optimize_sarimax_model.fit_sarimax_model", fake_fit_sarimax_model)
    monkeypatch.setattr("src.optimisation_model.optimize_sarimax_model.batch_forecast", fake_batch_forecast)
    monkeypatch.setattr(
        "src.optimisation_model.optimize_sarimax_model.save_sarimax_model",
        lambda fitted_model, output_path: output_path.write_text("dummy", encoding="utf-8"),
    )

    summary = run_optimization(
        train_csv=train_csv,
        val_csv=val_csv,
        best_params_json=best_params_json,
        trials_csv=trials_csv,
        best_model_pkl=best_model_pkl,
        best_predictions_csv=best_predictions_csv,
        n_trials=1,
        n_folds=2,
        n_jobs_folds=1,
    )
    monkeypatch.undo()

    best_payload = json.loads(best_params_json.read_text(encoding="utf-8"))
    predictions_df = pd.read_csv(best_predictions_csv)

    assert summary["model_family"] == "SARIMAX"
    assert summary["feature_count"] == 2
    assert best_payload["uses_exogenous_features"] is True
    assert best_payload["feature_count"] == 2
    assert len(predictions_df) == len(val_df)
    assert best_model_pkl.exists()


def test_resolved_fold_jobs_caps_to_fold_count() -> None:
    assert _resolved_fold_jobs(n_jobs_folds=-1, fold_count=5, cpu_count=12) == 5
    assert _resolved_fold_jobs(n_jobs_folds=3, fold_count=5, cpu_count=12) == 3


def test_evaluate_folds_in_parallel_prefers_processes(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("src.optimisation_model.optimize_sarimax_model.Parallel", FakeParallel)
    monkeypatch.setattr("src.optimisation_model.optimize_sarimax_model.delayed", fake_delayed)

    results = _evaluate_folds_in_parallel(
        params={"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 7], "trend": "c"},
        folds=[
            (
                pd.DataFrame({"origin_date": ["2021-01-01"], "target_baguette_t_plus_1": [1.0], "exog_sales_1_lag1": [1.0]}),
                pd.DataFrame({"origin_date": ["2021-01-02"], "target_baguette_t_plus_1": [2.0], "exog_sales_1_lag1": [2.0]}),
            )
        ] * 2,
        feature_columns=["exog_sales_1_lag1"],
        n_jobs_folds=2,
    )

    assert results == [(1.0, 2.0, 3.0, 4.0)]
    assert captured["prefer"] == "processes"
    assert len(delayed_calls) == 2


def test_objective_prunes_trial_when_a_fold_does_not_converge(monkeypatch: pytest.MonkeyPatch) -> None:
    train_df = pd.DataFrame({"origin_date": ["2021-01-01"], "exog_sales_1_lag1": [1.0], "target_baguette_t_plus_1": [1.0]})
    val_df = pd.DataFrame({"origin_date": ["2021-01-02"], "exog_sales_1_lag1": [2.0], "target_baguette_t_plus_1": [2.0]})
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    monkeypatch.setattr(
        "src.optimisation_model.optimize_sarimax_model._evaluate_folds_in_parallel",
        lambda params, folds, feature_columns, n_jobs_folds: (_ for _ in ()).throw(
            _NonConvergedFoldError("did not converge")
        ),
    )

    with pytest.raises(optuna.TrialPruned):
        _objective(trial=trial, train_df=train_df, val_df=val_df, n_folds=1, n_jobs_folds=1)
        


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
