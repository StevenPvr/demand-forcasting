from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.elasticnet_time_series import rolling_forecast
from src.evaluation.evaluate_elasticnet_model import (
    _should_log_progress,
    rolling_origin_predictions,
    smape,
    run_evaluation,
)


def test_smape_handles_zero_actuals_and_predictions() -> None:
    y_true = pd.Series([0.0, 2.0, 4.0])
    y_pred = pd.Series([0.0, 3.0, 2.0])

    assert smape(y_true=y_true, y_pred=y_pred) == pytest.approx(0.35555555555555557)


def test_should_log_progress_at_start_interval_and_end() -> None:
    assert _should_log_progress(index=0, total_rows=1000, every_n=250) is True
    assert _should_log_progress(index=248, total_rows=1000, every_n=250) is False
    assert _should_log_progress(index=249, total_rows=1000, every_n=250) is True
    assert _should_log_progress(index=999, total_rows=1000, every_n=250) is True


def test_rolling_origin_predictions_retrains_on_history() -> None:
    history_df = pd.DataFrame(
        {
            "origin_date": [f"2021-01-{index + 1:02d}" for index in range(6)],
            "target_date": [f"2021-01-{index + 2:02d}" for index in range(6)],
            "exog_sales_1_lag1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "target_baguette_t_plus_1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "origin_date": ["2021-02-01", "2021-02-02", "2021-02-03"],
            "target_date": ["2021-02-02", "2021-02-03", "2021-02-04"],
            "exog_sales_1_lag1": [6.0, 7.0, 8.0],
            "target_baguette_t_plus_1": [6.0, 7.0, 8.0],
        }
    )
    best_payload = {
        "best_params": {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "fit_intercept": True,
            "max_iter": 5000,
        },
        "model_family": "ElasticNet",
    }

    predictions_df = rolling_origin_predictions(
        history_df=history_df,
        test_df=test_df,
        best_payload=best_payload,
    )

    assert list(predictions_df["train_rows_used"]) == [6, 7, 8]
    assert list(predictions_df["target_date"]) == ["2021-02-02", "2021-02-03", "2021-02-04"]
    assert {"lower_80", "upper_80", "lower_95", "upper_95"} <= set(predictions_df.columns)


def test_rolling_forecast_refits_on_full_available_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "target_date": ["2021-01-02", "2021-01-03", "2021-01-04"],
            "exog_sales_1_lag1": [1.0, 2.0, 3.0],
            "target_baguette_t_plus_1": [1.0, 2.0, 3.0],
        }
    )
    future_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-04", "2021-01-05"],
            "target_date": ["2021-01-05", "2021-01-06"],
            "exog_sales_1_lag1": [4.0, 5.0],
            "target_baguette_t_plus_1": [4.0, 5.0],
        }
    )
    fitted_history_sizes: list[int] = []

    class DummyModel:
        pass

    def fake_fit_elasticnet_model(
        history_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        params: dict[str, object],
        random_seed: int,
    ) -> DummyModel:
        del feature_columns, target_column, params, random_seed
        fitted_history_sizes.append(len(history_df))
        return DummyModel()

    monkeypatch.setattr("src.elasticnet_time_series.fit_elasticnet_model", fake_fit_elasticnet_model)
    monkeypatch.setattr(
        "src.elasticnet_time_series.predict_frame",
        lambda fitted_model, dataset_df, feature_columns: pd.Series(
            [10.0] * len(dataset_df),
            index=dataset_df.index,
            dtype=float,
        ),
    )
    monkeypatch.setattr("src.elasticnet_time_series.residual_scale", lambda *args, **kwargs: 1.0)

    predictions_df = rolling_forecast(
        history_df=history_df,
        future_df=future_df,
        feature_columns=["exog_sales_1_lag1"],
        target_column="target_baguette_t_plus_1",
        params={
            "alpha": 0.1,
            "l1_ratio": 0.9,
            "fit_intercept": True,
            "max_iter": 5000,
        },
        random_seed=7,
    )

    assert fitted_history_sizes == [3, 4]
    assert predictions_df["train_rows_used"].tolist() == [3, 4]


def test_rolling_origin_predictions_uses_best_payload_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-01"],
            "target_date": ["2021-01-02"],
            "exog_sales_1_lag1": [1.0],
            "target_baguette_t_plus_1": [1.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-02"],
            "target_date": ["2021-01-03"],
            "exog_sales_1_lag1": [2.0],
            "target_baguette_t_plus_1": [2.0],
        }
    )
    captured: dict[str, object] = {}

    def fake_rolling_forecast(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "origin_date": "2021-01-02",
                    "target_date": "2021-01-03",
                    "actual": 2.0,
                    "prediction_raw": 2.0,
                    "prediction_rounded": 2.0,
                    "lower_80": 1.0,
                    "upper_80": 3.0,
                    "lower_95": 0.5,
                    "upper_95": 3.5,
                    "train_rows_used": 1,
                    "used_alpha": 0.1,
                    "used_l1_ratio": 0.9,
                }
            ]
        )

    monkeypatch.setattr(
        "src.evaluation.evaluate_elasticnet_model.rolling_forecast",
        fake_rolling_forecast,
    )

    predictions_df = rolling_origin_predictions(
        history_df=history_df,
        test_df=test_df,
        best_payload={
            "best_params": {
                "alpha": 0.1,
                "l1_ratio": 0.9,
                "fit_intercept": False,
                "max_iter": 5000,
            },
            "model_family": "ElasticNet",
        },
        n_jobs=3,
    )

    assert len(predictions_df) == 1
    assert captured["params"] == {
        "alpha": 0.1,
        "l1_ratio": 0.9,
        "fit_intercept": False,
        "max_iter": 5000,
    }


def test_run_evaluation_writes_metrics_predictions_model_card_and_diagnostics(
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    best_params_json = tmp_path / "best_params.json"
    metrics_json = tmp_path / "metrics.json"
    predictions_csv = tmp_path / "predictions.csv"
    predictions_plot_png = tmp_path / "predictions.png"
    diagnostics_json = tmp_path / "diagnostics.json"
    model_card_json = tmp_path / "model_card.json"
    statistical_baselines_json = tmp_path / "statistical_baselines.json"
    final_model_pkl = tmp_path / "final_model.pkl"
    residuals_plot_png = tmp_path / "residuals.png"
    residuals_qq_png = tmp_path / "residuals_qq.png"
    residuals_acf_pacf_png = tmp_path / "residuals_acf_pacf.png"

    train_df = pd.DataFrame(
        {
            "origin_date": [f"2021-01-{index + 1:02d}" for index in range(8)],
            "target_date": [f"2021-01-{index + 2:02d}" for index in range(8)],
            "exog_sales_1_lag1": [float(index) for index in range(8)],
            "exog_calendar_target_day_of_week": [float(index % 7) for index in range(8)],
            "target_baguette_t_plus_1": [float(index) for index in range(8)],
        }
    )
    val_df = pd.DataFrame(
        {
            "origin_date": [f"2021-02-{index + 1:02d}" for index in range(4)],
            "target_date": [f"2021-02-{index + 2:02d}" for index in range(4)],
            "exog_sales_1_lag1": [float(index + 8) for index in range(4)],
            "exog_calendar_target_day_of_week": [float((index + 1) % 7) for index in range(4)],
            "target_baguette_t_plus_1": [float(index + 8) for index in range(4)],
        }
    )
    test_df = pd.DataFrame(
        {
            "origin_date": ["2021-03-01", "2021-03-02", "2021-03-03"],
            "target_date": ["2021-03-02", "2021-03-03", "2021-03-04"],
            "exog_sales_1_lag1": [12.0, 13.0, 14.0],
            "exog_calendar_target_day_of_week": [0.0, 1.0, 2.0],
            "target_baguette_t_plus_1": [12.0, 13.0, 14.0],
        }
    )
    payload = {
        "best_params": {
            "alpha": 0.1,
            "l1_ratio": 0.9,
            "fit_intercept": True,
            "max_iter": 5000,
        },
        "model_family": "ElasticNet",
    }
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    best_params_json.write_text(json.dumps(payload), encoding="utf-8")
    statistical_baselines_json.write_text(
        json.dumps(
            {
                "baseline_count": 2,
                "baselines_ranked_by_mae": [
                    {"name": "same_weekday_mean_4", "mae": 1.0, "rank_mae": 1},
                    {"name": "naive_lag_1", "mae": 2.0, "rank_mae": 2},
                ],
                "best_baseline": {
                    "name": "same_weekday_mean_4",
                    "mae": 1.0,
                    "rank_mae": 1,
                    "prediction_rows": [
                        {
                            "origin_date": "2021-03-01",
                            "target_date": "2021-03-02",
                            "actual": 12.0,
                            "prediction": 11.0,
                            "absolute_error": 1.0,
                        },
                        {
                            "origin_date": "2021-03-02",
                            "target_date": "2021-03-03",
                            "actual": 13.0,
                            "prediction": 13.0,
                            "absolute_error": 0.0,
                        },
                        {
                            "origin_date": "2021-03-03",
                            "target_date": "2021-03-04",
                            "actual": 14.0,
                            "prediction": 15.0,
                            "absolute_error": 1.0,
                        },
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_evaluation(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        best_params_json=best_params_json,
        metrics_json=metrics_json,
        predictions_csv=predictions_csv,
        predictions_plot_png=predictions_plot_png,
        diagnostics_json=diagnostics_json,
        model_card_json=model_card_json,
        statistical_baselines_json=statistical_baselines_json,
        final_model_pkl=final_model_pkl,
        residuals_plot_png=residuals_plot_png,
        residuals_qq_png=residuals_qq_png,
        residuals_acf_pacf_png=residuals_acf_pacf_png,
    )

    metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    diagnostics_payload = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    model_card_payload = json.loads(model_card_json.read_text(encoding="utf-8"))
    statistical_baselines_payload = json.loads(statistical_baselines_json.read_text(encoding="utf-8"))
    predictions_df = pd.read_csv(predictions_csv)

    assert summary["test_rows"] == 3
    assert summary["feature_count"] == 2
    assert summary["model_family"] == "ElasticNet"
    assert list(metrics_payload.keys())[0] == "business_impact"
    assert "mae" in metrics_payload
    assert "rmse" in metrics_payload
    assert "smape" in metrics_payload
    assert "mase" in metrics_payload
    assert "naive_lag_1_mae" in metrics_payload
    assert "seasonal_naive_lag_7_mae" in metrics_payload
    assert "absolute_mae_saved_vs_best_baseline" in metrics_payload
    assert "rounded_model_mae_baguettes" in metrics_payload
    assert "rounded_best_statistical_baseline_mae_baguettes" in metrics_payload
    assert "rounded_mae_gap_baguettes_vs_best_baseline" in metrics_payload
    assert "baguette_unit_cost_eur" in metrics_payload
    assert "test_day_count" in metrics_payload
    assert "estimated_savings_eur_vs_best_baseline" in metrics_payload
    assert "best_statistical_baseline_name" in metrics_payload
    assert "ljung_box_pvalue" in diagnostics_payload
    assert statistical_baselines_payload["baseline_count"] > 0
    assert statistical_baselines_payload["baselines_ranked_by_mae"][0]["name"]
    assert list(model_card_payload.keys())[0] == "business_impact"
    assert model_card_payload["best_params"]["alpha"] == 0.1
    assert model_card_payload["best_params"]["l1_ratio"] == 0.9
    assert model_card_payload["statistical_baselines_path"] == str(statistical_baselines_json)
    assert model_card_payload["best_statistical_baseline"]["name"] == statistical_baselines_payload["baselines_ranked_by_mae"][0]["name"]
    assert model_card_payload["baseline_savings"]["best_statistical_baseline_name"] == "same_weekday_mean_4"
    assert model_card_payload["baseline_savings"]["baguette_unit_cost_eur"] == 1.0
    assert model_card_payload["baseline_savings"]["test_day_count"] == 3
    assert model_card_payload["baseline_savings"]["rounded_best_statistical_baseline_mae_baguettes"] == 1
    assert model_card_payload["baseline_savings"]["rounded_model_mae_baguettes"] == 0
    assert model_card_payload["baseline_savings"]["rounded_mae_gap_baguettes_vs_best_baseline"] == 1
    assert model_card_payload["baseline_savings"]["estimated_savings_eur_vs_best_baseline"] == 3.0
    assert model_card_payload["business_impact"]["estimated_savings_eur_vs_best_baseline"] == 3.0
    assert model_card_payload["business_impact"]["rounded_mae_gap_baguettes_vs_best_baseline"] == 1
    assert len(predictions_df) == 3
    assert "best_statistical_baseline_prediction" in predictions_df.columns
    assert "absolute_error_saved_vs_best_baseline" in predictions_df.columns
    assert predictions_plot_png.exists()
    assert residuals_plot_png.exists()
    assert residuals_qq_png.exists()
    assert residuals_acf_pacf_png.exists()
    assert final_model_pkl.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
