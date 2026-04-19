from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sarimax_time_series import rolling_forecast
from src.evaluation.evaluate_sarimax_model import run_evaluation


def test_rolling_forecast_refits_sarimax_on_full_available_history() -> None:
    history_df = pd.DataFrame(
        {
            "origin_date": [f"2021-01-{index + 1:02d}" for index in range(14)],
            "target_date": [f"2021-01-{index + 2:02d}" for index in range(14)],
            "exog_sales_1_lag1": [20.0 + float(index % 7) for index in range(14)],
            "target_baguette_t_plus_1": [20.0 + float(index % 7) for index in range(14)],
        }
    )
    future_df = pd.DataFrame(
        {
            "origin_date": ["2021-01-15", "2021-01-16"],
            "target_date": ["2021-01-16", "2021-01-17"],
            "exog_sales_1_lag1": [21.0, 22.0],
            "target_baguette_t_plus_1": [21.0, 22.0],
        }
    )

    predictions_df = rolling_forecast(
        history_df=history_df,
        future_df=future_df,
        feature_columns=["exog_sales_1_lag1"],
        target_column="target_baguette_t_plus_1",
        params={"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 7], "trend": "c"},
    )

    assert predictions_df["train_rows_used"].tolist() == [14, 15]
    assert {"used_order", "used_seasonal_order", "used_trend"} <= set(predictions_df.columns)


def test_run_evaluation_writes_sarimax_outputs(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    best_params_json = tmp_path / "best_params.json"
    metrics_json = tmp_path / "metrics.json"
    predictions_csv = tmp_path / "predictions.csv"
    predictions_plot_png = tmp_path / "predictions.png"
    diagnostics_json = tmp_path / "diagnostics.json"
    model_card_json = tmp_path / "model_card.json"
    final_model_pkl = tmp_path / "final_model.pkl"
    residuals_plot_png = tmp_path / "residuals.png"
    residuals_qq_png = tmp_path / "residuals_qq.png"
    residuals_acf_pacf_png = tmp_path / "residuals_acf_pacf.png"
    statistical_baselines_json = tmp_path / "statistical_baselines.json"

    train_df = pd.DataFrame(
        {
            "origin_date": [f"2021-01-{index + 1:02d}" for index in range(28)],
            "target_date": [f"2021-01-{index + 2:02d}" for index in range(28)],
            "exog_sales_1_lag1": [20.0 + float(index % 7) for index in range(28)],
            "target_baguette_t_plus_1": [20.0 + float(index % 7) for index in range(28)],
        }
    )
    val_df = pd.DataFrame(
        {
            "origin_date": [f"2021-02-{index + 1:02d}" for index in range(6)],
            "target_date": [f"2021-02-{index + 2:02d}" for index in range(6)],
            "exog_sales_1_lag1": [21.0 + float(index % 7) for index in range(6)],
            "target_baguette_t_plus_1": [21.0 + float(index % 7) for index in range(6)],
        }
    )
    test_df = pd.DataFrame(
        {
            "origin_date": [f"2021-03-{index + 1:02d}" for index in range(3)],
            "target_date": [f"2021-03-{index + 2:02d}" for index in range(3)],
            "exog_sales_1_lag1": [22.0 + float(index % 7) for index in range(3)],
            "target_baguette_t_plus_1": [22.0 + float(index % 7) for index in range(3)],
        }
    )
    payload = {
        "best_params": {"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 7], "trend": "c"},
        "model_family": "SARIMAX",
    }
    baseline_payload = {
        "best_baseline": {
            "name": "naive_lag_1",
            "mae": 5.4,
            "prediction_rows": [
                {
                    "origin_date": row["origin_date"],
                    "target_date": row["target_date"],
                    "prediction": 20.0,
                    "absolute_error": abs(float(row["target_baguette_t_plus_1"]) - 20.0),
                }
                for row in test_df.to_dict(orient="records")
            ],
        }
    }
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    best_params_json.write_text(json.dumps(payload), encoding="utf-8")
    statistical_baselines_json.write_text(json.dumps(baseline_payload), encoding="utf-8")

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
        final_model_pkl=final_model_pkl,
        residuals_plot_png=residuals_plot_png,
        residuals_qq_png=residuals_qq_png,
        residuals_acf_pacf_png=residuals_acf_pacf_png,
        statistical_baselines_json=statistical_baselines_json,
    )

    model_card_payload = json.loads(model_card_json.read_text(encoding="utf-8"))
    predictions_df = pd.read_csv(predictions_csv)
    metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))

    assert summary["model_family"] == "SARIMAX"
    assert summary["feature_count"] == 1
    assert "estimated_savings_eur_vs_best_baseline" in summary
    assert model_card_payload["model_family"] == "SARIMAX"
    assert model_card_payload["feature_columns"] == ["exog_sales_1_lag1"]
    assert model_card_payload["best_statistical_baseline"]["name"] == "naive_lag_1"
    assert "baseline_savings" in model_card_payload
    assert "business_impact" in metrics_payload
    assert len(predictions_df) == 3
    assert final_model_pkl.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
