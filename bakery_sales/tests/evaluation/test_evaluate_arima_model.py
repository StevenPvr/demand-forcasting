from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.arima_time_series import rolling_forecast
from src.evaluation.evaluate_arima_model import rolling_origin_predictions, run_evaluation


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


def test_rolling_forecast_refits_sarima_on_full_available_history() -> None:
    history_df = _product_frame("BAGUETTE", "2021-01-01", 14)
    future_df = _product_frame("BAGUETTE", "2021-01-15", 2)

    predictions_df = rolling_forecast(
        history_df=history_df,
        future_df=future_df,
        date_column="date",
        target_column="quantity",
        params={"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "trend": "n"},
    )

    assert predictions_df["train_rows_used"].tolist() == [14, 15]
    assert {"used_order", "used_seasonal_order", "used_trend"} <= set(predictions_df.columns)


def test_run_evaluation_writes_per_product_arima_outputs(tmp_path: Path) -> None:
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
    test_df = pd.concat(
        [
            _product_frame("BAGUETTE", "2021-02-04", 3),
            _product_frame("CROISSANT", "2021-02-04", 3),
        ],
        ignore_index=True,
    )
    payload = {
        "model_family": "ARIMA_BY_PRODUCT",
        "product_models": {
            "BAGUETTE": {
                "best_params": {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "trend": "n"},
            },
            "CROISSANT": {
                "best_params": {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "trend": "n"},
            },
        },
    }
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    best_params_json.write_text(json.dumps(payload), encoding="utf-8")

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
    )

    metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    model_card_payload = json.loads(model_card_json.read_text(encoding="utf-8"))
    predictions_df = pd.read_csv(predictions_csv)

    assert summary["model_family"] == "ARIMA_BY_PRODUCT"
    assert summary["feature_count"] == 0
    assert summary["product_count"] == 2
    assert metrics_payload["product_count"] == 2
    assert set(metrics_payload["per_product_metrics"]) == {"BAGUETTE", "CROISSANT"}
    assert model_card_payload["model_family"] == "ARIMA_BY_PRODUCT"
    assert model_card_payload["feature_columns"] == []
    assert len(predictions_df) == 6
    assert set(predictions_df["product"]) == {"BAGUETTE", "CROISSANT"}
    assert final_model_pkl.exists()
    assert predictions_plot_png.exists()
    assert residuals_plot_png.exists()
    assert residuals_qq_png.exists()
    assert residuals_acf_pacf_png.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
