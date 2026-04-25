from __future__ import annotations

"""Evaluation du meilleur modele SARIMAX sur le split test."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.constants import CSV_ENCODING, DEFAULT_EVALUATION_JOBS, TARGET_COLUMN
from src.evaluation.evaluate_elasticnet_model import (
    _baseline_comparison_payload,
    _configure_plot_style,
    _diagnostics_payload,
    _load_best_payload,
    _load_split,
    _load_statistical_baselines_payload,
    _log_progress,
    _metrics_payload,
    _plot_actual_vs_predicted,
    _plot_residuals,
    _plot_residuals_acf_pacf,
    _plot_residuals_qq,
)
from src.evaluation.paths_sarimax import STATISTICAL_BASELINES_JSON
from src.sarimax_time_series import (
    feature_columns as shared_feature_columns,
    fit_sarimax_model,
    rolling_forecast,
    save_sarimax_model,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes exogenes du dataset journalier."""

    return shared_feature_columns(dataset_df, TARGET_COLUMN)


def rolling_origin_predictions(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_payload: dict[str, Any],
    n_jobs: int = DEFAULT_EVALUATION_JOBS,
) -> pd.DataFrame:
    """Predictions rolling-origin en reentrainant SARIMAX avant chaque ligne test."""

    del n_jobs
    selected_feature_columns = _feature_columns(history_df)
    return rolling_forecast(
        history_df=history_df,
        future_df=test_df,
        feature_columns=selected_feature_columns,
        target_column=TARGET_COLUMN,
        params=best_payload["best_params"],
        progress_callback=_log_progress,
    )


def run_evaluation(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    best_params_json: Path,
    metrics_json: Path,
    predictions_csv: Path,
    predictions_plot_png: Path,
    diagnostics_json: Path,
    model_card_json: Path,
    final_model_pkl: Path,
    residuals_plot_png: Path,
    residuals_qq_png: Path,
    residuals_acf_pacf_png: Path,
    statistical_baselines_json: Path | None = STATISTICAL_BASELINES_JSON,
    n_jobs: int = DEFAULT_EVALUATION_JOBS,
) -> dict[str, int | float | str]:
    """Entraine sur train+val puis evalue SARIMAX en rolling-origin sur le test."""

    start_time = time.perf_counter()
    _configure_plot_style()
    train_df = _load_split(train_csv)
    val_df = _load_split(val_csv)
    test_df = _load_split(test_csv)
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    best_payload = _load_best_payload(best_params_json)
    selected_feature_columns = _feature_columns(history_df)
    predictions_df = rolling_origin_predictions(history_df=history_df, test_df=test_df, best_payload=best_payload, n_jobs=n_jobs)
    metrics_payload = _metrics_payload(history_df, predictions_df)
    diagnostics_payload = _diagnostics_payload(predictions_df)
    baselines_payload = _load_statistical_baselines_payload(statistical_baselines_json)
    predictions_df, baseline_savings_payload = _baseline_comparison_payload(predictions_df, baselines_payload)
    if baseline_savings_payload is not None:
        metrics_payload.update(baseline_savings_payload)
    business_impact_payload: dict[str, Any] | None = None
    if baseline_savings_payload is not None:
        business_impact_payload = {
            "estimated_savings_eur_vs_best_baseline": baseline_savings_payload["estimated_savings_eur_vs_best_baseline"],
            "rounded_mae_gap_baguettes_vs_best_baseline": baseline_savings_payload["rounded_mae_gap_baguettes_vs_best_baseline"],
            "rounded_model_mae_baguettes": baseline_savings_payload["rounded_model_mae_baguettes"],
            "rounded_best_statistical_baseline_mae_baguettes": baseline_savings_payload["rounded_best_statistical_baseline_mae_baguettes"],
            "baguette_unit_cost_eur": baseline_savings_payload["baguette_unit_cost_eur"],
            "test_day_count": baseline_savings_payload["test_day_count"],
        }
    final_model = fit_sarimax_model(
        history_df=history_df,
        feature_columns=selected_feature_columns,
        target_column=TARGET_COLUMN,
        params=best_payload["best_params"],
    )
    save_sarimax_model(final_model, final_model_pkl)
    metrics_output_payload: dict[str, Any] = (
        {"business_impact": business_impact_payload, **metrics_payload}
        if business_impact_payload is not None
        else dict(metrics_payload)
    )
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics_output_payload, indent=2), encoding=CSV_ENCODING)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False, encoding=CSV_ENCODING)
    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json.write_text(json.dumps(diagnostics_payload, indent=2), encoding=CSV_ENCODING)
    model_card_payload: dict[str, Any] = {}
    if business_impact_payload is not None:
        model_card_payload["business_impact"] = business_impact_payload
    model_card_payload.update(best_payload)
    model_card_payload["model_family"] = "SARIMAX"
    model_card_payload["feature_count"] = int(len(selected_feature_columns))
    model_card_payload["feature_columns"] = selected_feature_columns
    model_card_payload["metrics"] = metrics_output_payload
    model_card_payload["diagnostics"] = diagnostics_payload
    model_card_payload["final_model_path"] = str(final_model_pkl)
    model_card_payload["predictions_path"] = str(predictions_csv)
    if baselines_payload is not None and statistical_baselines_json is not None:
        model_card_payload["statistical_baselines_path"] = str(statistical_baselines_json)
    if baseline_savings_payload is not None:
        model_card_payload["best_statistical_baseline"] = {
            "name": baseline_savings_payload["best_statistical_baseline_name"],
            "mae": baseline_savings_payload["best_statistical_baseline_mae"],
        }
        model_card_payload["baseline_savings"] = baseline_savings_payload
    model_card_json.parent.mkdir(parents=True, exist_ok=True)
    model_card_json.write_text(json.dumps(model_card_payload, indent=2), encoding=CSV_ENCODING)
    _plot_actual_vs_predicted(predictions_df, predictions_plot_png)
    _plot_residuals(predictions_df, residuals_plot_png)
    _plot_residuals_qq(predictions_df, residuals_qq_png)
    _plot_residuals_acf_pacf(predictions_df, residuals_acf_pacf_png)
    LOGGER.info(
        "SARIMAX evaluation completed in %.3f seconds with MAE=%.6f RMSE=%.6f sMAPE=%.6f",
        time.perf_counter() - start_time,
        float(metrics_payload["mae"]),
        float(metrics_payload["rmse"]),
        float(metrics_payload["smape"]),
    )
    if baseline_savings_payload is not None:
        LOGGER.info(
            "Business impact vs best baseline: rounded model MAE=%d, rounded baseline MAE=%d, gap=%d baguettes/day, unit_cost=%.2f EUR, test_days=%d, estimated_savings=%.2f EUR",
            baseline_savings_payload["rounded_model_mae_baguettes"],
            baseline_savings_payload["rounded_best_statistical_baseline_mae_baguettes"],
            baseline_savings_payload["rounded_mae_gap_baguettes_vs_best_baseline"],
            baseline_savings_payload["baguette_unit_cost_eur"],
            baseline_savings_payload["test_day_count"],
            baseline_savings_payload["estimated_savings_eur_vs_best_baseline"],
        )
    return {
        "model_family": "SARIMAX",
        "test_rows": int(len(predictions_df)),
        "feature_count": int(len(selected_feature_columns)),
        "mae": float(metrics_payload["mae"]),
        "rmse": float(metrics_payload["rmse"]),
        "smape": float(metrics_payload["smape"]),
        "absolute_mae_saved_vs_best_baseline": float(metrics_payload.get("absolute_mae_saved_vs_best_baseline", 0.0)),
        "estimated_savings_eur_vs_best_baseline": float(metrics_payload.get("estimated_savings_eur_vs_best_baseline", 0.0)),
    }
