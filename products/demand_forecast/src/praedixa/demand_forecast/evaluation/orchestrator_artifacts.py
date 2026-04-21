from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

from praedixa.demand_forecast.contracts.targets import build_target_contract_metadata
from praedixa.demand_forecast.evaluation.bakery_metrics import compute_metrics_payload
from praedixa.demand_forecast.evaluation.bakery_metrics import (
    compute_probabilistic_metrics_payload,
)
from praedixa.demand_forecast.evaluation.bakery_style import (
    DEFAULT_UNIT_COST_EUR,
    FIELD_BASELINE_NAME,
    build_simple_economic_gain_payload,
    build_statistical_baselines_payload,
)
from praedixa.demand_forecast.evaluation.bakery_enrichment import (
    enrich_predictions_with_best_baseline,
)
from praedixa.demand_forecast.evaluation.orchestrator_context import (
    EvaluationPreparedContext,
)
from praedixa.demand_forecast.evaluation.reporting import (
    build_canonical_predictions_frame,
    build_diagnostics_payload,
    json_dump,
)
from praedixa.demand_forecast.training.constants import (
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
)


@dataclass(frozen=True)
class EvaluationRunArtifacts:
    output_paths: dict[str, Path]
    metrics_payload: dict[str, Any]
    probabilistic_metrics_payload: dict[str, Any]
    baselines_payload: dict[str, Any] | None
    predictions_df: pd.DataFrame
    probabilistic_predictions_df: pd.DataFrame
    diagnostics_payload: dict[str, Any]
    economic_gain_payload: dict[str, Any]
    daily_report_df: pd.DataFrame
    final_model: Any
    model_card_payload: dict[str, Any]
    evaluation_metadata_payload: dict[str, Any]


def evaluation_output_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "metrics_json": target_dir / "foundation_tft_test_metrics.json",
        "probabilistic_metrics_json": target_dir / "foundation_tft_probabilistic_metrics.json",
        "statistical_baselines_json": target_dir / "foundation_tft_statistical_baselines.json",
        "predictions_csv": target_dir / "foundation_tft_test_predictions.csv",
        "probabilistic_predictions_csv": target_dir / "foundation_tft_probabilistic_predictions.csv",
        "diagnostics_json": target_dir / "foundation_tft_test_diagnostics.json",
        "model_card_json": target_dir / "foundation_tft_model_card.json",
        "final_model": target_dir / "foundation_tft_final_model.pt",
        "evaluation_metadata": target_dir / "foundation_tft_evaluation_metadata.json",
        "economic_gain_json": target_dir / "foundation_tft_economic_gain_vs_best_baseline.json",
        "daily_refit_metrics_csv": target_dir / "foundation_tft_daily_refit_metrics.csv",
        "actual_vs_predicted_plot": target_dir / "foundation_tft_actual_vs_predicted.png",
        "residuals_plot": target_dir / "foundation_tft_residuals.png",
    }


def _evaluation_metrics_payload(
    *,
    context: EvaluationPreparedContext,
    canonical_predictions_df: pd.DataFrame,
    baseline_savings_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics_payload = {
        "model_family": "foundation_tft",
        **compute_metrics_payload(context.history_reference, canonical_predictions_df),
    }
    if baseline_savings_payload is not None:
        metrics_payload["business_impact"] = baseline_savings_payload
    return metrics_payload


def _evaluation_diagnostics_payload(
    *,
    context: EvaluationPreparedContext,
    probabilistic_predictions_df: pd.DataFrame,
    probabilistic_metrics_payload: dict[str, Any],
    best_iteration: int,
) -> dict[str, Any]:
    return build_diagnostics_payload(
        probabilistic_predictions_df,
        evaluation_mode=context.evaluation_mode,
        feature_count=len(context.feature_cols),
        best_iteration=best_iteration,
        overlap_metadata=context.overlap_metadata,
        daily_refit=True,
        overall_metrics=cast(dict[str, Any] | None, probabilistic_metrics_payload.get("overall_metrics")),
    )


def _baseline_prediction_payloads(
    *,
    context: EvaluationPreparedContext,
    predictions_df: pd.DataFrame,
) -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    baselines_payload = build_statistical_baselines_payload(
        context.history_reference,
        context.scored_reference_test,
    )
    probabilistic_predictions_df = predictions_df.copy()
    canonical_predictions_df = build_canonical_predictions_frame(probabilistic_predictions_df)
    canonical_predictions_df, baseline_savings_payload = enrich_predictions_with_best_baseline(
        canonical_predictions_df,
        baselines_payload,
        field_baseline_name=FIELD_BASELINE_NAME,
        unit_cost_eur=DEFAULT_UNIT_COST_EUR,
    )
    return (
        baselines_payload,
        canonical_predictions_df,
        probabilistic_predictions_df,
        baseline_savings_payload,
    )


def build_evaluation_payloads(
    *,
    context: EvaluationPreparedContext,
    predictions_df: pd.DataFrame,
    best_iteration: int,
) -> tuple[
    dict[str, Any] | None,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    baselines_payload, canonical_predictions_df, probabilistic_predictions_df, baseline_savings_payload = (
        _baseline_prediction_payloads(
            context=context,
            predictions_df=predictions_df,
        )
    )
    metrics_payload = _evaluation_metrics_payload(
        context=context,
        canonical_predictions_df=canonical_predictions_df,
        baseline_savings_payload=baseline_savings_payload,
    )
    probabilistic_metrics_payload = compute_probabilistic_metrics_payload(probabilistic_predictions_df)
    diagnostics_payload = _evaluation_diagnostics_payload(
        context=context,
        probabilistic_predictions_df=probabilistic_predictions_df,
        probabilistic_metrics_payload=probabilistic_metrics_payload,
        best_iteration=best_iteration,
    )
    economic_gain_payload = build_simple_economic_gain_payload(
        canonical_predictions_df,
        metrics_payload,
    )
    return (
        baselines_payload,
        canonical_predictions_df,
        probabilistic_predictions_df,
        baseline_savings_payload,
        metrics_payload,
        probabilistic_metrics_payload,
        diagnostics_payload,
        economic_gain_payload,
    )


def build_evaluation_metadata_payload(
    *,
    context: EvaluationPreparedContext,
    request: Any,
    best_iteration: int,
    final_model: Any,
) -> dict[str, Any]:
    return {
        "evaluation_mode": context.evaluation_mode,
        "feature_cols": context.feature_cols,
        "constant_feature_cols": context.constant_feature_cols,
        "identifier_feature_cols": context.identifier_feature_cols,
        "excluded_risky_feature_cols": list(DEFAULT_EXCLUDED_RISKY_FEATURE_COLS),
        "missing_in_test_feature_cols": context.missing_in_test_feature_cols,
        "best_iteration": int(best_iteration),
        "daily_refit": True,
        "train_rows": int(len(context.train_frame)),
        "valid_rows": int(len(context.valid_frame)),
        "test_rows": int(len(context.test_frame)),
        "history_reference_rows": int(len(context.history_reference)),
        "reference_test_rows": int(len(context.scored_reference_test)),
        "reference_train_csv": str(request.bakery_reference_train_csv),
        "reference_val_csv": str(request.bakery_reference_val_csv),
        "reference_test_csv": str(request.bakery_reference_test_csv),
        "duckdb_path": str(request.duckdb_path),
        "gold_table": request.gold_table,
        "train_sample_fraction": float(request.train_sample_fraction),
        "tuning_sample_fraction": float(request.tuning_sample_fraction),
        "final_model_runtime_profile": getattr(final_model, "runtime_profile", None),
        "final_model_system_info": getattr(final_model, "system_info", {}),
        "final_model_git_sha": getattr(final_model, "git_sha", None),
        **build_target_contract_metadata(context.target_contract),
        "reference_overlap": context.overlap_metadata,
    }


def build_evaluation_model_card_payload(
    *,
    context: EvaluationPreparedContext,
    metrics_payload: dict[str, Any],
    probabilistic_metrics_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any],
    baseline_savings_payload: dict[str, Any] | None,
    economic_gain_payload: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "model_family": "foundation_tft",
        "feature_count": int(len(context.feature_cols)),
        "feature_columns": context.feature_cols,
        "metrics": metrics_payload,
        "diagnostics": diagnostics_payload,
        "forecast_output_contract": {
            "point_forecast_column": diagnostics_payload["probabilistic_summary"]["point_forecast_column"],
            "median_forecast_column": diagnostics_payload["probabilistic_summary"]["median_forecast_column"],
            "quantile_columns": diagnostics_payload["probabilistic_summary"]["quantile_columns"],
            "intervals": diagnostics_payload["probabilistic_summary"]["intervals"],
        },
        "probabilistic_forecast": diagnostics_payload["probabilistic_summary"],
        "probabilistic_metrics_path": str(output_paths["probabilistic_metrics_json"]),
        "probabilistic_predictions_path": str(output_paths["probabilistic_predictions_csv"]),
        "probabilistic_metrics": probabilistic_metrics_payload,
        "business_impact": baseline_savings_payload,
        "economic_gain": economic_gain_payload,
        "target_contract": build_target_contract_metadata(context.target_contract),
        "final_model_path": str(output_paths["final_model"]),
        "predictions_path": str(output_paths["predictions_csv"]),
        "statistical_baselines_path": str(output_paths["statistical_baselines_json"]),
        "evaluation_metadata_path": str(output_paths["evaluation_metadata"]),
        "daily_refit_metrics_path": str(output_paths["daily_refit_metrics_csv"]),
    }


def log_evaluation_completion(
    *,
    evaluation_mode: str,
    metrics_payload: dict[str, Any],
    baseline_savings_payload: dict[str, Any] | None,
    economic_gain_payload: dict[str, Any],
    logger: logging.Logger,
) -> None:
    overall_metrics = metrics_payload["overall_metrics"]
    logger.info(
        "Evaluation complete: mode=%s test_rows=%s mae=%.6f rmse=%.6f smape=%.6f coverage_80=%s coverage_95=%s best_baseline=%s estimated_savings=%.2f",
        evaluation_mode,
        overall_metrics["test_rows"],
        overall_metrics["mae"],
        overall_metrics["rmse"],
        overall_metrics["smape"],
        overall_metrics.get("coverage_80"),
        overall_metrics.get("coverage_95"),
        baseline_savings_payload["best_statistical_baseline_name"] if baseline_savings_payload is not None else None,
        economic_gain_payload["total_estimated_savings_eur_vs_best_baselines"],
    )
    for product_name, product_gain in economic_gain_payload["per_product_gain"].items():
        logger.info(
            "Economic gain by product: product=%s best_baseline=%s model_mae=%.6f baseline_mae=%.6f absolute_mae_saved=%.6f estimated_savings_eur=%.2f",
            product_name,
            product_gain["best_baseline_name"],
            product_gain["model_mae"],
            product_gain["best_baseline_mae"],
            product_gain["absolute_mae_saved_vs_best_baseline"],
            product_gain["estimated_savings_eur_vs_best_baseline"],
        )


def persist_evaluation_outputs(
    *,
    artifacts: EvaluationRunArtifacts,
    save_model_fn: Callable[[Any, Path], Path],
    plot_actual_vs_predicted_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_fn: Callable[[pd.DataFrame, Path], None],
) -> dict[str, Path]:
    output_paths = artifacts.output_paths
    json_dump(output_paths["metrics_json"], artifacts.metrics_payload)
    json_dump(output_paths["probabilistic_metrics_json"], artifacts.probabilistic_metrics_payload)
    json_dump(output_paths["statistical_baselines_json"], artifacts.baselines_payload or {})
    artifacts.predictions_df.to_csv(output_paths["predictions_csv"], index=False)
    artifacts.probabilistic_predictions_df.to_csv(output_paths["probabilistic_predictions_csv"], index=False)
    json_dump(output_paths["diagnostics_json"], artifacts.diagnostics_payload)
    json_dump(output_paths["economic_gain_json"], artifacts.economic_gain_payload)
    artifacts.daily_report_df.to_csv(output_paths["daily_refit_metrics_csv"], index=False)
    save_model_fn(artifacts.final_model, output_paths["final_model"])
    json_dump(output_paths["evaluation_metadata"], artifacts.evaluation_metadata_payload)
    plot_actual_vs_predicted_fn(artifacts.predictions_df, output_paths["actual_vs_predicted_plot"])
    plot_residuals_fn(artifacts.predictions_df, output_paths["residuals_plot"])
    json_dump(output_paths["model_card_json"], artifacts.model_card_payload)
    return output_paths
