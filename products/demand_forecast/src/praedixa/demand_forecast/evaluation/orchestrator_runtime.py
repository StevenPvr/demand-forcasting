from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.evaluation.orchestrator_artifacts import (
    EvaluationRunArtifacts,
    build_evaluation_metadata_payload,
    build_evaluation_model_card_payload,
    build_evaluation_payloads,
    evaluation_output_paths,
    log_evaluation_completion,
    persist_evaluation_outputs,
)
from praedixa.demand_forecast.evaluation.orchestrator_context import (
    load_evaluation_frames,
    prepare_evaluation_context,
)


EvaluationPayloads = tuple[
    dict[str, Any] | None,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]


def _run_daily_refit(
    *,
    context: Any,
    logger: logging.Logger,
    evaluate_daily_refit_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    return evaluate_daily_refit_fn(
        train_frame=context.train_frame,
        valid_frame=context.valid_frame,
        test_frame=context.test_frame,
        feature_cols=context.feature_cols,
        target_contract=context.target_contract,
        model_params=context.best_params,
        logger=logger,
    )


def _fit_final_model(
    *,
    context: Any,
    best_iteration: int,
    fit_final_model_fn: Callable[..., Any],
) -> Any:
    fit_frame = pd.concat(
        [context.train_frame, context.valid_frame, context.test_frame],
        ignore_index=True,
    )
    return fit_final_model_fn(
        fit_frame=fit_frame,
        feature_cols=context.feature_cols,
        target_contract=context.target_contract,
        model_params=context.best_params,
        num_boost_round=best_iteration,
    )


def _loaded_evaluation_context(
    *,
    request: Any,
    logger: logging.Logger,
) -> Any:
    loaded = load_evaluation_frames(
        train_selection_input_path=request.train_selection_input_path,
        train_tuning_input_path=request.train_tuning_input_path,
        val_input_path=request.val_input_path,
        requested_target_col=request.target_col,
        duckdb_path=request.duckdb_path,
        gold_table=request.gold_table,
        train_sample_fraction=request.train_sample_fraction,
        tuning_sample_fraction=request.tuning_sample_fraction,
        bakery_reference_train_csv=request.bakery_reference_train_csv,
        bakery_reference_val_csv=request.bakery_reference_val_csv,
        bakery_reference_test_csv=request.bakery_reference_test_csv,
        logger=logger,
    )
    return prepare_evaluation_context(
        loaded=loaded,
        requested_target_col=request.target_col,
        best_params_path=request.best_params_path,
        logger=logger,
    )


def _built_evaluation_artifacts(
    *,
    request: Any,
    target_dir: Path,
    context: Any,
    predictions_df: pd.DataFrame,
    daily_report_df: pd.DataFrame,
    best_iteration: int,
    fit_final_model_fn: Callable[..., Any],
) -> EvaluationRunArtifacts:
    payloads = build_evaluation_payloads(
        context=context,
        predictions_df=predictions_df,
        best_iteration=best_iteration,
    )
    final_model = _fit_final_model(
        context=context,
        best_iteration=best_iteration,
        fit_final_model_fn=fit_final_model_fn,
    )
    return _evaluation_run_artifacts(
        request=request,
        target_dir=target_dir,
        context=context,
        payloads=payloads,
        daily_report_df=daily_report_df,
        best_iteration=best_iteration,
        final_model=final_model,
    )


def _evaluation_run_artifacts(
    *,
    request: Any,
    target_dir: Path,
    context: Any,
    payloads: EvaluationPayloads,
    daily_report_df: pd.DataFrame,
    best_iteration: int,
    final_model: Any,
) -> EvaluationRunArtifacts:
    baselines_payload, predictions_df, probabilistic_predictions_df, baseline_savings_payload, metrics_payload, probabilistic_metrics_payload, diagnostics_payload, economic_gain_payload = payloads
    output_paths = evaluation_output_paths(target_dir)
    model_card_payload, evaluation_metadata_payload = _artifact_side_payloads(
        request=request,
        context=context,
        output_paths=output_paths,
        probabilistic_metrics_payload=probabilistic_metrics_payload,
        baseline_savings_payload=baseline_savings_payload,
        metrics_payload=metrics_payload,
        diagnostics_payload=diagnostics_payload,
        economic_gain_payload=economic_gain_payload,
        best_iteration=best_iteration,
        final_model=final_model,
    )
    return EvaluationRunArtifacts(
        output_paths=output_paths,
        metrics_payload=metrics_payload,
        probabilistic_metrics_payload=probabilistic_metrics_payload,
        baselines_payload=baselines_payload,
        predictions_df=predictions_df,
        probabilistic_predictions_df=probabilistic_predictions_df,
        diagnostics_payload=diagnostics_payload,
        economic_gain_payload=economic_gain_payload,
        daily_report_df=daily_report_df,
        final_model=final_model,
        model_card_payload=model_card_payload,
        evaluation_metadata_payload=evaluation_metadata_payload,
    )


def _artifact_side_payloads(
    *,
    request: Any,
    context: Any,
    output_paths: dict[str, Path],
    probabilistic_metrics_payload: dict[str, Any],
    baseline_savings_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any],
    economic_gain_payload: dict[str, Any],
    best_iteration: int,
    final_model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        build_evaluation_model_card_payload(
            context=context,
            metrics_payload=metrics_payload,
            probabilistic_metrics_payload=probabilistic_metrics_payload,
            diagnostics_payload=diagnostics_payload,
            baseline_savings_payload=baseline_savings_payload,
            economic_gain_payload=economic_gain_payload,
            output_paths=output_paths,
        ),
        build_evaluation_metadata_payload(
            context=context,
            request=request,
            best_iteration=best_iteration,
            final_model=final_model,
        ),
    )


def run_evaluation_pipeline(
    *,
    request: Any,
    target_dir: Path,
    logger: logging.Logger,
    evaluate_daily_refit_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame, int]],
    fit_final_model_fn: Callable[..., Any],
    save_model_fn: Callable[[Any, Path], Path],
    plot_actual_vs_predicted_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_fn: Callable[[pd.DataFrame, Path], None],
) -> dict[str, Path]:
    raise_if_tft_backend_required("evaluation.build_evaluation_outputs")
    context = _loaded_evaluation_context(request=request, logger=logger)
    predictions_df, daily_report_df, best_iteration = _run_daily_refit(
        context=context,
        logger=logger,
        evaluate_daily_refit_fn=evaluate_daily_refit_fn,
    )
    artifacts = _built_evaluation_artifacts(
        request=request,
        target_dir=target_dir,
        context=context,
        predictions_df=predictions_df,
        daily_report_df=daily_report_df,
        best_iteration=best_iteration,
        fit_final_model_fn=fit_final_model_fn,
    )
    persisted_outputs = persist_evaluation_outputs(
        artifacts=artifacts,
        save_model_fn=save_model_fn,
        plot_actual_vs_predicted_fn=plot_actual_vs_predicted_fn,
        plot_residuals_fn=plot_residuals_fn,
    )
    log_evaluation_completion(
        evaluation_mode=context.evaluation_mode,
        metrics_payload=artifacts.metrics_payload,
        baseline_savings_payload=artifacts.metrics_payload.get("business_impact"),
        economic_gain_payload=artifacts.economic_gain_payload,
        logger=logger,
    )
    return persisted_outputs
