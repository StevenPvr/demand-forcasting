from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from praedixa.demand_forecast.evaluation.orchestrator_artifacts import (
    EvaluationRunArtifacts,
    build_evaluation_feature_manifest_payload,
    build_evaluation_metadata_payload,
    build_evaluation_model_card_payload,
    build_evaluation_payloads,
    build_evaluation_split_manifest_payload,
    build_promotable_bundle_manifest_payload,
    evaluation_output_paths,
    log_evaluation_completion,
    persist_evaluation_outputs,
)
from praedixa.demand_forecast.contracts.targets import build_target_contract_metadata
from praedixa.demand_forecast.evaluation.orchestrator_context import (
    load_evaluation_frames,
    prepare_evaluation_context,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    filter_training_eligible_rows,
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
        [context.train_frame, context.valid_frame],
        ignore_index=True,
    )
    fit_frame = filter_training_eligible_rows(
        fit_frame,
        label="evaluation_final_model_train_valid",
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
        evaluation_dataset_source=request.evaluation_dataset_source,
        model_backend=request.model_backend,
        logger=logger,
    )
    return prepare_evaluation_context(
        loaded=loaded,
        requested_target_col=request.target_col,
        best_params_path=request.best_params_path,
        model_backend=request.model_backend,
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
        duckdb_path=request.duckdb_path,
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
    (
        baselines_payload,
        predictions_df,
        probabilistic_predictions_df,
        baseline_savings_payload,
        metrics_payload,
        probabilistic_metrics_payload,
        diagnostics_payload,
        economic_gain_payload,
    ) = payloads
    output_paths = evaluation_output_paths(
        target_dir,
        model_family=_model_family_for_paths(str(context.model_backend)),
    )
    feature_manifest_payload, feature_roles_payload = (
        build_evaluation_feature_manifest_payload(context=context)
    )
    split_manifest_payload = build_evaluation_split_manifest_payload(context=context)
    target_contract_payload = build_target_contract_metadata(context.target_contract)
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
        feature_manifest_payload=feature_manifest_payload,
        feature_roles_payload=feature_roles_payload,
        split_manifest_payload=split_manifest_payload,
        target_contract_payload=target_contract_payload,
        metrics_payload=metrics_payload,
        probabilistic_metrics_payload=probabilistic_metrics_payload,
        baselines_payload=baselines_payload,
        predictions_df=predictions_df,
        probabilistic_predictions_df=probabilistic_predictions_df,
        diagnostics_payload=diagnostics_payload,
        economic_gain_payload=economic_gain_payload,
        daily_report_df=daily_report_df,
        final_model=final_model,
        interpretability_payload=getattr(final_model, "interpretability_payload", None),
        model_card_payload=model_card_payload,
        evaluation_metadata_payload=evaluation_metadata_payload,
        promotable_bundle_manifest_payload=build_promotable_bundle_manifest_payload(
            output_paths=output_paths,
            context=context,
            final_model=final_model,
        ),
    )


def _model_family_for_paths(model_backend: str) -> str:
    if model_backend in {"xgboost", "chronos2", "moirai", "timesfm"}:
        return model_backend
    return "foundation_tft"


def _weight_experiment_label(multiplier: float) -> str:
    label = f"{float(multiplier):g}".replace(".", "_")
    return f"bakery_effective_share_{label}"


def _context_with_bakery_weight(context: Any, *, multiplier: float) -> Any:
    target_share = min(max(float(multiplier) / 100.0, 1e-6), 0.999999)
    best_params = {
        **context.best_params,
        "daily_refit_bakery_weight_strategy": "target_effective_share",
        "daily_refit_bakery_min_effective_share": 0.50,
        "daily_refit_bakery_max_effective_share": target_share,
        "daily_refit_bakery_weight_experiment": _weight_experiment_label(multiplier),
    }
    return replace(context, best_params=best_params)


def _run_weighted_evaluation_branch(
    *,
    request: Any,
    target_dir: Path,
    context: Any,
    multiplier: float,
    logger: logging.Logger,
    evaluate_daily_refit_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame, int]],
    fit_final_model_fn: Callable[..., Any],
    save_model_fn: Callable[[Any, Path], Path],
    plot_actual_vs_predicted_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_qq_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_acf_pacf_fn: Callable[[pd.DataFrame, Path], None],
) -> dict[str, Path]:
    experiment_label = _weight_experiment_label(multiplier)
    branch_context = _context_with_bakery_weight(context, multiplier=multiplier)
    branch_dir = target_dir / experiment_label
    branch_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting weighted bakery refit evaluation branch: label=%s multiplier=%.3f output_dir=%s",
        experiment_label,
        multiplier,
        branch_dir,
    )
    predictions_df, daily_report_df, best_iteration = _run_daily_refit(
        context=branch_context,
        logger=logger,
        evaluate_daily_refit_fn=evaluate_daily_refit_fn,
    )
    artifacts = _built_evaluation_artifacts(
        request=request,
        target_dir=branch_dir,
        context=branch_context,
        predictions_df=predictions_df,
        daily_report_df=daily_report_df,
        best_iteration=best_iteration,
        fit_final_model_fn=fit_final_model_fn,
    )
    outputs = persist_evaluation_outputs(
        artifacts=artifacts,
        save_model_fn=save_model_fn,
        plot_actual_vs_predicted_fn=plot_actual_vs_predicted_fn,
        plot_residuals_fn=plot_residuals_fn,
        plot_residuals_qq_fn=plot_residuals_qq_fn,
        plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf_fn,
    )
    log_evaluation_completion(
        evaluation_mode=f"{branch_context.evaluation_mode}:{experiment_label}",
        metrics_payload=artifacts.metrics_payload,
        baseline_savings_payload=artifacts.metrics_payload.get("business_impact"),
        economic_gain_payload=artifacts.economic_gain_payload,
        logger=logger,
    )
    return {f"{experiment_label}.{key}": value for key, value in outputs.items()}


def _run_parallel_weighted_evaluations(
    *,
    request: Any,
    target_dir: Path,
    context: Any,
    logger: logging.Logger,
    evaluate_daily_refit_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame, int]],
    fit_final_model_fn: Callable[..., Any],
    save_model_fn: Callable[[Any, Path], Path],
    plot_actual_vs_predicted_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_qq_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_acf_pacf_fn: Callable[[pd.DataFrame, Path], None],
) -> dict[str, Path]:
    multipliers = tuple(
        float(multiplier)
        for multiplier in (request.daily_refit_bakery_weight_multipliers or ())
    )
    workers = max(1, min(len(multipliers), int(request.daily_refit_parallel_workers)))
    logger.info(
        "Starting parallel bakery effective-share evaluations from single loaded dataset: "
        "target_shares=%s workers=%s",
        multipliers,
        workers,
    )
    outputs: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_weighted_evaluation_branch,
                request=request,
                target_dir=target_dir,
                context=context,
                multiplier=multiplier,
                logger=logger,
                evaluate_daily_refit_fn=evaluate_daily_refit_fn,
                fit_final_model_fn=fit_final_model_fn,
                save_model_fn=save_model_fn,
                plot_actual_vs_predicted_fn=plot_actual_vs_predicted_fn,
                plot_residuals_fn=plot_residuals_fn,
                plot_residuals_qq_fn=plot_residuals_qq_fn,
                plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf_fn,
            )
            for multiplier in multipliers
        ]
        for future in as_completed(futures):
            outputs.update(future.result())
    return outputs


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
            interpretability_payload=getattr(
                final_model, "interpretability_payload", None
            ),
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
    require_backend_available_fn: Callable[[str], None],
    plot_actual_vs_predicted_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_qq_fn: Callable[[pd.DataFrame, Path], None],
    plot_residuals_acf_pacf_fn: Callable[[pd.DataFrame, Path], None],
) -> dict[str, Path]:
    require_backend_available_fn("evaluation.build_evaluation_outputs")
    context = _loaded_evaluation_context(request=request, logger=logger)
    if request.daily_refit_bakery_weight_multipliers:
        return _run_parallel_weighted_evaluations(
            request=request,
            target_dir=target_dir,
            context=context,
            logger=logger,
            evaluate_daily_refit_fn=evaluate_daily_refit_fn,
            fit_final_model_fn=fit_final_model_fn,
            save_model_fn=save_model_fn,
            plot_actual_vs_predicted_fn=plot_actual_vs_predicted_fn,
            plot_residuals_fn=plot_residuals_fn,
            plot_residuals_qq_fn=plot_residuals_qq_fn,
            plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf_fn,
        )
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
        plot_residuals_qq_fn=plot_residuals_qq_fn,
        plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf_fn,
    )
    log_evaluation_completion(
        evaluation_mode=context.evaluation_mode,
        metrics_payload=artifacts.metrics_payload,
        baseline_savings_payload=artifacts.metrics_payload.get("business_impact"),
        economic_gain_payload=artifacts.economic_gain_payload,
        logger=logger,
    )
    return persisted_outputs
