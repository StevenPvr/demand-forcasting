from __future__ import annotations

from pathlib import Path
import logging
from typing import Mapping, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.feature_screening.baselines import evaluate_statistical_baselines
from praedixa.demand_forecast.feature_screening.constants import DEFAULT_PROGRESS_LOG_EVERY
from praedixa.demand_forecast.feature_screening.correlation import (
    build_candidate_memmap,
    filter_correlated_lag_features_memmap,
)
from praedixa.demand_forecast.feature_screening.orchestrator_io import (
    build_final_output_columns,
    json_dump,
    materialize_selected_frames,
    required_scoring_context_columns,
    write_addon_checkpoint,
    write_selected_frames,
)
from praedixa.demand_forecast.feature_screening.orchestrator_models import (
    CorrelationOnlyContext,
    FullLagSelectionContext,
    LagSelectionExecutionContext,
    OptimizeNonLagModelParamsFn,
    RunSingleAddonLagSelectionFn,
)
from praedixa.demand_forecast.feature_screening.splits import build_walk_forward_folds
from praedixa.platform.utils.memory import read_parquet_projected


def run_lag_selection_with_memmap(
    context: LagSelectionExecutionContext,
    *,
    optimize_non_lag_model_params_fn: OptimizeNonLagModelParamsFn,
    run_single_addon_lag_selection_memmap_fn: RunSingleAddonLagSelectionFn,
) -> dict[str, Path]:
    candidate_memmap_path, candidate_matrix = candidate_memmap_assets(context)
    try:
        candidate_to_index, filtered_lag_cols, after_corr_count, required_cols = filter_lag_candidates_with_memmap(
            candidate_matrix=candidate_matrix,
            lag_candidate_cols=context.prepared.lag_candidate_cols,
            selection_train=context.prepared.selection_train,
            target_col=context.target_col,
            all_columns=context.prepared.all_columns,
            output_paths=context.prepared.output_paths,
            pearson_threshold=context.pearson_threshold,
            spearman_threshold=context.spearman_threshold,
            logger=context.logger,
            max_selected_lag_features=context.max_selected_lag_features,
        )
        if context.correlation_only:
            return run_correlation_only_selection(
                context=correlation_only_context(context, required_cols, filtered_lag_cols, after_corr_count)
            )
        return run_full_lag_selection(
            context=full_lag_selection_context(
                context=context,
                required_cols=required_cols,
                filtered_lag_cols=filtered_lag_cols,
                candidate_matrix=candidate_matrix,
                candidate_to_index=candidate_to_index,
                after_corr_count=after_corr_count,
            ),
            optimize_non_lag_model_params_fn=optimize_non_lag_model_params_fn,
            run_single_addon_lag_selection_memmap_fn=run_single_addon_lag_selection_memmap_fn,
        )
    finally:
        candidate_matrix.flush()
        del candidate_matrix
        if candidate_memmap_path.exists():
            candidate_memmap_path.unlink()


def candidate_memmap_assets(context: LagSelectionExecutionContext) -> tuple[Path, np.memmap]:
    return build_candidate_memmap(
        context.prepared.source_path,
        context.prepared.lag_candidate_cols,
        context.prepared.selection_mask,
        context.prepared.work_dir,
        memmap_progress_log_every=25,
        logger=context.logger,
    )


def correlation_only_context(
    context: LagSelectionExecutionContext,
    required_cols: list[str],
    filtered_lag_cols: list[str],
    after_corr_count: int,
) -> CorrelationOnlyContext:
    return CorrelationOnlyContext(
        source_path=context.prepared.source_path,
        output_paths=context.prepared.output_paths,
        split_metadata=context.prepared.split_metadata,
        date_col=context.date_col,
        target_col=context.target_col,
        selection_mask=context.prepared.selection_mask,
        holdout_mask=context.prepared.holdout_mask,
        non_lag_feature_cols=context.non_lag_feature_cols,
        required_scoring_context_cols=required_cols,
        filtered_lag_cols=filtered_lag_cols,
        lag_candidate_cols=context.prepared.lag_candidate_cols,
        lag_candidate_count_after_corr_filter=after_corr_count,
        n_folds=context.n_folds,
        pearson_threshold=context.pearson_threshold,
        spearman_threshold=context.spearman_threshold,
        tuning_trials=context.tuning_trials,
        tuning_random_seed=context.tuning_random_seed,
        model_params=context.model_params,
    )


def full_lag_selection_context(
    *,
    context: LagSelectionExecutionContext,
    required_cols: list[str],
    filtered_lag_cols: list[str],
    candidate_matrix: np.memmap,
    candidate_to_index: dict[str, int],
    after_corr_count: int,
) -> FullLagSelectionContext:
    return FullLagSelectionContext(
        source_path=context.prepared.source_path,
        output_paths=context.prepared.output_paths,
        split_metadata=context.prepared.split_metadata,
        selection_train=context.prepared.selection_train,
        base_frame=context.prepared.base_frame,
        selection_mask=context.prepared.selection_mask,
        holdout_mask=context.prepared.holdout_mask,
        date_col=context.date_col,
        target_col=context.target_col,
        lag_candidate_cols=context.prepared.lag_candidate_cols,
        lag_candidate_count_after_corr_filter=after_corr_count,
        non_lag_feature_cols=context.non_lag_feature_cols,
        required_scoring_context_cols=required_cols,
        filtered_lag_cols=filtered_lag_cols,
        candidate_matrix=candidate_matrix,
        candidate_to_index=candidate_to_index,
        baseline_feature_cols=baseline_feature_columns(context.prepared.all_columns),
        n_folds=context.n_folds,
        pearson_threshold=context.pearson_threshold,
        spearman_threshold=context.spearman_threshold,
        tuning_trials=context.tuning_trials,
        tuning_random_seed=context.tuning_random_seed,
        model_params=context.model_params,
        num_workers=context.num_workers,
        logger=context.logger,
    )


def filter_lag_candidates_with_memmap(
    *,
    candidate_matrix: np.memmap,
    lag_candidate_cols: list[str],
    selection_train: pd.DataFrame,
    target_col: str,
    all_columns: list[str],
    output_paths: dict[str, Path],
    pearson_threshold: float,
    spearman_threshold: float,
    max_selected_lag_features: int | None,
    logger: logging.Logger,
) -> tuple[dict[str, int], list[str], int, list[str]]:
    candidate_to_index = {column: index for index, column in enumerate(lag_candidate_cols)}
    filtered_lag_cols, correlation_report = filter_correlated_lag_features_memmap(
        candidate_matrix,
        lag_candidate_cols,
        selection_train[target_col].to_numpy(dtype=np.float32, copy=False),
        pearson_threshold=pearson_threshold,
        spearman_threshold=spearman_threshold,
        correlation_progress_log_every=25,
        logger=logger,
    )
    after_corr_count = len(filtered_lag_cols)
    if max_selected_lag_features is not None:
        filtered_lag_cols = filtered_lag_cols[: max(0, int(max_selected_lag_features))]
    correlation_report.to_csv(output_paths["lag_correlation_report"], index=False)
    return candidate_to_index, filtered_lag_cols, after_corr_count, required_scoring_context_columns(all_columns)


def run_correlation_only_selection(*, context: CorrelationOnlyContext) -> dict[str, Path]:
    final_cols = build_final_output_columns(
        date_col=context.date_col,
        target_col=context.target_col,
        non_lag_feature_cols=context.non_lag_feature_cols,
        selected_lag_features=context.filtered_lag_cols,
        required_scoring_context_cols=context.required_scoring_context_cols,
    )
    train_selected, tuning_selected = materialize_selected_frames(
        source_path=context.source_path,
        final_cols=final_cols,
        date_col=context.date_col,
        selection_mask=context.selection_mask,
        holdout_mask=context.holdout_mask,
    )
    write_selected_frames(train_selected=train_selected, tuning_selected=tuning_selected, output_paths=context.output_paths)
    addon_report = pd.DataFrame(
        [{"feature": feature, "keep": True, "selection_stage": "correlation_only", "wape_improvement_pct": np.nan} for feature in context.filtered_lag_cols]
    )
    addon_report.to_csv(context.output_paths["lag_addon_importance_report"], index=False)
    pd.DataFrame([{"mode": "correlation_only", "tuning_trials_skipped": context.tuning_trials, "reason": "pearson_and_spearman_filter_only"}]).to_csv(
        context.output_paths["non_lag_model_tuning_report"], index=False
    )
    json_dump(context.output_paths["best_non_lag_model_params"], {"mode": "correlation_only", "best_non_lag_model_params": context.model_params or {}})
    write_addon_checkpoint(
        context.output_paths["lag_addon_importance_report"],
        context.output_paths["selected_lag_features"],
        [{str(key): value for key, value in row.items()} for row in addon_report.to_dict(orient="records")],
        context.filtered_lag_cols,
    )
    json_dump(context.output_paths["baseline_report"], correlation_only_baseline_payload(context))
    json_dump(context.output_paths["split_metadata"], correlation_only_split_payload(context))
    return return_output_paths(context.output_paths)


def correlation_only_baseline_payload(context: CorrelationOnlyContext) -> dict[str, object]:
    return {
        "mode": "correlation_only",
        "best_baseline": None,
        "all_baselines": [],
        "base_model_report": [],
        "best_non_lag_model_params": context.model_params or {},
    }


def correlation_only_split_payload(context: CorrelationOnlyContext) -> dict[str, object]:
    return {
        **context.split_metadata,
        "n_folds": context.n_folds,
        "pearson_threshold": context.pearson_threshold,
        "spearman_threshold": context.spearman_threshold,
        "tuning_trials": 0,
        "tuning_random_seed": context.tuning_random_seed,
        "lag_candidate_count": len(context.lag_candidate_cols),
        "lag_candidate_count_after_corr_filter": context.lag_candidate_count_after_corr_filter,
        "lag_candidate_count_after_cap": len(context.filtered_lag_cols),
        "selected_lag_feature_count": len(context.filtered_lag_cols),
        "non_lag_feature_count": len(context.non_lag_feature_cols),
        "required_scoring_context_cols": context.required_scoring_context_cols,
        "correlation_only": True,
    }


def run_full_lag_selection(
    *,
    context: FullLagSelectionContext,
    optimize_non_lag_model_params_fn: OptimizeNonLagModelParamsFn,
    run_single_addon_lag_selection_memmap_fn: RunSingleAddonLagSelectionFn,
) -> dict[str, Path]:
    raise_if_tft_backend_required("features_selection_lag.build_lag_selection_outputs")
    folds = build_walk_forward_folds(context.selection_train, date_col=context.date_col, n_folds=context.n_folds, logger=context.logger)
    baseline_frame = build_baseline_frame(context)
    baseline_report_rows, best_baseline = evaluate_statistical_baselines(baseline_frame, folds, logger=context.logger, target_col=context.target_col)
    tuned_model_params, tuning_report = optimize_non_lag_model_params_fn(
        frame=context.selection_train,
        folds=folds,
        feature_cols=context.non_lag_feature_cols,
        logger=context.logger,
        target_col=context.target_col,
        n_trials=context.tuning_trials,
        random_seed=context.tuning_random_seed,
        baseline_wape=float(cast(float, best_baseline["mean_wape"])),
        base_model_params=context.model_params,
        num_workers=context.num_workers,
    )
    tuning_report.to_csv(context.output_paths["non_lag_model_tuning_report"], index=False)
    json_dump(context.output_paths["best_non_lag_model_params"], tuned_model_params)
    write_addon_checkpoint(context.output_paths["lag_addon_importance_report"], context.output_paths["selected_lag_features"], [], [])
    selected_lag_features, addon_report, base_model_report = run_single_addon_lag_selection_memmap_fn(
        base_frame=context.selection_train,
        folds=folds,
        base_feature_cols=context.non_lag_feature_cols,
        candidate_cols=context.filtered_lag_cols,
        candidate_matrix=context.candidate_matrix,
        candidate_to_index=context.candidate_to_index,
        logger=context.logger,
        target_col=context.target_col,
        model_params=tuned_model_params,
        num_workers=context.num_workers,
        checkpoint_every=DEFAULT_PROGRESS_LOG_EVERY,
        checkpoint_callback=build_addon_checkpoint_callback(context),
    )
    persist_full_selection_outputs(
        context=context,
        tuned_model_params=tuned_model_params,
        baseline_report_rows=baseline_report_rows,
        best_baseline=best_baseline,
        base_model_report=base_model_report,
        selected_lag_features=selected_lag_features,
        addon_report=addon_report,
        folds=folds,
    )
    return return_output_paths(context.output_paths)


def build_baseline_frame(context: FullLagSelectionContext) -> pd.DataFrame:
    baseline_frame = context.selection_train.loc[:, [context.date_col, context.target_col]].copy()
    for baseline_col in context.baseline_feature_cols:
        baseline_frame[baseline_col] = baseline_column_values(context, baseline_col)
    return baseline_frame


def baseline_column_values(context: FullLagSelectionContext, baseline_col: str) -> np.ndarray | pd.Series:
    if baseline_col in context.candidate_to_index:
        return np.asarray(context.candidate_matrix[:, context.candidate_to_index[baseline_col]], dtype=np.float32)
    baseline_values = read_parquet_projected(context.source_path, columns=[baseline_col])
    return baseline_values.loc[context.selection_mask, baseline_col].reset_index(drop=True)


def persist_full_selection_outputs(
    *,
    context: FullLagSelectionContext,
    tuned_model_params: dict[str, object],
    baseline_report_rows: list[dict[str, object]],
    best_baseline: dict[str, object],
    base_model_report: Mapping[str, object] | list[dict[str, object]],
    selected_lag_features: list[str],
    addon_report: pd.DataFrame,
    folds: list[dict[str, object]],
) -> None:
    final_cols = build_final_output_columns(
        date_col=context.date_col,
        target_col=context.target_col,
        non_lag_feature_cols=context.non_lag_feature_cols,
        selected_lag_features=selected_lag_features,
        required_scoring_context_cols=context.required_scoring_context_cols,
    )
    train_selected, tuning_selected = materialize_selected_frames(
        source_path=context.source_path,
        final_cols=final_cols,
        date_col=context.date_col,
        selection_mask=context.selection_mask,
        holdout_mask=context.holdout_mask,
    )
    write_selected_frames(train_selected=train_selected, tuning_selected=tuning_selected, output_paths=context.output_paths)
    write_addon_checkpoint(
        context.output_paths["lag_addon_importance_report"],
        context.output_paths["selected_lag_features"],
        [{str(key): value for key, value in row.items()} for row in addon_report.to_dict(orient="records")],
        selected_lag_features,
    )
    json_dump(
        context.output_paths["baseline_report"],
        {
            "best_baseline": best_baseline,
            "all_baselines": baseline_report_rows,
            "base_model_report": base_model_report,
            "best_non_lag_model_params": tuned_model_params,
        },
    )
    json_dump(context.output_paths["split_metadata"], full_selection_split_payload(context, selected_lag_features, folds))


def full_selection_split_payload(
    context: FullLagSelectionContext,
    selected_lag_features: list[str],
    folds: list[dict[str, object]],
) -> dict[str, object]:
    return {
        **context.split_metadata,
        "n_folds": context.n_folds,
        "pearson_threshold": context.pearson_threshold,
        "spearman_threshold": context.spearman_threshold,
        "tuning_trials": context.tuning_trials,
        "tuning_random_seed": context.tuning_random_seed,
        "lag_candidate_count": len(context.lag_candidate_cols),
        "lag_candidate_count_after_corr_filter": context.lag_candidate_count_after_corr_filter,
        "lag_candidate_count_after_cap": len(context.filtered_lag_cols),
        "selected_lag_feature_count": len(selected_lag_features),
        "non_lag_feature_count": len(context.non_lag_feature_cols),
        "required_scoring_context_cols": context.required_scoring_context_cols,
        "correlation_only": False,
        "folds": [{key: value for key, value in fold.items() if key not in {"train_idx", "valid_idx"}} for fold in folds],
    }


def baseline_feature_columns(all_columns: list[str]) -> list[str]:
    candidates = [
        "target_lag_7",
        "target_lag_14",
        "target_lag_21",
        "target_lag_28",
        "target_same_dow_mean_4w",
        "sale_amount_lag_1",
        "sale_amount_lag_7",
        "sale_amount_lag_14",
        "sale_amount_lag_21",
        "sale_amount_lag_28",
        "sale_amount_rolling_mean_7",
        "sale_amount_rolling_mean_28",
    ]
    return [column for column in candidates if column in all_columns]


def build_addon_checkpoint_callback(context: FullLagSelectionContext):
    def checkpoint(report_rows: list[dict[str, object]], selected_features: list[str]) -> None:
        write_addon_checkpoint(
            context.output_paths["lag_addon_importance_report"],
            context.output_paths["selected_lag_features"],
            report_rows,
            selected_features,
        )

    return checkpoint


def return_output_paths(output_paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "train_selected": output_paths["train_selected"],
        "tuning_selected": output_paths["tuning_selected"],
        "selected_lag_features": output_paths["selected_lag_features"],
        "lag_correlation_report": output_paths["lag_correlation_report"],
        "non_lag_model_tuning_report": output_paths["non_lag_model_tuning_report"],
        "best_non_lag_model_params": output_paths["best_non_lag_model_params"],
        "lag_addon_importance_report": output_paths["lag_addon_importance_report"],
        "baseline_report": output_paths["baseline_report"],
        "split_metadata": output_paths["split_metadata"],
    }
