from __future__ import annotations

import json
import logging
from pathlib import Path

from distributed import as_completed
import numpy as np
import pandas as pd

from research_praedixa.distributed.budgets import build_runtime_budget
from research_praedixa.distributed.cluster import connect_client, wait_for_workers
from research_praedixa.distributed.config import DistributedConfig
from research_praedixa.distributed.logging import execution_identity
from research_praedixa.distributed.sync import sync_project_tree
from research_praedixa.features_selection_lag.pipeline import (
    DEFAULT_CORRELATION_PROGRESS_LOG_EVERY,
    DEFAULT_DATE_COL,
    DEFAULT_MEMMAP_PROGRESS_LOG_EVERY,
    DEFAULT_N_FOLDS,
    DEFAULT_PEARSON_THRESHOLD,
    DEFAULT_PROGRESS_LOG_EVERY,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPEARMAN_THRESHOLD,
    DEFAULT_TARGET_COL,
    DEFAULT_TUNING_TRIALS,
    _build_candidate_memmap,
    _json_dump,
    _score_candidate_task_memmap,
    _write_addon_checkpoint,
    build_walk_forward_folds,
    evaluate_statistical_baselines,
    filter_correlated_lag_features_memmap,
    get_lag_candidate_columns_from_names,
    optimize_non_lag_model_params,
)
from research_praedixa.memory_utils import get_parquet_columns, read_parquet_projected


logger = logging.getLogger(__name__)


def _score_candidate_batch_task(
    *,
    base_frame_path: str,
    memmap_path: str,
    candidate_cols: list[str],
    candidate_to_index: dict[str, int],
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
    base_mean_wape: float,
    base_scores: list[float],
) -> dict[str, object]:
    base_frame = read_parquet_projected(base_frame_path)
    memmap = np.memmap(memmap_path, mode="r", dtype=np.float32, shape=(len(base_frame), len(candidate_to_index)))
    rows = [
        _score_candidate_task_memmap(
            base_frame=base_frame,
            folds=folds,
            base_feature_cols=base_feature_cols,
            candidate=candidate,
            candidate_values=np.asarray(memmap[:, candidate_to_index[candidate]], dtype=np.float32),
            target_col=target_col,
            model_params=model_params,
            threads_per_worker=int(model_params.get("n_jobs", 1)),
            base_mean_wape=base_mean_wape,
            base_scores=base_scores,
        )
        for candidate in candidate_cols
    ]
    return {
        "rows": rows,
        "execution_identity": execution_identity(),
    }


def run_distributed_lag_selection(
    *,
    config: DistributedConfig,
    input_path: str | Path,
    output_dir: str | Path = "data/features_selection_lag_distributed",
    date_col: str = DEFAULT_DATE_COL,
    target_col: str = DEFAULT_TARGET_COL,
    train_fraction: float = 0.7,
    n_folds: int = DEFAULT_N_FOLDS,
    pearson_threshold: float = DEFAULT_PEARSON_THRESHOLD,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    tuning_random_seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
) -> dict[str, Path]:
    source_path = Path(input_path).resolve()
    target_dir = (config.repo_path / output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    work_dir = target_dir / "_cache"
    work_dir.mkdir(parents=True, exist_ok=True)

    all_columns = get_parquet_columns(source_path)
    lag_candidate_cols = get_lag_candidate_columns_from_names(all_columns)
    non_lag_feature_cols = [
        column
        for column in all_columns
        if column not in {target_col, date_col} and column not in lag_candidate_cols
    ]
    date_frame = read_parquet_projected(source_path, columns=[date_col])
    date_frame[date_col] = pd.to_datetime(date_frame[date_col])
    total_rows = int(len(date_frame))
    unique_dates = pd.Index(date_frame[date_col].drop_duplicates().sort_values())
    split_idx = max(1, int(len(unique_dates) * train_fraction))
    split_idx = min(split_idx, len(unique_dates) - 1)
    train_dates = unique_dates[:split_idx]
    holdout_dates = unique_dates[split_idx:]
    selection_mask = date_frame[date_col].isin(train_dates).to_numpy()
    holdout_mask = date_frame[date_col].isin(holdout_dates).to_numpy()
    split_metadata = {
        "train_fraction": train_fraction,
        "total_rows": total_rows,
        "train_rows": int(selection_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "train_unique_dates": int(len(train_dates)),
        "holdout_unique_dates": int(len(holdout_dates)),
    }
    base_cols = [date_col, target_col, *non_lag_feature_cols]
    base_frame = read_parquet_projected(source_path, columns=base_cols)
    base_frame[date_col] = pd.to_datetime(base_frame[date_col])
    selection_train = base_frame.loc[selection_mask, base_cols].copy().reset_index(drop=True)
    tuning_holdout = base_frame.loc[holdout_mask, base_cols].copy().reset_index(drop=True)
    base_frame_path = work_dir / "selection_train_base.parquet"
    selection_train.to_parquet(base_frame_path, index=False)

    candidate_memmap_path, candidate_matrix = _build_candidate_memmap(
        source_path=source_path,
        candidate_cols=lag_candidate_cols,
        selection_mask=selection_mask,
        work_dir=work_dir,
    )
    candidate_to_index = {column: index for index, column in enumerate(lag_candidate_cols)}
    filtered_lag_cols, correlation_report = filter_correlated_lag_features_memmap(
        candidate_matrix=candidate_matrix,
        candidate_cols=lag_candidate_cols,
        target_values=selection_train[target_col].to_numpy(dtype=np.float32, copy=False),
        pearson_threshold=pearson_threshold,
        spearman_threshold=spearman_threshold,
    )
    folds = build_walk_forward_folds(selection_train, date_col=date_col, n_folds=n_folds)

    baseline_feature_cols = [
        column
        for column in [
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
        if column in all_columns
    ]
    baseline_frame = selection_train.loc[:, [date_col, target_col]].copy()
    for baseline_col in baseline_feature_cols:
        if baseline_col in candidate_to_index:
            baseline_frame[baseline_col] = np.asarray(candidate_matrix[:, candidate_to_index[baseline_col]], dtype=np.float32)
        else:
            baseline_values = read_parquet_projected(source_path, columns=[baseline_col])
            baseline_frame[baseline_col] = baseline_values.loc[selection_mask, baseline_col].reset_index(drop=True)
    baseline_report_rows, best_baseline = evaluate_statistical_baselines(
        baseline_frame,
        folds,
        target_col=target_col,
    )
    tuned_model_params, tuning_report = optimize_non_lag_model_params(
        frame=selection_train,
        folds=folds,
        feature_cols=non_lag_feature_cols,
        target_col=target_col,
        n_trials=tuning_trials,
        random_seed=tuning_random_seed,
        baseline_wape=float(best_baseline["mean_wape"]),
        base_model_params=model_params,
    )
    tuned_model_params["n_jobs"] = build_runtime_budget(config).xgboost_threads_per_task
    base_scores = [
        float(score)
        for score in tuning_report.iloc[0]["fold_wape_scores"]
    ] if not tuning_report.empty else []
    base_mean_wape = float(tuning_report.iloc[0]["mean_wape"]) if not tuning_report.empty else float(best_baseline["mean_wape"])

    correlation_report_path = target_dir / "lag_correlation_filter_report.csv"
    tuning_report_path = target_dir / "non_lag_model_tuning_report.csv"
    best_params_path = target_dir / "best_non_lag_model_params.json"
    addon_report_path = target_dir / "lag_addon_importance_report.csv"
    selected_lag_features_path = target_dir / "selected_lag_features.json"
    baseline_report_path = target_dir / "baseline_report.json"
    split_metadata_path = target_dir / "split_metadata.json"
    train_output_path = target_dir / "train_selection_70_selected.parquet"
    tuning_output_path = target_dir / "train_tuning_30_selected.parquet"

    correlation_report.to_csv(correlation_report_path, index=False)
    tuning_report.to_csv(tuning_report_path, index=False)
    _json_dump(best_params_path, tuned_model_params)
    _write_addon_checkpoint(addon_report_path, selected_lag_features_path, [], [])
    sync_project_tree(config)

    batch_size = max(1, int(config.pipelines.feature_selection.candidate_batch_size))
    batches = [filtered_lag_cols[index:index + batch_size] for index in range(0, len(filtered_lag_cols), batch_size)]
    client = connect_client(config)
    wait_for_workers(client, minimum_workers=build_runtime_budget(config).cluster_task_slots)
    futures = [
        client.submit(
            _score_candidate_batch_task,
            base_frame_path=str(base_frame_path),
            memmap_path=str(candidate_memmap_path),
            candidate_cols=batch,
            candidate_to_index=candidate_to_index,
            folds=folds,
            base_feature_cols=non_lag_feature_cols,
            target_col=target_col,
            model_params=tuned_model_params,
            base_mean_wape=base_mean_wape,
            base_scores=base_scores,
            pure=False,
        )
        for batch in batches
    ]
    results_by_feature: dict[str, dict[str, object]] = {}
    worker_execution: list[dict[str, object]] = []
    completed = 0
    for future in as_completed(futures):
        result = future.result()
        worker_execution.append(result["execution_identity"])
        for row in result["rows"]:
            results_by_feature[str(row["feature"])] = row
        completed += len(result["rows"])
        ordered_rows = [results_by_feature[feature] for feature in filtered_lag_cols if feature in results_by_feature]
        selected_features = [feature for feature in filtered_lag_cols if feature in results_by_feature and results_by_feature[feature]["keep"]]
        _write_addon_checkpoint(addon_report_path, selected_lag_features_path, ordered_rows, selected_features)
        logger.info(
            "Distributed add-on progress: tested=%s/%s kept=%s",
            completed,
            len(filtered_lag_cols),
            len(selected_features),
        )
    addon_report = pd.DataFrame([results_by_feature[feature] for feature in filtered_lag_cols]).sort_values(
        by="wape_improvement_pct",
        ascending=False,
    ).reset_index(drop=True)
    selected_lag_features = [feature for feature in filtered_lag_cols if results_by_feature[feature]["keep"]]

    required_scoring_context_cols = [
        column
        for column in ("target_lag_7", "sale_amount_lag_7", "lag_7", "target_same_dow_mean_4w", "same_dow_mean_4w")
        if column in all_columns
    ]
    final_cols = [
        date_col,
        target_col,
        *non_lag_feature_cols,
        *selected_lag_features,
        *[
            column
            for column in required_scoring_context_cols
            if column not in {*non_lag_feature_cols, *selected_lag_features}
        ],
    ]
    final_frame = read_parquet_projected(source_path, columns=final_cols)
    final_frame[date_col] = pd.to_datetime(final_frame[date_col])
    train_selected = final_frame.loc[selection_mask, final_cols].copy().sort_values(date_col).reset_index(drop=True)
    tuning_selected = final_frame.loc[holdout_mask, final_cols].copy().sort_values(date_col).reset_index(drop=True)
    train_selected.to_parquet(train_output_path, index=False)
    tuning_selected.to_parquet(tuning_output_path, index=False)
    _write_addon_checkpoint(
        addon_report_path=addon_report_path,
        selected_lag_features_path=selected_lag_features_path,
        report_rows=addon_report.to_dict(orient="records"),
        selected_features=selected_lag_features,
    )
    _json_dump(
        baseline_report_path,
        {
            "best_baseline": best_baseline,
            "all_baselines": baseline_report_rows,
            "worker_execution": worker_execution,
        },
    )
    _json_dump(
        split_metadata_path,
        {
            **split_metadata,
            "n_folds": n_folds,
            "pearson_threshold": pearson_threshold,
            "spearman_threshold": spearman_threshold,
            "tuning_trials": tuning_trials,
            "tuning_random_seed": tuning_random_seed,
            "lag_candidate_count": len(lag_candidate_cols),
            "lag_candidate_count_after_corr_filter": len(filtered_lag_cols),
            "selected_lag_feature_count": len(selected_lag_features),
            "non_lag_feature_count": len(non_lag_feature_cols),
            "required_scoring_context_cols": required_scoring_context_cols,
        },
    )
    candidate_matrix.flush()
    return {
        "train_selected": train_output_path,
        "tuning_selected": tuning_output_path,
        "selected_lag_features": selected_lag_features_path,
        "lag_correlation_report": correlation_report_path,
        "non_lag_model_tuning_report": tuning_report_path,
        "best_non_lag_model_params": best_params_path,
        "lag_addon_importance_report": addon_report_path,
        "baseline_report": baseline_report_path,
        "split_metadata": split_metadata_path,
    }
