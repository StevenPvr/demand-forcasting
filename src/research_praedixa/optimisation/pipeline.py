from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path

import duckdb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from research_praedixa.memory_utils import downcast_pandas_frame, read_parquet_projected
from research_praedixa.features_selection_lag.pipeline import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_N_FOLDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TUNING_TRIALS,
    DEFAULT_TUNING_PROGRESS_LOG_EVERY,
    compute_wape,
    compute_wape_improvement_pct,
)
from research_praedixa.target_utils import (
    DEFAULT_VARIATION_TARGET_COL,
    TargetContract,
    build_target_contract_metadata,
    ensure_learning_target_column,
    reconstruct_absolute_predictions,
    resolve_target_contract,
)
from research_praedixa.xgboost_utils import (
    DEFAULT_XGBOOST_MODEL_PARAMS,
    prepare_xgboost_feature_frame,
    predict_with_xgboost_matrix,
    select_numeric_feature_columns,
    train_xgboost_with_matrices,
)


DEFAULT_DUCKDB_PATH = Path("data/warehouse/praedixa.duckdb")
DEFAULT_GOLD_TABLE = "gold.gold_feature_panel_d1"
DEFAULT_TRAIN_INPUT_PATH: Path | None = None
DEFAULT_TUNING_INPUT_PATH: Path | None = None
DEFAULT_OUTPUT_DIR = Path("data/optimisation")
DEFAULT_TARGET_COL = DEFAULT_VARIATION_TARGET_COL
DEFAULT_DATE_COL = "dt"
DEFAULT_DATASET_SOURCE_COL = "dataset_source"
DEFAULT_SAMPLE_STORE_COL = "location_id"
DEFAULT_TRAIN_SAMPLE_FRACTION = 0.20
DEFAULT_TUNING_SAMPLE_FRACTION = 0.20
DEFAULT_MAX_PARALLEL_FOLD_WORKERS = 5
DEFAULT_TARGET_TRANSFORM = "log1p"
DEFAULT_IDENTIFIER_FEATURE_COLS = (
    "series_id",
    "location_id",
    "product_id",
    "client_id",
    "gold_run_id",
)
DEFAULT_EXCLUDED_RISKY_FEATURE_COLS = (
    "current_day_demand_qty",
    "observed_revenue_net",
)

logger = logging.getLogger(__name__)


def build_tuning_walk_forward_folds(
    tuning_frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    ordered = tuning_frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values(date_col).reset_index(drop=True)
    unique_dates = pd.Index(ordered[date_col].drop_duplicates().sort_values())

    if len(unique_dates) < (n_folds + 1):
        raise ValueError(f"Need at least {n_folds + 1} unique dates to build {n_folds} walk-forward folds.")

    valid_dates_per_fold = max(1, len(unique_dates) // (n_folds + 1))
    folds: list[dict[str, object]] = []
    for fold_idx in range(n_folds):
        train_end = valid_dates_per_fold * (fold_idx + 1)
        valid_end = min(train_end + valid_dates_per_fold, len(unique_dates))
        train_dates = unique_dates[:train_end]
        valid_dates = unique_dates[train_end:valid_end]
        if len(valid_dates) == 0:
            break

        train_mask = ordered[date_col].isin(train_dates)
        valid_mask = ordered[date_col].isin(valid_dates)
        folds.append(
            {
                "fold": fold_idx + 1,
                "train_idx": ordered.index[train_mask].to_numpy(),
                "valid_idx": ordered.index[valid_mask].to_numpy(),
                "train_dates": int(len(train_dates)),
                "valid_dates": int(len(valid_dates)),
                "train_end_date": train_dates.max().strftime("%Y-%m-%d"),
                "valid_start_date": valid_dates.min().strftime("%Y-%m-%d"),
                "valid_end_date": valid_dates.max().strftime("%Y-%m-%d"),
            }
        )

    if len(folds) != n_folds:
        raise ValueError(f"Expected {n_folds} folds, got {len(folds)}.")
    logger.info(
        "Optimisation walk-forward folds built: n_folds=%s valid_dates_per_fold=%s",
        len(folds),
        valid_dates_per_fold,
    )
    return folds


def build_tuning_walk_forward_folds_by_dataset(
    tuning_frame: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    dataset_source_col: str = DEFAULT_DATASET_SOURCE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    dataset_folds: list[dict[str, object]] = []
    for dataset_source, dataset_frame in tuning_frame.groupby(dataset_source_col, sort=False):
        ordered = dataset_frame.copy()
        ordered[date_col] = pd.to_datetime(ordered[date_col])
        ordered = ordered.sort_values(date_col).reset_index(drop=True)
        unique_dates = pd.Index(ordered[date_col].drop_duplicates().sort_values())
        if len(unique_dates) < (n_folds + 1):
            raise ValueError(
                f"Dataset `{dataset_source}` needs at least {n_folds + 1} unique dates "
                f"to build {n_folds} walk-forward folds, got {len(unique_dates)}."
            )

        valid_dates_per_fold = max(1, len(unique_dates) // (n_folds + 1))
        for fold_idx in range(n_folds):
            train_end = valid_dates_per_fold * (fold_idx + 1)
            valid_end = min(train_end + valid_dates_per_fold, len(unique_dates))
            train_dates = unique_dates[:train_end]
            valid_dates = unique_dates[train_end:valid_end]
            if len(valid_dates) == 0:
                break

            train_mask = ordered[date_col].isin(train_dates)
            valid_mask = ordered[date_col].isin(valid_dates)
            dataset_folds.append(
                {
                    "dataset_source": str(dataset_source),
                    "fold": fold_idx + 1,
                    "train_idx": ordered.index[train_mask].to_numpy(),
                    "valid_idx": ordered.index[valid_mask].to_numpy(),
                    "train_dates": int(len(train_dates)),
                    "valid_dates": int(len(valid_dates)),
                    "train_end_date": train_dates.max().strftime("%Y-%m-%d"),
                    "valid_start_date": valid_dates.min().strftime("%Y-%m-%d"),
                    "valid_end_date": valid_dates.max().strftime("%Y-%m-%d"),
                }
            )
    logger.info(
        "Optimisation walk-forward folds built by dataset: datasets=%s total_fold_evaluations=%s",
        sorted(tuning_frame[dataset_source_col].dropna().unique().tolist()),
        len(dataset_folds),
    )
    return dataset_folds


def build_grouped_tuning_walk_forward_folds_by_dataset(
    tuning_frame: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    dataset_source_col: str = DEFAULT_DATASET_SOURCE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    grouped_folds: list[dict[str, object]] = []
    per_dataset_metadata: dict[str, list[dict[str, object]]] = {}

    for dataset_source, dataset_frame in tuning_frame.groupby(dataset_source_col, sort=False):
        ordered = dataset_frame.copy()
        ordered[date_col] = pd.to_datetime(ordered[date_col])
        ordered = ordered.sort_values(date_col)
        dataset_positions = ordered.index.to_numpy(dtype=np.int32)
        unique_dates = pd.Index(ordered[date_col].drop_duplicates().sort_values())
        if len(unique_dates) < (n_folds + 1):
            raise ValueError(
                f"Dataset `{dataset_source}` needs at least {n_folds + 1} unique dates "
                f"to build {n_folds} walk-forward folds, got {len(unique_dates)}."
            )

        valid_dates_per_fold = max(1, len(unique_dates) // (n_folds + 1))
        dataset_fold_metadata: list[dict[str, object]] = []
        for fold_idx in range(n_folds):
            train_end = valid_dates_per_fold * (fold_idx + 1)
            valid_end = min(train_end + valid_dates_per_fold, len(unique_dates))
            train_dates = unique_dates[:train_end]
            valid_dates = unique_dates[train_end:valid_end]
            if len(valid_dates) == 0:
                break

            train_mask = ordered[date_col].isin(train_dates).to_numpy()
            valid_mask = ordered[date_col].isin(valid_dates).to_numpy()
            dataset_fold_metadata.append(
                {
                    "dataset_source": str(dataset_source),
                    "fold": fold_idx + 1,
                    "train_idx": dataset_positions[train_mask],
                    "valid_idx": dataset_positions[valid_mask],
                    "train_dates": int(len(train_dates)),
                    "valid_dates": int(len(valid_dates)),
                    "train_end_date": train_dates.max().strftime("%Y-%m-%d"),
                    "valid_start_date": valid_dates.min().strftime("%Y-%m-%d"),
                    "valid_end_date": valid_dates.max().strftime("%Y-%m-%d"),
                }
            )
        per_dataset_metadata[str(dataset_source)] = dataset_fold_metadata

    for fold_idx in range(n_folds):
        per_dataset_fold = [metadata[fold_idx] for metadata in per_dataset_metadata.values()]
        grouped_folds.append(
            {
                "fold": fold_idx + 1,
                "train_idx": np.concatenate(
                    [np.asarray(item["train_idx"], dtype=np.int32) for item in per_dataset_fold]
                ),
                "valid_idx": np.concatenate(
                    [np.asarray(item["valid_idx"], dtype=np.int32) for item in per_dataset_fold]
                ),
                "datasets": [str(item["dataset_source"]) for item in per_dataset_fold],
                "per_dataset": [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"train_idx", "valid_idx"}
                    }
                    for item in per_dataset_fold
                ],
            }
        )
    logger.info(
        "Grouped optimisation folds built by dataset: n_folds=%s datasets=%s",
        len(grouped_folds),
        sorted(per_dataset_metadata.keys()),
    )
    return grouped_folds


def _sample_optuna_params(trial: optuna.trial.Trial, random_seed: int = DEFAULT_RANDOM_SEED) -> dict[str, object]:
    return {
        "objective": "reg:squarederror",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": int(random_seed + trial.number + 1),
        "verbosity": 0,
        "tree_method": "hist",
    }


def _fit_and_score_xgboost_on_tuning(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object] | None = None,
    total_threads: int | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> dict[str, object]:
    resolved_params = {**DEFAULT_XGBOOST_MODEL_PARAMS, **(model_params or {})}
    max_parallel_fold_workers = int(
        resolved_params.pop("max_parallel_fold_workers", DEFAULT_MAX_PARALLEL_FOLD_WORKERS)
    )
    num_threads = int(total_threads or resolved_params.get("n_jobs", os.cpu_count() or 1))
    fold_workers = max(1, min(len(folds), max_parallel_fold_workers))
    threads_per_fold = max(1, num_threads // fold_workers)
    logger.info(
        "Optuna fold execution plan: fold_workers=%s threads_per_fold=%s total_threads=%s",
        fold_workers,
        threads_per_fold,
        num_threads,
    )
    learning_target_col = target_contract.learning_target_col
    absolute_target_col = target_contract.absolute_target_col

    shared_matrix_frame = pd.concat(
        [
            train_frame.loc[:, [*feature_cols, learning_target_col]],
            tuning_frame.loc[:, [*feature_cols, learning_target_col]],
        ],
        axis=0,
        ignore_index=True,
    )
    if target_transform == "log1p":
        if (shared_matrix_frame[learning_target_col] < 0).any():
            raise ValueError("Negative targets are incompatible with log1p target transformation.")
        shared_matrix_frame[learning_target_col] = np.log1p(
            shared_matrix_frame[learning_target_col].astype(float)
        )
    shared_feature_matrix = prepare_xgboost_feature_frame(shared_matrix_frame, feature_cols).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    shared_learning_target = shared_matrix_frame[learning_target_col].to_numpy(dtype=np.float32, copy=False)
    train_row_count = len(train_frame)
    base_train_indices = np.arange(train_row_count, dtype=np.int32)
    train_dataset_sources = train_frame[DEFAULT_DATASET_SOURCE_COL].reset_index(drop=True)
    tuning_dataset_sources = tuning_frame[DEFAULT_DATASET_SOURCE_COL].reset_index(drop=True)
    shared_dataset_sources = pd.concat(
        [train_dataset_sources, tuning_dataset_sources],
        axis=0,
        ignore_index=True,
    ).to_numpy(copy=False)
    shared_absolute_target = pd.concat(
        [
            train_frame[absolute_target_col].reset_index(drop=True),
            tuning_frame[absolute_target_col].reset_index(drop=True),
        ],
        axis=0,
        ignore_index=True,
    ).to_numpy(dtype=np.float32, copy=False)
    shared_reconstruction_anchor: np.ndarray | None = None
    if target_contract.reconstruction_anchor_col is not None:
        shared_reconstruction_anchor = pd.concat(
            [
                train_frame[target_contract.reconstruction_anchor_col].reset_index(drop=True),
                tuning_frame[target_contract.reconstruction_anchor_col].reset_index(drop=True),
            ],
            axis=0,
            ignore_index=True,
        ).to_numpy(dtype=np.float32, copy=False)

    def _reconstruct_absolute_predictions_from_arrays(
        raw_predictions: np.ndarray,
        *,
        valid_indices: np.ndarray,
    ) -> np.ndarray:
        raw_predictions = np.asarray(raw_predictions, dtype=np.float32)
        if target_contract.target_mode == "delta_log_wow":
            if shared_reconstruction_anchor is None:
                raise ValueError("WoW delta reconstruction requires an anchor array in the scoring frame.")
            anchor = shared_reconstruction_anchor[valid_indices]
            absolute_predictions = np.expm1(raw_predictions + np.log1p(np.clip(anchor, 0.0, None)))
        elif target_contract.target_mode == "log1p":
            absolute_predictions = np.expm1(raw_predictions)
        else:
            absolute_predictions = raw_predictions
        return np.clip(absolute_predictions, 0.0, None)

    def _fit_single_fold(fold: dict[str, object]) -> dict[str, object]:
        fold_train_idx = np.asarray(fold["train_idx"], dtype=np.int32)
        fold_valid_idx = np.asarray(fold["valid_idx"], dtype=np.int32)
        if fold_train_idx.size == 0 or fold_valid_idx.size == 0:
            raise ValueError(
                f"Target contract `{target_contract.target_mode}` produced an empty grouped fold "
                f"{fold['fold']}."
            )
        train_indices = np.concatenate([base_train_indices, train_row_count + fold_train_idx]).astype(np.int32)
        valid_indices = train_row_count + fold_valid_idx
        fold_train_weights = _compute_equal_dataset_row_weights_from_values(
            shared_dataset_sources[train_indices]
        )
        fold_train_matrix = xgb.DMatrix(
            shared_feature_matrix[train_indices],
            label=shared_learning_target[train_indices],
            feature_names=feature_cols,
            weight=fold_train_weights,
        )
        fold_valid_matrix = xgb.DMatrix(
            shared_feature_matrix[valid_indices],
            label=shared_learning_target[valid_indices],
            feature_names=feature_cols,
        )
        model = train_xgboost_with_matrices(
            train_matrix=fold_train_matrix,
            model_params=resolved_params,
            default_params=DEFAULT_XGBOOST_MODEL_PARAMS,
            default_n_estimators=2000,
            valid_matrix=fold_valid_matrix,
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
            num_threads_override=threads_per_fold,
        )
        predictions = predict_with_xgboost_matrix(model, fold_valid_matrix)
        predictions = _reconstruct_absolute_predictions_from_arrays(
            predictions,
            valid_indices=valid_indices,
        )
        fold_results: list[dict[str, object]] = []
        scored = pd.DataFrame(
            {
                DEFAULT_DATASET_SOURCE_COL: shared_dataset_sources[valid_indices],
                absolute_target_col: shared_absolute_target[valid_indices],
                "prediction": predictions,
            }
        )
        for dataset_source, dataset_frame in scored.groupby(DEFAULT_DATASET_SOURCE_COL, sort=False):
            fold_results.append(
                {
                    "dataset_source": str(dataset_source),
                    "fold": int(fold["fold"]),
                    "wape": float(compute_wape(dataset_frame[absolute_target_col], dataset_frame["prediction"])),
                    "feature_count": int(len(feature_cols)),
                    "rows_scored": int(len(dataset_frame)),
                }
            )
        return {"fold": int(fold["fold"]), "dataset_results": fold_results}

    if fold_workers == 1:
        grouped_fold_results = [_fit_single_fold(fold) for fold in folds]
    else:
        with ThreadPoolExecutor(max_workers=fold_workers) as executor:
            future_to_index = {
                executor.submit(_fit_single_fold, fold): index
                for index, fold in enumerate(folds)
            }
            results_by_index: dict[int, dict[str, object]] = {}
            for future in as_completed(future_to_index):
                results_by_index[future_to_index[future]] = future.result()
        grouped_fold_results = [results_by_index[index] for index in range(len(folds))]

    fold_results: list[dict[str, object]] = []
    for grouped_fold_result in grouped_fold_results:
        fold_results.extend(grouped_fold_result["dataset_results"])

    dataset_mean_wape: dict[str, float] = {}
    for dataset_source in sorted({result["dataset_source"] for result in fold_results}):
        dataset_scores = [float(result["wape"]) for result in fold_results if result["dataset_source"] == dataset_source]
        dataset_mean_wape[dataset_source] = float(np.mean(dataset_scores))
    macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
    return {
        "macro_mean_wape": macro_mean_wape,
        "dataset_mean_wape": dataset_mean_wape,
        "fold_results": fold_results,
    }


def optimize_xgboost_params(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    baseline_wape: float,
    target_contract: TargetContract,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> tuple[dict[str, object], pd.DataFrame]:
    total_threads = int((model_params or {}).get("n_jobs", DEFAULT_XGBOOST_MODEL_PARAMS["n_jobs"]))
    logger.info(
        "Starting final optimisation with Optuna: trials=%s feature_count=%s baseline_wape=%.6f",
        tuning_trials,
        len(feature_cols),
        baseline_wape,
    )

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_counter = {"completed": 0}

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = {**(model_params or {}), **_sample_optuna_params(trial=trial, random_seed=random_seed)}
        tuning_result = _fit_and_score_xgboost_on_tuning(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            folds=folds,
            feature_cols=feature_cols,
            target_contract=target_contract,
            model_params=trial_params,
            total_threads=total_threads,
            target_transform=target_transform,
        )
        mean_wape = float(tuning_result["macro_mean_wape"])
        improvement_pct = compute_wape_improvement_pct(baseline_wape, mean_wape)
        trial.set_user_attr("mean_wape", mean_wape)
        trial.set_user_attr("dataset_mean_wape", tuning_result["dataset_mean_wape"])
        trial.set_user_attr("fold_wape_scores", tuning_result["fold_results"])
        trial_counter["completed"] += 1
        completed = trial_counter["completed"]
        if completed == 1 or completed % DEFAULT_TUNING_PROGRESS_LOG_EVERY == 0 or completed == tuning_trials:
            logger.info(
                "Final optimisation progress: trial=%s/%s current_best_improvement_pct=%.6f",
                completed,
                tuning_trials,
                max((t.value for t in study.trials if t.value is not None), default=improvement_pct),
            )
        return improvement_pct

    study.optimize(objective, n_trials=tuning_trials, n_jobs=1)

    tuning_rows = []
    for trial in study.trials:
        tuning_rows.append(
            {
                "trial": trial.number,
                "mean_wape": float(trial.user_attrs["mean_wape"]),
                "baseline_wape_improvement_pct": float(trial.value),
                "fold_wape_scores": trial.user_attrs["fold_wape_scores"],
                "dataset_mean_wape": trial.user_attrs["dataset_mean_wape"],
                **trial.params,
            }
        )
    tuning_report = pd.DataFrame(tuning_rows).sort_values(
        by="baseline_wape_improvement_pct",
        ascending=False,
    ).reset_index(drop=True)
    best_params = {**(model_params or {}), **study.best_params}
    best_params["random_state"] = random_seed + study.best_trial.number + 1
    best_params["verbosity"] = 0
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"
    best_params["tree_method"] = "hist"
    best_params["n_jobs"] = total_threads
    best_params.pop("max_parallel_fold_workers", None)
    logger.info(
        "Final optimisation complete: best_trial=%s best_wape=%.6f best_improvement_pct=%.6f",
        study.best_trial.number,
        study.best_trial.user_attrs["mean_wape"],
        study.best_trial.value,
    )
    return best_params, tuning_report


def _json_dump(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _resolve_sampling_store_col(frame: pd.DataFrame) -> str:
    candidate_columns = (
        "location_id",
        "store_id",
        "series_id",
        "product_id",
    )
    for candidate in candidate_columns:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Unable to resolve a sampling store column. "
        f"Tried: {', '.join(candidate_columns)}."
    )


def _compute_equal_dataset_row_weights(
    frame: pd.DataFrame,
    *,
    dataset_source_col: str,
) -> np.ndarray:
    return _compute_equal_dataset_row_weights_from_values(frame[dataset_source_col])


def _compute_equal_dataset_row_weights_from_values(dataset_sources: pd.Series | np.ndarray) -> np.ndarray:
    source_series = pd.Series(dataset_sources, copy=False)
    dataset_counts = source_series.value_counts(dropna=False)
    dataset_count = max(1, len(dataset_counts))
    relative_weights = source_series.map(
        lambda dataset: 1.0 / (dataset_count * float(dataset_counts.loc[dataset]))
    ).to_numpy(dtype=float)
    # Preserve equal dataset weighting while keeping Hessians on a usable scale for XGBoost splits.
    return relative_weights * float(len(source_series))


def _summarize_dataset_weights(
    frame: pd.DataFrame,
    *,
    dataset_source_col: str,
) -> dict[str, dict[str, float | int]]:
    dataset_counts = frame[dataset_source_col].value_counts(dropna=False).sort_index()
    dataset_count = max(1, len(dataset_counts))
    scale_factor = float(len(frame))
    summary: dict[str, dict[str, float | int]] = {}
    for dataset_source, row_count in dataset_counts.items():
        relative_row_weight = 1.0 / (dataset_count * float(row_count))
        effective_row_weight = relative_row_weight * scale_factor
        summary[str(dataset_source)] = {
            "rows": int(row_count),
            "row_weight": float(effective_row_weight),
            "total_weight": float(relative_row_weight * float(row_count)),
            "effective_total_weight": float(effective_row_weight * float(row_count)),
        }
    return summary


def _compute_dataset_macro_wape(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    dataset_source_col: str,
    target_col: str,
) -> float:
    scored = frame[[dataset_source_col, target_col]].copy()
    scored["prediction"] = predictions
    dataset_wapes: list[float] = []
    for _, dataset_frame in scored.groupby(dataset_source_col, sort=False):
        dataset_wapes.append(compute_wape(dataset_frame[target_col], dataset_frame["prediction"]))
    if not dataset_wapes:
        raise ValueError("Unable to compute dataset-macro WAPE with an empty validation frame.")
    return float(np.mean(dataset_wapes))


def _resolve_baseline_prediction(
    frame: pd.DataFrame,
    baseline_name: str,
    *,
    target_contract: TargetContract | None = None,
) -> pd.Series | None:
    if baseline_name == "naive_lag_1":
        for column in ("sale_amount_lag_1", "lag_1"):
            if column in frame.columns:
                return frame[column]
        return None
    if baseline_name == "seasonal_naive_lag_7":
        anchor_candidates = []
        if target_contract is not None and target_contract.reconstruction_anchor_col is not None:
            anchor_candidates.append(target_contract.reconstruction_anchor_col)
        for column in (*anchor_candidates, "target_lag_7", "sale_amount_lag_7", "lag_7", "seasonal_naive_d7"):
            if column in frame.columns:
                return frame[column]
        return None
    if baseline_name == "trailing_mean_7":
        for column in ("sale_amount_rolling_mean_7", "rolling_mean_7"):
            if column in frame.columns:
                return frame[column]
        return None
    if baseline_name == "trailing_mean_28":
        for column in ("sale_amount_rolling_mean_28", "rolling_mean_28", "moving_average_28"):
            if column in frame.columns:
                return frame[column]
        return None
    if baseline_name == "same_weekday_mean_4":
        for column in ("target_same_dow_mean_4w", "same_dow_mean_4w"):
            if column in frame.columns:
                return frame[column]
        return None
    raise ValueError(f"Unsupported baseline `{baseline_name}`.")


def evaluate_statistical_baselines_macro(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    *,
    absolute_target_col: str,
    target_contract: TargetContract | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_names = (
        "naive_lag_1",
        "seasonal_naive_lag_7",
        "trailing_mean_7",
        "trailing_mean_28",
        "same_weekday_mean_4",
    )
    report_rows: list[dict[str, object]] = []
    for baseline_name in baseline_names:
        fold_results: list[dict[str, object]] = []
        for fold in folds:
            valid_frame = frame.iloc[fold["valid_idx"]]
            predictions = _resolve_baseline_prediction(
                valid_frame,
                baseline_name,
                target_contract=target_contract,
            )
            if predictions is None or pd.isna(predictions).all():
                fold_results = []
                break
            scored_frame = valid_frame[[absolute_target_col]].copy()
            scored_frame["prediction"] = pd.to_numeric(predictions, errors="coerce")
            scored_frame = scored_frame[
                scored_frame[absolute_target_col].notna() & scored_frame["prediction"].notna()
            ].copy()
            if scored_frame.empty:
                fold_results = []
                break
            fold_results.append(
                {
                    "dataset_source": str(fold["dataset_source"]),
                    "fold": int(fold["fold"]),
                    "wape": float(compute_wape(scored_frame[absolute_target_col], scored_frame["prediction"])),
                    "rows_scored": int(len(scored_frame)),
                }
            )
        if not fold_results:
            continue
        dataset_mean_wape: dict[str, float] = {}
        for dataset_source in sorted({result["dataset_source"] for result in fold_results}):
            dataset_scores = [result["wape"] for result in fold_results if result["dataset_source"] == dataset_source]
            dataset_mean_wape[dataset_source] = float(np.mean(dataset_scores))
        macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
        report_rows.append(
            {
                "baseline_name": baseline_name,
                "mean_wape": macro_mean_wape,
                "dataset_mean_wape": dataset_mean_wape,
                "fold_wape_scores": fold_results,
            }
        )

    if not report_rows:
        raise ValueError("No statistical baseline could be evaluated with the available columns.")

    best_row = min(report_rows, key=lambda row: row["mean_wape"])
    logger.info(
        "Best statistical baseline under macro-dataset WAPE: name=%s mean_wape=%.6f",
        best_row["baseline_name"],
        best_row["mean_wape"],
    )
    return report_rows, best_row


def _drop_constant_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    constant_feature_cols = [
        column
        for column in feature_cols
        if frame[column].nunique(dropna=False) <= 1
    ]
    filtered_feature_cols = [column for column in feature_cols if column not in set(constant_feature_cols)]
    return filtered_feature_cols, constant_feature_cols


def _build_feature_audit_payload(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
    raw_feature_cols: list[str],
    constant_feature_cols: list[str],
    identifier_feature_cols: list[str],
    folds: list[dict[str, object]],
    target_contract: TargetContract,
) -> dict[str, object]:
    def _target_summary(frame: pd.DataFrame, target_col: str) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for dataset_source, dataset_frame in frame.groupby(DEFAULT_DATASET_SOURCE_COL, sort=False):
            target = dataset_frame[target_col]
            summary[str(dataset_source)] = {
                "rows": int(len(dataset_frame)),
                "min": float(target.min()),
                "max": float(target.max()),
                "mean": float(target.mean()),
                "std": float(target.std()),
                "nonpositive_rows": int((target <= 0).fillna(False).sum()),
            }
        return summary

    missingness = []
    for column in feature_cols:
        missingness.append(
            {
                "feature": column,
                "null_rate_train": float(train_frame[column].isna().mean()),
                "null_rate_tuning": float(tuning_frame[column].isna().mean()),
            }
        )
    missingness.sort(key=lambda row: row["null_rate_train"], reverse=True)

    return {
        "raw_feature_count": len(raw_feature_cols),
        "final_feature_count": len(feature_cols),
        "constant_feature_cols": constant_feature_cols,
        "identifier_feature_cols": identifier_feature_cols,
        "excluded_risky_feature_cols": list(DEFAULT_EXCLUDED_RISKY_FEATURE_COLS),
        **build_target_contract_metadata(target_contract),
        "learning_target_summary_train": _target_summary(train_frame, target_contract.learning_target_col),
        "learning_target_summary_tuning": _target_summary(tuning_frame, target_contract.learning_target_col),
        "absolute_target_summary_train": _target_summary(train_frame, target_contract.absolute_target_col),
        "absolute_target_summary_tuning": _target_summary(tuning_frame, target_contract.absolute_target_col),
        "folds": [
            {
                key: value
                for key, value in fold.items()
                if key not in {"train_idx", "valid_idx"}
            }
            for fold in folds
        ],
        "top_missingness": missingness[:25],
    }


def _log_sampling_summary(label: str, metadata: dict[str, object]) -> None:
    logger.info(
        "%s sampling summary: sampled_rows=%s original_rows=%s sample_fraction=%.4f strategy=%s sample_store_col=%s",
        label,
        metadata["sampled_rows"],
        metadata["original_rows"],
        metadata["sample_fraction"],
        metadata["sample_strategy"],
        metadata["sample_store_col"],
    )
    for dataset_source, dataset_metadata in metadata["datasets"].items():
        logger.info(
            "%s retained rows: dataset=%s sampled_rows=%s original_rows=%s stores=%s unique_dates=%s strata=%s",
            label,
            dataset_source,
            dataset_metadata["sampled_rows"],
            dataset_metadata["original_rows"],
            dataset_metadata["store_count"],
            dataset_metadata["unique_dates"],
            dataset_metadata["strata_count"],
        )


def _build_gold_split_sampling_query(
    *,
    gold_table: str,
    split_bucket: str,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
    sample_fraction: float,
    selected_columns: list[str] | None = None,
) -> str:
    ordering_columns = [date_col, sample_store_col]
    for candidate in ("location_id", "product_id"):
        if candidate not in ordering_columns:
            ordering_columns.append(candidate)
    final_order_by = ", ".join(ordering_columns)
    select_list = "*"
    if selected_columns is not None:
        select_list = ", ".join(selected_columns)
    return f"""
with scoped as (
    select {select_list}
    from {gold_table}
    where split_bucket = '{split_bucket}'
),
ranked as (
    select
        *,
        row_number() over (
            partition by {dataset_source_col}, {date_col}, {sample_store_col}
            order by hash(
                coalesce(cast(product_id as varchar), ''),
                coalesce(cast(series_id as varchar), '')
            )
        ) as rn_sample,
        count(*) over (
            partition by {dataset_source_col}, {date_col}, {sample_store_col}
        ) as stratum_row_count
    from scoped
)
select *
from ranked
where rn_sample <= greatest(1, cast(ceil(stratum_row_count * {sample_fraction:.12f}) as bigint))
order by {final_order_by}
"""


def _resolve_gold_projection_columns(
    schema_preview: pd.DataFrame,
    *,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
) -> list[str]:
    required_columns = {
        date_col,
        dataset_source_col,
        sample_store_col,
        *DEFAULT_IDENTIFIER_FEATURE_COLS,
    }
    for column in schema_preview.columns:
        series = schema_preview[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            required_columns.add(column)
    return [column for column in schema_preview.columns if column in required_columns]


def _load_gold_split_sampling_metadata(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    split_bucket: str,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
    sample_fraction: float,
) -> dict[str, object]:
    original_counts = connection.execute(
        f"""
        with strata as (
            select
                {dataset_source_col} as dataset_source,
                {date_col} as sampled_dt,
                {sample_store_col} as sample_store,
                count(*) as stratum_rows
            from {gold_table}
            where split_bucket = '{split_bucket}'
            group by 1, 2, 3
        )
        select
            dataset_source,
            sum(stratum_rows) as original_rows,
            sum(greatest(1, cast(ceil(stratum_rows * {sample_fraction:.12f}) as bigint))) as sampled_rows,
            count(distinct sample_store) as store_count,
            count(distinct sampled_dt) as unique_dates,
            count(*) as strata_count
        from strata
        group by 1
        order by 1
        """
    ).fetchall()
    datasets: dict[str, dict[str, int | float]] = {}
    total_original_rows = 0
    total_sampled_rows = 0
    for dataset_source, original_rows, sampled_rows, store_count, unique_dates, strata_count in original_counts:
        dataset_original_rows = int(original_rows)
        dataset_sampled_rows = int(sampled_rows)
        datasets[str(dataset_source)] = {
            "original_rows": dataset_original_rows,
            "sampled_rows": dataset_sampled_rows,
            "sample_fraction": float(sample_fraction),
            "store_count": int(store_count),
            "unique_dates": int(unique_dates),
            "strata_count": int(strata_count),
        }
        total_original_rows += dataset_original_rows
        total_sampled_rows += dataset_sampled_rows
    return {
        "sample_fraction": float(sample_fraction),
        "sample_strategy": "date_store_stratified",
        "sample_store_col": sample_store_col,
        "original_rows": total_original_rows,
        "sampled_rows": total_sampled_rows,
        "datasets": datasets,
    }


def _resolve_sampling_order_cols(frame: pd.DataFrame) -> list[str]:
    order_cols: list[str] = []
    for candidate in ("product_id", "series_id"):
        if candidate in frame.columns:
            order_cols.append(candidate)
    return order_cols


def _sample_frame_stratified_by_date_store(
    frame: pd.DataFrame,
    *,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
    sample_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.empty:
        return frame.copy(), {
            "sample_fraction": float(sample_fraction),
            "sample_strategy": "date_store_stratified",
            "sample_store_col": sample_store_col,
            "original_rows": 0,
            "sampled_rows": 0,
            "datasets": {},
        }

    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("Sampling fraction must be in (0, 1].")

    ordered = frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    sort_columns = [dataset_source_col, date_col, sample_store_col, *_resolve_sampling_order_cols(ordered)]
    ordered = ordered.sort_values(sort_columns).reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    dataset_metadata: dict[str, dict[str, int | float]] = {}

    for dataset_source, dataset_frame in ordered.groupby(dataset_source_col, sort=False):
        per_dataset_parts: list[pd.DataFrame] = []
        for (_, _), stratum_frame in dataset_frame.groupby([date_col, sample_store_col], sort=False):
            sample_size = max(1, int(np.ceil(stratum_frame.shape[0] * sample_fraction)))
            per_dataset_parts.append(stratum_frame.head(sample_size))

        sampled_dataset_frame = pd.concat(per_dataset_parts, axis=0, ignore_index=False)
        sampled_parts.append(sampled_dataset_frame)
        dataset_metadata[str(dataset_source)] = {
            "original_rows": int(dataset_frame.shape[0]),
            "sampled_rows": int(sampled_dataset_frame.shape[0]),
            "sample_fraction": float(sample_fraction),
            "store_count": int(dataset_frame[sample_store_col].nunique()),
            "unique_dates": int(dataset_frame[date_col].nunique()),
            "strata_count": int(dataset_frame.groupby([date_col, sample_store_col], sort=False).ngroups),
        }

    sampled_frame = pd.concat(sampled_parts, axis=0, ignore_index=False)
    sampled_frame = sampled_frame.sort_values(sort_columns).reset_index(drop=True)
    return sampled_frame, {
        "sample_fraction": float(sample_fraction),
        "sample_strategy": "date_store_stratified",
        "sample_store_col": sample_store_col,
        "original_rows": int(frame.shape[0]),
        "sampled_rows": int(sampled_frame.shape[0]),
        "datasets": dataset_metadata,
    }


def _refresh_sampling_metadata_from_sampled_frame(
    metadata: dict[str, object],
    sampled_frame: pd.DataFrame,
    *,
    dataset_source_col: str,
    sample_store_col: str,
    date_col: str,
) -> dict[str, object]:
    refreshed = {
        **metadata,
        "datasets": {
            str(dataset_source): dict(dataset_metadata)
            for dataset_source, dataset_metadata in metadata["datasets"].items()
        },
    }
    sampled_counts = sampled_frame[dataset_source_col].value_counts(dropna=False).sort_index()
    refreshed["sampled_rows"] = int(len(sampled_frame))
    for dataset_source, sampled_rows in sampled_counts.items():
        dataset_key = str(dataset_source)
        dataset_metadata = refreshed["datasets"].setdefault(
            dataset_key,
            {
                "original_rows": int(sampled_rows),
                "sample_fraction": float(metadata["sample_fraction"]),
                "store_count": int(
                    sampled_frame.loc[sampled_frame[dataset_source_col] == dataset_source, sample_store_col].nunique()
                ),
                "unique_dates": int(
                    sampled_frame.loc[sampled_frame[dataset_source_col] == dataset_source, date_col].nunique()
                ),
                "strata_count": int(
                    sampled_frame.loc[sampled_frame[dataset_source_col] == dataset_source]
                    .groupby([date_col, sample_store_col], sort=False)
                    .ngroups
                ),
            },
        )
        dataset_metadata["sampled_rows"] = int(sampled_rows)
    return refreshed


def load_gold_train_tuning_frames(
    duckdb_path: str | Path = DEFAULT_DUCKDB_PATH,
    gold_table: str = DEFAULT_GOLD_TABLE,
    *,
    date_col: str = DEFAULT_DATE_COL,
    dataset_source_col: str = DEFAULT_DATASET_SOURCE_COL,
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION,
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object], dict[str, object]]:
    logger.info(
        "Loading sampled gold splits for optimisation: duckdb_path=%s gold_table=%s train_split=train tuning_split=val train_sample_fraction=%.4f tuning_sample_fraction=%.4f",
        duckdb_path,
        gold_table,
        train_sample_fraction,
        tuning_sample_fraction,
    )
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        schema_preview = connection.execute(f"select * from {gold_table} limit 0").fetchdf()
        sample_store_col = _resolve_sampling_store_col(schema_preview)
        projection_columns = _resolve_gold_projection_columns(
            schema_preview,
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
        )
        train_sampling_metadata = _load_gold_split_sampling_metadata(
            connection,
            gold_table=gold_table,
            split_bucket="train",
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            sample_fraction=train_sample_fraction,
        )
        tuning_sampling_metadata = _load_gold_split_sampling_metadata(
            connection,
            gold_table=gold_table,
            split_bucket="val",
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            sample_fraction=tuning_sample_fraction,
        )
        _log_sampling_summary("Planned train", train_sampling_metadata)
        _log_sampling_summary("Planned validation", tuning_sampling_metadata)
        logger.info("Submitting sampled gold train query.")
        train_frame = connection.execute(
            _build_gold_split_sampling_query(
                gold_table=gold_table,
                split_bucket="train",
                date_col=date_col,
                dataset_source_col=dataset_source_col,
                sample_store_col=sample_store_col,
                sample_fraction=train_sample_fraction,
                selected_columns=projection_columns,
            )
        ).fetchdf()
        logger.info("Submitting sampled gold validation query.")
        tuning_frame = connection.execute(
            _build_gold_split_sampling_query(
                gold_table=gold_table,
                split_bucket="val",
                date_col=date_col,
                dataset_source_col=dataset_source_col,
                sample_store_col=sample_store_col,
                sample_fraction=tuning_sample_fraction,
                selected_columns=projection_columns,
            )
        ).fetchdf()
    finally:
        connection.close()

    train_frame = downcast_pandas_frame(train_frame)
    tuning_frame = downcast_pandas_frame(tuning_frame)
    if train_frame.empty or tuning_frame.empty:
        raise ValueError("Gold dataset must contain non-empty train and val splits for optimisation.")
    train_sampling_metadata = _refresh_sampling_metadata_from_sampled_frame(
        train_sampling_metadata,
        train_frame,
        dataset_source_col=dataset_source_col,
        sample_store_col=sample_store_col,
        date_col=date_col,
    )
    tuning_sampling_metadata = _refresh_sampling_metadata_from_sampled_frame(
        tuning_sampling_metadata,
        tuning_frame,
        dataset_source_col=dataset_source_col,
        sample_store_col=sample_store_col,
        date_col=date_col,
    )
    logger.info(
        "Sampled gold splits loaded: train_rows=%s tuning_rows=%s columns=%s sample_store_col=%s",
        len(train_frame),
        len(tuning_frame),
        len(train_frame.columns),
        sample_store_col,
    )
    return train_frame, tuning_frame, sample_store_col, train_sampling_metadata, tuning_sampling_metadata


def build_optimisation_outputs(
    train_input_path: str | Path | None = DEFAULT_TRAIN_INPUT_PATH,
    tuning_input_path: str | Path | None = DEFAULT_TUNING_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_col: str = DEFAULT_DATE_COL,
    target_col: str = DEFAULT_TARGET_COL,
    n_folds: int = DEFAULT_N_FOLDS,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    tuning_random_seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
    duckdb_path: str | Path = DEFAULT_DUCKDB_PATH,
    gold_table: str = DEFAULT_GOLD_TABLE,
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION,
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION,
) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting optimisation artifact build: output_dir=%s n_folds=%s tuning_trials=%s train_sample_fraction=%.4f tuning_sample_fraction=%.4f",
        target_dir,
        n_folds,
        tuning_trials,
        train_sample_fraction,
        tuning_sample_fraction,
    )

    if train_input_path is None or tuning_input_path is None:
        logger.info("No parquet inputs provided; loading optimisation frames from gold.")
        (
            train_frame,
            tuning_frame,
            sample_store_col,
            train_sampling_metadata,
            tuning_sampling_metadata,
        ) = load_gold_train_tuning_frames(
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            date_col=date_col,
            dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
        )
        train_path = None
        tuning_path = None
    else:
        logger.info("Loading optimisation frames from parquet inputs: train=%s tuning=%s", train_input_path, tuning_input_path)
        train_path = Path(train_input_path)
        tuning_path = Path(tuning_input_path)
        train_frame = read_parquet_projected(train_path)
        tuning_frame = read_parquet_projected(tuning_path)

    train_frame[date_col] = pd.to_datetime(train_frame[date_col])
    tuning_frame[date_col] = pd.to_datetime(tuning_frame[date_col])
    if DEFAULT_DATASET_SOURCE_COL not in train_frame.columns:
        train_frame[DEFAULT_DATASET_SOURCE_COL] = "legacy"
    if DEFAULT_DATASET_SOURCE_COL not in tuning_frame.columns:
        tuning_frame[DEFAULT_DATASET_SOURCE_COL] = "legacy"
    train_frame = train_frame.sort_values(date_col).reset_index(drop=True)
    tuning_frame = tuning_frame.sort_values(date_col).reset_index(drop=True)
    target_contract = resolve_target_contract(
        train_frame,
        tuning_frame,
        requested_target_col=target_col,
    )
    train_frame = ensure_learning_target_column(train_frame, target_contract)
    tuning_frame = ensure_learning_target_column(tuning_frame, target_contract)
    resolved_target_col = target_contract.learning_target_col
    absolute_target_col = target_contract.absolute_target_col
    resolved_target_transform = target_contract.target_mode
    dropped_train_rows = int(train_frame[resolved_target_col].isna().sum())
    dropped_tuning_rows = int(tuning_frame[resolved_target_col].isna().sum())
    if dropped_train_rows > 0 or dropped_tuning_rows > 0:
        logger.info(
            "Dropping rows without a usable learning target: train_rows_dropped=%s tuning_rows_dropped=%s target_col=%s",
            dropped_train_rows,
            dropped_tuning_rows,
            resolved_target_col,
        )
        train_frame = train_frame[train_frame[resolved_target_col].notna()].copy()
        tuning_frame = tuning_frame[tuning_frame[resolved_target_col].notna()].copy()
    if train_input_path is not None and tuning_input_path is not None:
        sample_store_col = _resolve_sampling_store_col(train_frame)
        train_frame, train_sampling_metadata = _sample_frame_stratified_by_date_store(
            train_frame,
            date_col=date_col,
            dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
            sample_store_col=sample_store_col,
            sample_fraction=train_sample_fraction,
        )
        tuning_frame, tuning_sampling_metadata = _sample_frame_stratified_by_date_store(
            tuning_frame,
            date_col=date_col,
            dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
            sample_store_col=sample_store_col,
            sample_fraction=tuning_sample_fraction,
        )
    feature_cols = select_numeric_feature_columns(
        train_frame,
        excluded_cols={
            date_col,
            resolved_target_col,
            absolute_target_col,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
        },
    )
    identifier_feature_cols = [
        column
        for column in DEFAULT_IDENTIFIER_FEATURE_COLS
        if column in train_frame.columns and column not in feature_cols
    ]
    raw_feature_cols = [*feature_cols, *identifier_feature_cols]
    feature_cols = [*raw_feature_cols]
    feature_cols, constant_feature_cols = _drop_constant_feature_columns(train_frame, feature_cols)
    _log_sampling_summary("Train", train_sampling_metadata)
    _log_sampling_summary("Validation", tuning_sampling_metadata)
    train_dataset_weights = _summarize_dataset_weights(
        train_frame,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
    )
    tuning_dataset_weights = _summarize_dataset_weights(
        tuning_frame,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
    )
    logger.info(
        (
            "Loaded optimisation datasets: train_rows=%s tuning_rows=%s feature_count=%s "
            "learning_target_col=%s absolute_target_col=%s sample_store_col=%s"
        ),
        len(train_frame),
        len(tuning_frame),
        len(feature_cols),
        resolved_target_col,
        absolute_target_col,
        sample_store_col,
    )
    logger.info("Target contract for optimisation: %s", build_target_contract_metadata(target_contract))
    logger.info("Identifier features included in optimisation: %s", identifier_feature_cols)
    logger.info("Constant features dropped before optimisation: %s", constant_feature_cols)
    logger.info("Train dataset weights summary: %s", train_dataset_weights)
    logger.info("Validation dataset weights summary: %s", tuning_dataset_weights)

    logger.info("Building walk-forward folds by dataset on sampled validation split.")
    baseline_folds = build_tuning_walk_forward_folds_by_dataset(
        tuning_frame,
        date_col=date_col,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        n_folds=n_folds,
    )
    training_folds = build_grouped_tuning_walk_forward_folds_by_dataset(
        tuning_frame,
        date_col=date_col,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        n_folds=n_folds,
    )
    logger.info("Evaluating statistical baselines on sampled validation split with dataset-macro WAPE.")
    baseline_rows, best_baseline = evaluate_statistical_baselines_macro(
        tuning_frame,
        baseline_folds,
        absolute_target_col=absolute_target_col,
        target_contract=target_contract,
    )
    logger.info(
        "Best statistical baseline selected under dataset-macro WAPE: name=%s mean_wape=%.6f",
        best_baseline["baseline_name"],
        best_baseline["mean_wape"],
    )
    logger.info("Starting Optuna XGBoost optimisation with native Optuna logs enabled.")
    best_params, tuning_report = optimize_xgboost_params(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=training_folds,
        feature_cols=feature_cols,
        baseline_wape=float(best_baseline["mean_wape"]),
        target_contract=target_contract,
        tuning_trials=tuning_trials,
        random_seed=tuning_random_seed,
        model_params=model_params,
        target_transform=resolved_target_transform,
    )

    best_params_path = target_dir / "best_optuna_params.json"
    trials_report_path = target_dir / "optuna_trials_report.csv"
    baseline_report_path = target_dir / "baseline_report.json"
    metadata_path = target_dir / "optimisation_metadata.json"
    feature_audit_path = target_dir / "optimisation_feature_audit.json"

    _json_dump(best_params_path, best_params)
    tuning_report.to_csv(trials_report_path, index=False)
    _json_dump(
        baseline_report_path,
        {
            "best_baseline": best_baseline,
            "all_baselines": baseline_rows,
        },
    )
    _json_dump(
        feature_audit_path,
        _build_feature_audit_payload(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            feature_cols=feature_cols,
            raw_feature_cols=raw_feature_cols,
            constant_feature_cols=constant_feature_cols,
            identifier_feature_cols=identifier_feature_cols,
            folds=baseline_folds,
            target_contract=target_contract,
        ),
    )
    _json_dump(
        metadata_path,
        {
            "train_input_path": str(train_path) if train_path is not None else None,
            "tuning_input_path": str(tuning_path) if tuning_path is not None else None,
            "duckdb_path": str(duckdb_path) if train_path is None else None,
            "gold_table": gold_table if train_path is None else None,
            "n_folds": n_folds,
            "tuning_trials": tuning_trials,
            "tuning_random_seed": tuning_random_seed,
            **build_target_contract_metadata(target_contract),
            "sample_store_col": sample_store_col,
            "identifier_feature_cols": identifier_feature_cols,
            "constant_feature_cols": constant_feature_cols,
            "excluded_risky_feature_cols": list(DEFAULT_EXCLUDED_RISKY_FEATURE_COLS),
            "train_sampling": train_sampling_metadata,
            "tuning_sampling": tuning_sampling_metadata,
            "train_dataset_weights": train_dataset_weights,
            "tuning_dataset_weights": tuning_dataset_weights,
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_frame)),
            "tuning_rows": int(len(tuning_frame)),
            "folds": [
                {
                    key: value
                    for key, value in fold.items()
                    if key not in {"train_idx", "valid_idx"}
                }
                for fold in baseline_folds
            ],
            "training_folds": [
                {
                    key: value
                    for key, value in fold.items()
                    if key not in {"train_idx", "valid_idx"}
                }
                for fold in training_folds
            ],
        },
    )
    logger.info(
        "Optimisation artifacts written: best_params=%s trials_report=%s",
        best_params_path,
        trials_report_path,
    )
    return {
        "best_params": best_params_path,
        "optuna_trials_report": trials_report_path,
        "baseline_report": baseline_report_path,
        "optimisation_metadata": metadata_path,
        "optimisation_feature_audit": feature_audit_path,
    }
