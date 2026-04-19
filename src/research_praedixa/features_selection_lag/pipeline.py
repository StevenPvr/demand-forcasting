from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import platform
from pathlib import Path
import tempfile

import numpy as np
import optuna
import pandas as pd

from research_praedixa.memory_utils import (
    get_parquet_columns,
    load_numeric_parquet_column,
    read_parquet_projected,
)
from research_praedixa.xgboost_utils import (
    DEFAULT_XGBOOST_MODEL_PARAMS,
    fit_xgboost_booster,
    predict_with_xgboost_booster,
)


DEFAULT_INPUT_PATH = Path("data/data_cleaning/data_train_cleaned.parquet")
DEFAULT_OUTPUT_DIR = Path("data/features_selection_lag")
DEFAULT_TARGET_COL = "target"
DEFAULT_DATE_COL = "dt"
DEFAULT_TRAIN_FRACTION = 0.7
DEFAULT_N_FOLDS = 5
DEFAULT_PEARSON_THRESHOLD = 0.95
DEFAULT_SPEARMAN_THRESHOLD = 0.95
DEFAULT_TUNING_TRIALS = 50
DEFAULT_RANDOM_SEED = 42
DEFAULT_EARLY_STOPPING_ROUNDS = 100
DEFAULT_LAG_PATTERNS = ("_lag_", "_rolling_", "_ewm_")
DEFAULT_PROGRESS_LOG_EVERY = 25
DEFAULT_TUNING_PROGRESS_LOG_EVERY = 10
DEFAULT_MEMMAP_PROGRESS_LOG_EVERY = 25
DEFAULT_CORRELATION_PROGRESS_LOG_EVERY = 25
DEFAULT_MODEL_PARAMS = {**DEFAULT_XGBOOST_MODEL_PARAMS, "n_estimators": 2000}


logger = logging.getLogger(__name__)


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def get_lag_candidate_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if any(pattern in column for pattern in DEFAULT_LAG_PATTERNS)
    ]


def get_lag_candidate_columns_from_names(column_names: list[str]) -> list[str]:
    return [
        column
        for column in column_names
        if any(pattern in column for pattern in DEFAULT_LAG_PATTERNS)
    ]


def split_chronological_train_tuning(
    frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dated = frame.copy()
    dated[date_col] = pd.to_datetime(dated[date_col])
    dated = dated.sort_values(date_col).reset_index(drop=True)

    unique_dates = pd.Index(dated[date_col].drop_duplicates().sort_values())
    split_idx = max(1, int(len(unique_dates) * train_fraction))
    split_idx = min(split_idx, len(unique_dates) - 1)

    train_dates = unique_dates[:split_idx]
    holdout_dates = unique_dates[split_idx:]

    selection_train = dated[dated[date_col].isin(train_dates)].copy().reset_index(drop=True)
    tuning_holdout = dated[dated[date_col].isin(holdout_dates)].copy().reset_index(drop=True)

    metadata = {
        "train_fraction": train_fraction,
        "total_rows": int(len(frame)),
        "train_rows": int(len(selection_train)),
        "holdout_rows": int(len(tuning_holdout)),
        "total_unique_dates": int(len(unique_dates)),
        "train_unique_dates": int(len(train_dates)),
        "holdout_unique_dates": int(len(holdout_dates)),
        "train_start_date": train_dates.min().strftime("%Y-%m-%d"),
        "train_end_date": train_dates.max().strftime("%Y-%m-%d"),
        "holdout_start_date": holdout_dates.min().strftime("%Y-%m-%d"),
        "holdout_end_date": holdout_dates.max().strftime("%Y-%m-%d"),
    }
    logger.info(
        "Chronological split complete: train_rows=%s holdout_rows=%s train_dates=%s holdout_dates=%s",
        metadata["train_rows"],
        metadata["holdout_rows"],
        metadata["train_unique_dates"],
        metadata["holdout_unique_dates"],
    )
    return selection_train, tuning_holdout, metadata


def build_walk_forward_folds(
    frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    ordered = frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
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
        "Walk-forward folds built: n_folds=%s valid_dates_per_fold=%s",
        len(folds),
        valid_dates_per_fold,
    )
    return folds


def _drop_zero_variance_columns(frame: pd.DataFrame, candidate_cols: list[str]) -> tuple[list[str], dict[str, str]]:
    survivors: list[str] = []
    dropped: dict[str, str] = {}
    for column in candidate_cols:
        if frame[column].nunique(dropna=False) <= 1:
            dropped[column] = "zero_variance"
        else:
            survivors.append(column)
    return survivors, dropped


def _greedy_corr_filter(
    corr_matrix: pd.DataFrame,
    ranked_candidates: list[str],
    threshold: float,
    stage: str,
) -> tuple[list[str], dict[str, tuple[str, float, str]]]:
    survivors: list[str] = []
    dropped: dict[str, tuple[str, float, str]] = {}
    for column in ranked_candidates:
        drop_reason: tuple[str, float, str] | None = None
        for survivor in survivors:
            corr_value = float(corr_matrix.loc[column, survivor])
            if corr_value >= threshold:
                drop_reason = (survivor, corr_value, stage)
                break
        if drop_reason is None:
            survivors.append(column)
        else:
            dropped[column] = drop_reason
    return survivors, dropped


def filter_correlated_lag_features(
    frame: pd.DataFrame,
    candidate_cols: list[str],
    target_col: str = DEFAULT_TARGET_COL,
    pearson_threshold: float = DEFAULT_PEARSON_THRESHOLD,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    survivors, zero_var_dropped = _drop_zero_variance_columns(frame, candidate_cols)
    if not survivors:
        report = pd.DataFrame(
            {
                "feature": candidate_cols,
                "target_abs_corr": 0.0,
                "pearson_selected": False,
                "spearman_selected": False,
                "dropped_by": [zero_var_dropped.get(column, "unknown") for column in candidate_cols],
                "dropped_with": "",
                "corr_value": np.nan,
            }
        )
        return [], report

    target_abs_corr = frame[survivors].corrwith(frame[target_col], method="pearson").abs().fillna(0.0)
    ranked_candidates = target_abs_corr.sort_values(ascending=False).index.tolist()

    pearson_matrix = frame[ranked_candidates].corr(method="pearson").abs().fillna(0.0)
    pearson_survivors, pearson_dropped = _greedy_corr_filter(
        corr_matrix=pearson_matrix,
        ranked_candidates=ranked_candidates,
        threshold=pearson_threshold,
        stage="pearson",
    )

    spearman_matrix = frame[pearson_survivors].corr(method="spearman").abs().fillna(0.0)
    spearman_survivors, spearman_dropped = _greedy_corr_filter(
        corr_matrix=spearman_matrix,
        ranked_candidates=pearson_survivors,
        threshold=spearman_threshold,
        stage="spearman",
    )

    report_rows: list[dict[str, object]] = []
    final_selected = set(spearman_survivors)
    pearson_selected_set = set(pearson_survivors)
    for column in candidate_cols:
        dropped_by = ""
        dropped_with = ""
        corr_value = np.nan

        if column in zero_var_dropped:
            dropped_by = zero_var_dropped[column]
        elif column in pearson_dropped:
            dropped_with, corr_value, dropped_by = pearson_dropped[column]
        elif column in spearman_dropped:
            dropped_with, corr_value, dropped_by = spearman_dropped[column]

        report_rows.append(
            {
                "feature": column,
                "target_abs_corr": float(target_abs_corr.get(column, 0.0)),
                "pearson_selected": column in pearson_selected_set,
                "spearman_selected": column in final_selected,
                "dropped_by": dropped_by,
                "dropped_with": dropped_with,
                "corr_value": corr_value,
            }
        )

    report = pd.DataFrame(report_rows).sort_values(
        by=["spearman_selected", "pearson_selected", "target_abs_corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    logger.info(
        "Correlation filtering complete: initial=%s after_zero_variance=%s after_pearson=%s after_spearman=%s",
        len(candidate_cols),
        len(survivors),
        len(pearson_survivors),
        len(spearman_survivors),
    )
    return spearman_survivors, report


def compute_wape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = float(np.abs(y_true_arr).sum())
    if denominator == 0.0:
        return float("inf")
    return float(np.abs(y_true_arr - y_pred_arr).sum() / denominator)


def compute_wape_improvement_pct(base_wape: float, candidate_wape: float) -> float:
    if base_wape == 0.0:
        return 0.0
    return float(100.0 * (base_wape - candidate_wape) / base_wape)


def _corr_arrays(left: np.ndarray, right: np.ndarray, method: str) -> float:
    valid_mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(valid_mask):
        return 0.0

    left_valid = left[valid_mask]
    right_valid = right[valid_mask]
    if left_valid.size <= 1 or right_valid.size <= 1:
        return 0.0

    if np.nanstd(left_valid) == 0.0 or np.nanstd(right_valid) == 0.0:
        return 0.0

    if method == "pearson":
        corr_value = np.corrcoef(left_valid, right_valid)[0, 1]
    elif method == "spearman":
        corr_value = pd.Series(left_valid).corr(pd.Series(right_valid), method="spearman")
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    if pd.isna(corr_value):
        return 0.0
    return float(abs(corr_value))


def _build_candidate_memmap(
    source_path: Path,
    candidate_cols: list[str],
    selection_mask: np.ndarray,
    work_dir: Path,
) -> tuple[Path, np.memmap]:
    with tempfile.NamedTemporaryFile(
        prefix="lag_candidates_",
        suffix=".mmap",
        dir=work_dir,
        delete=False,
    ) as handle:
        mmap_path = Path(handle.name)

    row_count = int(selection_mask.sum())
    candidate_matrix = np.memmap(
        mmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(row_count, len(candidate_cols)),
    )
    logger.info(
        "Building lag candidate memmap: rows=%s candidates=%s path=%s",
        row_count,
        len(candidate_cols),
        mmap_path,
    )
    for candidate_idx, candidate in enumerate(candidate_cols):
        candidate_matrix[:, candidate_idx] = load_numeric_parquet_column(
            path=source_path,
            column=candidate,
            mask=selection_mask,
            dtype=np.float32,
        )
        processed = candidate_idx + 1
        if (
            processed == 1
            or processed % DEFAULT_MEMMAP_PROGRESS_LOG_EVERY == 0
            or processed == len(candidate_cols)
        ):
            logger.info(
                "Lag memmap progress: loaded=%s/%s last_feature=%s",
                processed,
                len(candidate_cols),
                candidate,
            )
    candidate_matrix.flush()
    logger.info("Lag candidate memmap complete: path=%s", mmap_path)
    return mmap_path, candidate_matrix


def filter_correlated_lag_features_memmap(
    candidate_matrix: np.memmap,
    candidate_cols: list[str],
    target_values: np.ndarray,
    pearson_threshold: float = DEFAULT_PEARSON_THRESHOLD,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    logger.info(
        "Starting correlation filtering on memmap candidates: candidates=%s pearson_threshold=%.2f spearman_threshold=%.2f",
        len(candidate_cols),
        pearson_threshold,
        spearman_threshold,
    )
    zero_var_dropped: dict[str, str] = {}
    target_abs_corr: dict[str, float] = {}
    survivors: list[str] = []
    candidate_to_index = {column: index for index, column in enumerate(candidate_cols)}

    for index, column in enumerate(candidate_cols, start=1):
        column_values = np.asarray(candidate_matrix[:, candidate_to_index[column]], dtype=np.float32)
        finite_values = column_values[np.isfinite(column_values)]
        if finite_values.size <= 1 or float(np.nanstd(finite_values)) == 0.0:
            zero_var_dropped[column] = "zero_variance"
            target_abs_corr[column] = 0.0
        else:
            target_abs_corr[column] = _corr_arrays(column_values, target_values, method="pearson")
            survivors.append(column)
        if (
            index == 1
            or index % DEFAULT_CORRELATION_PROGRESS_LOG_EVERY == 0
            or index == len(candidate_cols)
        ):
            logger.info(
                "Correlation progress [target scan]: processed=%s/%s survivors=%s zero_variance=%s last_feature=%s",
                index,
                len(candidate_cols),
                len(survivors),
                len(zero_var_dropped),
                column,
            )

    if not survivors:
        report = pd.DataFrame(
            {
                "feature": candidate_cols,
                "target_abs_corr": [target_abs_corr.get(column, 0.0) for column in candidate_cols],
                "pearson_selected": False,
                "spearman_selected": False,
                "dropped_by": [zero_var_dropped.get(column, "unknown") for column in candidate_cols],
                "dropped_with": "",
                "corr_value": np.nan,
            }
        )
        return [], report

    ranked_candidates = sorted(survivors, key=lambda column: target_abs_corr[column], reverse=True)

    pearson_survivors: list[str] = []
    pearson_dropped: dict[str, tuple[str, float, str]] = {}
    for index, column in enumerate(ranked_candidates, start=1):
        column_values = np.asarray(candidate_matrix[:, candidate_to_index[column]], dtype=np.float32)
        drop_reason: tuple[str, float, str] | None = None
        for survivor in pearson_survivors:
            survivor_values = np.asarray(candidate_matrix[:, candidate_to_index[survivor]], dtype=np.float32)
            corr_value = _corr_arrays(column_values, survivor_values, method="pearson")
            if corr_value >= pearson_threshold:
                drop_reason = (survivor, corr_value, "pearson")
                break
        if drop_reason is None:
            pearson_survivors.append(column)
        else:
            pearson_dropped[column] = drop_reason
        if (
            index == 1
            or index % DEFAULT_CORRELATION_PROGRESS_LOG_EVERY == 0
            or index == len(ranked_candidates)
        ):
            logger.info(
                "Correlation progress [pearson prune]: processed=%s/%s kept=%s dropped=%s last_feature=%s",
                index,
                len(ranked_candidates),
                len(pearson_survivors),
                len(pearson_dropped),
                column,
            )

    spearman_survivors: list[str] = []
    spearman_dropped: dict[str, tuple[str, float, str]] = {}
    for index, column in enumerate(pearson_survivors, start=1):
        column_values = np.asarray(candidate_matrix[:, candidate_to_index[column]], dtype=np.float32)
        drop_reason = None
        for survivor in spearman_survivors:
            survivor_values = np.asarray(candidate_matrix[:, candidate_to_index[survivor]], dtype=np.float32)
            corr_value = _corr_arrays(column_values, survivor_values, method="spearman")
            if corr_value >= spearman_threshold:
                drop_reason = (survivor, corr_value, "spearman")
                break
        if drop_reason is None:
            spearman_survivors.append(column)
        else:
            spearman_dropped[column] = drop_reason
        if (
            index == 1
            or index % DEFAULT_CORRELATION_PROGRESS_LOG_EVERY == 0
            or index == len(pearson_survivors)
        ):
            logger.info(
                "Correlation progress [spearman prune]: processed=%s/%s kept=%s dropped=%s last_feature=%s",
                index,
                len(pearson_survivors),
                len(spearman_survivors),
                len(spearman_dropped),
                column,
            )

    final_selected = set(spearman_survivors)
    pearson_selected_set = set(pearson_survivors)
    report_rows: list[dict[str, object]] = []
    for column in candidate_cols:
        dropped_by = ""
        dropped_with = ""
        corr_value = np.nan
        if column in zero_var_dropped:
            dropped_by = zero_var_dropped[column]
        elif column in pearson_dropped:
            dropped_with, corr_value, dropped_by = pearson_dropped[column]
        elif column in spearman_dropped:
            dropped_with, corr_value, dropped_by = spearman_dropped[column]

        report_rows.append(
            {
                "feature": column,
                "target_abs_corr": float(target_abs_corr.get(column, 0.0)),
                "pearson_selected": column in pearson_selected_set,
                "spearman_selected": column in final_selected,
                "dropped_by": dropped_by,
                "dropped_with": dropped_with,
                "corr_value": corr_value,
            }
        )

    report = pd.DataFrame(report_rows).sort_values(
        by=["spearman_selected", "pearson_selected", "target_abs_corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    logger.info(
        "Correlation filtering complete: initial=%s after_zero_variance=%s after_pearson=%s after_spearman=%s",
        len(candidate_cols),
        len(survivors),
        len(pearson_survivors),
        len(spearman_survivors),
    )
    return spearman_survivors, report


def resolve_parallelism(
    requested_workers: int | None,
    candidate_count: int,
    total_threads: int | None = None,
) -> tuple[int, int]:
    available_threads = max(1, total_threads or (os.cpu_count() or 1))
    if _is_macos():
        logger.info(
            "macOS detected: forcing outer parallelism to 1 worker for XGBoost stability, using all cores inside each fit."
        )
        return 1, available_threads
    max_workers = max(1, requested_workers or available_threads)
    resolved_workers = max(1, min(max_workers, max(1, candidate_count)))
    threads_per_worker = max(1, available_threads // resolved_workers)
    return resolved_workers, threads_per_worker


def _same_weekday_mean_4(frame: pd.DataFrame) -> pd.Series:
    if "target_same_dow_mean_4w" in frame.columns:
        return frame["target_same_dow_mean_4w"]
    candidate_cols = ["sale_amount_lag_7", "sale_amount_lag_14", "sale_amount_lag_21", "sale_amount_lag_28"]
    available_cols = [column for column in candidate_cols if column in frame.columns]
    if not available_cols:
        return pd.Series(np.nan, index=frame.index)
    return frame[available_cols].mean(axis=1)


def _sample_trial_params(trial_index: int, random_seed: int = DEFAULT_RANDOM_SEED) -> dict[str, object]:
    rng = np.random.default_rng(random_seed + trial_index)
    learning_rate = float(np.exp(rng.uniform(np.log(0.01), np.log(0.2))))
    return {
        "objective": "reg:squarederror",
        "n_estimators": int(rng.integers(100, 501)),
        "learning_rate": learning_rate,
        "max_depth": int(rng.integers(3, 11)),
        "min_child_weight": float(rng.uniform(1.0, 20.0)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
        "random_state": int(random_seed + trial_index),
        "verbosity": 0,
        "tree_method": "hist",
    }


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


def _resolve_baseline_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return None


def evaluate_statistical_baselines(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    target_col: str = DEFAULT_TARGET_COL,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_predictors = {
        "naive_lag_1": lambda part: _resolve_baseline_column(part, ("sale_amount_lag_1", "lag_1")),
        "seasonal_naive_lag_7": lambda part: _resolve_baseline_column(
            part,
            ("target_lag_7", "target_seasonal_naive_d7", "sale_amount_lag_7", "lag_7"),
        ),
        "trailing_mean_7": lambda part: _resolve_baseline_column(
            part,
            ("sale_amount_rolling_mean_7", "rolling_mean_7"),
        ),
        "trailing_mean_28": lambda part: _resolve_baseline_column(
            part,
            ("sale_amount_rolling_mean_28", "rolling_mean_28"),
        ),
        "same_weekday_mean_4": _same_weekday_mean_4,
    }

    report_rows: list[dict[str, object]] = []
    for baseline_name, predictor in baseline_predictors.items():
        fold_scores: list[float] = []
        for fold in folds:
            valid_frame = frame.iloc[fold["valid_idx"]]
            y_true = valid_frame[target_col]
            y_pred = predictor(valid_frame)
            if y_pred is None:
                fold_scores = []
                break
            if pd.isna(y_pred).all():
                fold_scores = []
                break
            fold_scores.append(compute_wape(y_true, y_pred))
        if not fold_scores:
            continue
        report_rows.append(
            {
                "baseline_name": baseline_name,
                "mean_wape": float(np.mean(fold_scores)),
                "fold_wape_scores": [float(score) for score in fold_scores],
            }
        )

    if not report_rows:
        raise ValueError("No statistical baseline could be evaluated with the available columns.")

    best_row = min(report_rows, key=lambda row: row["mean_wape"])
    logger.info(
        "Best statistical baseline: name=%s mean_wape=%.6f",
        best_row["baseline_name"],
        best_row["mean_wape"],
    )
    return report_rows, best_row


def _fit_and_score_xgboost(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_col: str = DEFAULT_TARGET_COL,
    model_params: dict[str, object] | None = None,
    num_threads_override: int | None = None,
) -> list[float]:
    resolved_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    num_threads = int(num_threads_override or resolved_params.get("n_jobs", os.cpu_count() or 1))
    fold_workers = max(1, min(len(folds), num_threads))
    threads_per_fold = max(1, num_threads // fold_workers)

    def _fit_single_fold(fold: dict[str, object]) -> float:
        train_frame = frame.iloc[fold["train_idx"]]
        valid_frame = frame.iloc[fold["valid_idx"]]
        model = fit_xgboost_booster(
            train_frame,
            feature_cols,
            target_col=target_col,
            model_params=resolved_params,
            default_params=DEFAULT_MODEL_PARAMS,
            default_n_estimators=300,
            valid_frame=valid_frame,
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
            num_threads_override=threads_per_fold,
        )
        predictions = predict_with_xgboost_booster(model, valid_frame, feature_cols)
        return compute_wape(valid_frame[target_col], predictions)

    if fold_workers == 1:
        return [_fit_single_fold(fold) for fold in folds]

    with ThreadPoolExecutor(max_workers=fold_workers) as executor:
        future_to_index = {
            executor.submit(_fit_single_fold, fold): index
            for index, fold in enumerate(folds)
        }
        scores_by_index: dict[int, float] = {}
        for future in as_completed(future_to_index):
            scores_by_index[future_to_index[future]] = float(future.result())
    return [scores_by_index[index] for index in range(len(folds))]


def optimize_non_lag_model_params(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_col: str = DEFAULT_TARGET_COL,
    n_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    baseline_wape: float | None = None,
    base_model_params: dict[str, object] | None = None,
    num_workers: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if baseline_wape is None:
        raise ValueError("baseline_wape must be provided for non-lag tuning.")

    total_threads = int((base_model_params or {}).get("n_jobs", os.cpu_count() or 1))
    logger.info(
        "Starting non-lag parameter tuning with Optuna: trials=%s feature_count=%s baseline_wape=%.6f",
        n_trials,
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
        trial_params = {**(base_model_params or {}), **_sample_optuna_params(trial=trial, random_seed=random_seed)}
        fold_scores = _fit_and_score_xgboost(
            frame=frame,
            folds=folds,
            feature_cols=feature_cols,
            target_col=target_col,
            model_params=trial_params,
            num_threads_override=total_threads,
        )
        mean_wape = float(np.mean(fold_scores))
        improvement_pct = compute_wape_improvement_pct(baseline_wape, mean_wape)
        trial.set_user_attr("mean_wape", mean_wape)
        trial.set_user_attr("fold_wape_scores", [float(score) for score in fold_scores])
        trial_counter["completed"] += 1
        completed = trial_counter["completed"]
        if completed == 1 or completed % DEFAULT_TUNING_PROGRESS_LOG_EVERY == 0 or completed == n_trials:
            logger.info(
                "Tuning progress: trial=%s/%s current_best_improvement_pct=%.6f",
                completed,
                n_trials,
                max((t.value for t in study.trials if t.value is not None), default=improvement_pct),
            )
        return improvement_pct

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    tuning_rows = []
    for trial in study.trials:
        trial_params = {**trial.params}
        trial_params["random_state"] = trial.user_attrs.get("random_state", random_seed + trial.number + 1)
        tuning_rows.append(
            {
                "trial": trial.number,
                "mean_wape": float(trial.user_attrs["mean_wape"]),
                "baseline_wape_improvement_pct": float(trial.value),
                "fold_wape_scores": trial.user_attrs["fold_wape_scores"],
                **trial.params,
            }
        )
    tuning_report = pd.DataFrame(tuning_rows).sort_values(
        by="baseline_wape_improvement_pct",
        ascending=False,
    ).reset_index(drop=True)
    best_params = {**(base_model_params or {}), **study.best_params}
    best_params["random_state"] = random_seed + study.best_trial.number + 1
    best_params["verbosity"] = 0
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"
    best_params["tree_method"] = "hist"
    best_params["n_jobs"] = total_threads
    logger.info(
        "Non-lag parameter tuning complete: best_trial=%s best_wape=%.6f best_improvement_pct=%.6f",
        study.best_trial.number,
        study.best_trial.user_attrs["mean_wape"],
        study.best_trial.value,
    )
    return best_params, tuning_report


def _score_candidate_task(
    *,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate: str,
    target_col: str,
    model_params: dict[str, object] | None,
    threads_per_worker: int,
    base_mean_wape: float,
    base_scores: list[float],
) -> dict[str, object]:
    candidate_scores = _fit_and_score_xgboost(
        frame=frame,
        folds=folds,
        feature_cols=[*base_feature_cols, candidate],
        target_col=target_col,
        model_params=model_params,
        num_threads_override=threads_per_worker,
    )
    candidate_mean_wape = float(np.mean(candidate_scores))
    gain_vs_base_model = base_mean_wape - candidate_mean_wape
    wape_improvement_pct = compute_wape_improvement_pct(base_mean_wape, candidate_mean_wape)
    keep = wape_improvement_pct > 0.0
    return {
        "feature": candidate,
        "base_mean_wape": base_mean_wape,
        "candidate_mean_wape": candidate_mean_wape,
        "gain_vs_base_model": gain_vs_base_model,
        "wape_improvement_pct": wape_improvement_pct,
        "keep": keep,
        "base_fold_wape_scores": [float(score) for score in base_scores],
        "candidate_fold_wape_scores": [float(score) for score in candidate_scores],
    }


def _score_candidate_task_memmap(
    *,
    base_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate: str,
    candidate_values: np.ndarray,
    target_col: str,
    model_params: dict[str, object] | None,
    threads_per_worker: int,
    base_mean_wape: float,
    base_scores: list[float],
) -> dict[str, object]:
    candidate_frame = base_frame.copy(deep=False)
    candidate_frame[candidate] = candidate_values
    candidate_scores = _fit_and_score_xgboost(
        frame=candidate_frame,
        folds=folds,
        feature_cols=[*base_feature_cols, candidate],
        target_col=target_col,
        model_params=model_params,
        num_threads_override=threads_per_worker,
    )
    candidate_mean_wape = float(np.mean(candidate_scores))
    gain_vs_base_model = base_mean_wape - candidate_mean_wape
    wape_improvement_pct = compute_wape_improvement_pct(base_mean_wape, candidate_mean_wape)
    keep = wape_improvement_pct > 0.0
    return {
        "feature": candidate,
        "base_mean_wape": base_mean_wape,
        "candidate_mean_wape": candidate_mean_wape,
        "gain_vs_base_model": gain_vs_base_model,
        "wape_improvement_pct": wape_improvement_pct,
        "keep": keep,
        "base_fold_wape_scores": [float(score) for score in base_scores],
        "candidate_fold_wape_scores": [float(score) for score in candidate_scores],
    }


def run_single_addon_lag_selection(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate_cols: list[str],
    target_col: str = DEFAULT_TARGET_COL,
    model_params: dict[str, object] | None = None,
    num_workers: int | None = None,
    checkpoint_every: int = DEFAULT_PROGRESS_LOG_EVERY,
    checkpoint_callback=None,
) -> tuple[list[str], pd.DataFrame, dict[str, object]]:
    total_threads = int((model_params or {}).get("n_jobs", os.cpu_count() or 1))
    resolved_workers, threads_per_worker = resolve_parallelism(
        requested_workers=num_workers,
        candidate_count=len(candidate_cols),
        total_threads=total_threads,
    )
    base_scores = _fit_and_score_xgboost(
        frame=frame,
        folds=folds,
        feature_cols=base_feature_cols,
        target_col=target_col,
        model_params=model_params,
        num_threads_override=total_threads,
    )
    base_mean_wape = float(np.mean(base_scores))
    logger.info(
        "Base XGBoost model scored: feature_count=%s mean_wape=%.6f candidate_count=%s workers=%s threads_per_worker=%s",
        len(base_feature_cols),
        base_mean_wape,
        len(candidate_cols),
        resolved_workers,
        threads_per_worker,
    )

    selected_features: list[str] = []
    report_rows: list[dict[str, object]] = []
    if resolved_workers == 1:
        for index, candidate in enumerate(candidate_cols, start=1):
            result = _score_candidate_task(
                frame=frame,
                folds=folds,
                base_feature_cols=base_feature_cols,
                candidate=candidate,
                target_col=target_col,
                model_params=model_params,
                threads_per_worker=threads_per_worker,
                base_mean_wape=base_mean_wape,
                base_scores=base_scores,
            )
            if result["keep"]:
                selected_features.append(candidate)
            report_rows.append(result)
            if checkpoint_callback and (index == 1 or index % checkpoint_every == 0 or index == len(candidate_cols)):
                checkpoint_callback(report_rows, selected_features)
            if index == 1 or index % DEFAULT_PROGRESS_LOG_EVERY == 0 or index == len(candidate_cols):
                logger.info(
                    "Add-on progress: tested=%s/%s kept=%s last_feature=%s wape_improvement_pct=%.6f",
                    index,
                    len(candidate_cols),
                    len(selected_features),
                    candidate,
                    result["wape_improvement_pct"],
                )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_candidate = {
                executor.submit(
                    _score_candidate_task,
                    frame=frame,
                    folds=folds,
                    base_feature_cols=base_feature_cols,
                    candidate=candidate,
                    target_col=target_col,
                    model_params=model_params,
                    threads_per_worker=threads_per_worker,
                    base_mean_wape=base_mean_wape,
                    base_scores=base_scores,
                ): candidate
                for candidate in candidate_cols
            }
            results_by_candidate: dict[str, dict[str, object]] = {}
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                result = future.result()
                results_by_candidate[candidate] = result
                completed += 1
                if result["keep"]:
                    selected_features.append(candidate)
                if checkpoint_callback and (
                    completed == 1 or completed % checkpoint_every == 0 or completed == len(candidate_cols)
                ):
                    checkpoint_callback(
                        [results_by_candidate[c] for c in candidate_cols if c in results_by_candidate],
                        [c for c in candidate_cols if c in results_by_candidate and results_by_candidate[c]["keep"]],
                    )
                if completed == 1 or completed % DEFAULT_PROGRESS_LOG_EVERY == 0 or completed == len(candidate_cols):
                    logger.info(
                        "Add-on progress: tested=%s/%s kept=%s last_feature=%s wape_improvement_pct=%.6f",
                        completed,
                        len(candidate_cols),
                        len(selected_features),
                        candidate,
                        result["wape_improvement_pct"],
                    )
        report_rows = [results_by_candidate[candidate] for candidate in candidate_cols]
        selected_features = [candidate for candidate in candidate_cols if results_by_candidate[candidate]["keep"]]

    report = pd.DataFrame(report_rows).sort_values(by="wape_improvement_pct", ascending=False).reset_index(drop=True)
    base_model_report = {
        "base_feature_count": len(base_feature_cols),
        "base_mean_wape": base_mean_wape,
        "base_fold_wape_scores": [float(score) for score in base_scores],
        "num_workers": resolved_workers,
        "threads_per_worker": threads_per_worker,
        "total_threads": total_threads,
    }
    logger.info(
        "Single add-on selection complete: selected=%s/%s",
        len(selected_features),
        len(candidate_cols),
    )
    return selected_features, report, base_model_report


def run_single_addon_lag_selection_memmap(
    *,
    base_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate_cols: list[str],
    candidate_matrix: np.memmap,
    candidate_to_index: dict[str, int],
    target_col: str = DEFAULT_TARGET_COL,
    model_params: dict[str, object] | None = None,
    num_workers: int | None = None,
    checkpoint_every: int = DEFAULT_PROGRESS_LOG_EVERY,
    checkpoint_callback=None,
) -> tuple[list[str], pd.DataFrame, dict[str, object]]:
    total_threads = int((model_params or {}).get("n_jobs", os.cpu_count() or 1))
    resolved_workers, threads_per_worker = resolve_parallelism(
        requested_workers=num_workers,
        candidate_count=len(candidate_cols),
        total_threads=total_threads,
    )
    base_scores = _fit_and_score_xgboost(
        frame=base_frame,
        folds=folds,
        feature_cols=base_feature_cols,
        target_col=target_col,
        model_params=model_params,
        num_threads_override=total_threads,
    )
    base_mean_wape = float(np.mean(base_scores))
    logger.info(
        "Base XGBoost model scored: feature_count=%s mean_wape=%.6f candidate_count=%s workers=%s threads_per_worker=%s",
        len(base_feature_cols),
        base_mean_wape,
        len(candidate_cols),
        resolved_workers,
        threads_per_worker,
    )

    selected_features: list[str] = []
    report_rows: list[dict[str, object]] = []
    if resolved_workers == 1:
        for index, candidate in enumerate(candidate_cols, start=1):
            result = _score_candidate_task_memmap(
                base_frame=base_frame,
                folds=folds,
                base_feature_cols=base_feature_cols,
                candidate=candidate,
                candidate_values=np.asarray(candidate_matrix[:, candidate_to_index[candidate]], dtype=np.float32),
                target_col=target_col,
                model_params=model_params,
                threads_per_worker=threads_per_worker,
                base_mean_wape=base_mean_wape,
                base_scores=base_scores,
            )
            if result["keep"]:
                selected_features.append(candidate)
            report_rows.append(result)
            if checkpoint_callback and (index == 1 or index % checkpoint_every == 0 or index == len(candidate_cols)):
                checkpoint_callback(report_rows, selected_features)
            if index == 1 or index % DEFAULT_PROGRESS_LOG_EVERY == 0 or index == len(candidate_cols):
                logger.info(
                    "Add-on progress: tested=%s/%s kept=%s last_feature=%s wape_improvement_pct=%.6f",
                    index,
                    len(candidate_cols),
                    len(selected_features),
                    candidate,
                    result["wape_improvement_pct"],
                )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_candidate = {
                executor.submit(
                    _score_candidate_task_memmap,
                    base_frame=base_frame,
                    folds=folds,
                    base_feature_cols=base_feature_cols,
                    candidate=candidate,
                    candidate_values=np.asarray(candidate_matrix[:, candidate_to_index[candidate]], dtype=np.float32),
                    target_col=target_col,
                    model_params=model_params,
                    threads_per_worker=threads_per_worker,
                    base_mean_wape=base_mean_wape,
                    base_scores=base_scores,
                ): candidate
                for candidate in candidate_cols
            }
            results_by_candidate: dict[str, dict[str, object]] = {}
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                result = future.result()
                results_by_candidate[candidate] = result
                completed += 1
                if result["keep"]:
                    selected_features.append(candidate)
                if checkpoint_callback and (
                    completed == 1 or completed % checkpoint_every == 0 or completed == len(candidate_cols)
                ):
                    checkpoint_callback(
                        [results_by_candidate[c] for c in candidate_cols if c in results_by_candidate],
                        [c for c in candidate_cols if c in results_by_candidate and results_by_candidate[c]["keep"]],
                    )
                if completed == 1 or completed % DEFAULT_PROGRESS_LOG_EVERY == 0 or completed == len(candidate_cols):
                    logger.info(
                        "Add-on progress: tested=%s/%s kept=%s last_feature=%s wape_improvement_pct=%.6f",
                        completed,
                        len(candidate_cols),
                        len(selected_features),
                        candidate,
                        result["wape_improvement_pct"],
                    )
        report_rows = [results_by_candidate[candidate] for candidate in candidate_cols]
        selected_features = [candidate for candidate in candidate_cols if results_by_candidate[candidate]["keep"]]

    report = pd.DataFrame(report_rows).sort_values(by="wape_improvement_pct", ascending=False).reset_index(drop=True)
    base_model_report = {
        "base_feature_count": len(base_feature_cols),
        "base_mean_wape": base_mean_wape,
        "base_fold_wape_scores": [float(score) for score in base_scores],
        "num_workers": resolved_workers,
        "threads_per_worker": threads_per_worker,
        "total_threads": total_threads,
    }
    logger.info(
        "Single add-on selection complete: selected=%s/%s",
        len(selected_features),
        len(candidate_cols),
    )
    return selected_features, report, base_model_report


def _json_dump(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_addon_checkpoint(
    addon_report_path: Path,
    selected_lag_features_path: Path,
    report_rows: list[dict[str, object]],
    selected_features: list[str],
) -> None:
    if report_rows:
        pd.DataFrame(report_rows).sort_values(by="wape_improvement_pct", ascending=False).to_csv(
            addon_report_path,
            index=False,
        )
    _json_dump(
        selected_lag_features_path,
        {
            "selected_lag_features": selected_features,
            "selected_lag_feature_count": len(selected_features),
            "evaluated_feature_count": len(report_rows),
        },
    )


def build_lag_selection_outputs(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_col: str = DEFAULT_DATE_COL,
    target_col: str = DEFAULT_TARGET_COL,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    n_folds: int = DEFAULT_N_FOLDS,
    pearson_threshold: float = DEFAULT_PEARSON_THRESHOLD,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    tuning_random_seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
    num_workers: int | None = None,
) -> dict[str, Path]:
    source_path = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting lag feature selection: input=%s output_dir=%s", source_path, target_dir)

    all_columns = get_parquet_columns(source_path)
    lag_candidate_cols = get_lag_candidate_columns_from_names(all_columns)
    non_lag_feature_cols = [
        column
        for column in all_columns
        if column not in {target_col, date_col}
        and column not in lag_candidate_cols
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
        "total_unique_dates": int(len(unique_dates)),
        "train_unique_dates": int(len(train_dates)),
        "holdout_unique_dates": int(len(holdout_dates)),
        "train_start_date": train_dates.min().strftime("%Y-%m-%d"),
        "train_end_date": train_dates.max().strftime("%Y-%m-%d"),
        "holdout_start_date": holdout_dates.min().strftime("%Y-%m-%d"),
        "holdout_end_date": holdout_dates.max().strftime("%Y-%m-%d"),
    }
    logger.info(
        "Chronological split complete: train_rows=%s holdout_rows=%s train_dates=%s holdout_dates=%s",
        split_metadata["train_rows"],
        split_metadata["holdout_rows"],
        split_metadata["train_unique_dates"],
        split_metadata["holdout_unique_dates"],
    )
    logger.info(
        "Feature groups resolved: lag_candidates=%s non_lag_features=%s",
        len(lag_candidate_cols),
        len(non_lag_feature_cols),
    )

    base_cols = [date_col, target_col, *non_lag_feature_cols]
    base_frame = read_parquet_projected(source_path, columns=base_cols)
    base_frame[date_col] = pd.to_datetime(base_frame[date_col])
    selection_train = base_frame.loc[selection_mask, base_cols].copy().reset_index(drop=True)
    tuning_holdout = base_frame.loc[holdout_mask, base_cols].copy().reset_index(drop=True)
    logger.info("Loaded cleaned training dataset: rows=%s cols=%s", total_rows, len(all_columns))

    work_dir = target_dir / "_cache"
    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_memmap_path: Path | None = None
    candidate_matrix: np.memmap | None = None

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

    try:
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

        baseline_frame = selection_train.loc[:, [date_col, target_col]].copy()
        for baseline_col in baseline_feature_cols:
            if baseline_col in candidate_to_index:
                baseline_frame[baseline_col] = np.asarray(
                    candidate_matrix[:, candidate_to_index[baseline_col]],
                    dtype=np.float32,
                )
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
            num_workers=num_workers,
        )

        train_output_path = target_dir / "train_selection_70_selected.parquet"
        tuning_output_path = target_dir / "train_tuning_30_selected.parquet"
        selected_lag_features_path = target_dir / "selected_lag_features.json"
        correlation_report_path = target_dir / "lag_correlation_filter_report.csv"
        tuning_report_path = target_dir / "non_lag_model_tuning_report.csv"
        best_params_path = target_dir / "best_non_lag_model_params.json"
        addon_report_path = target_dir / "lag_addon_importance_report.csv"
        baseline_report_path = target_dir / "baseline_report.json"
        split_metadata_path = target_dir / "split_metadata.json"

        correlation_report.to_csv(correlation_report_path, index=False)
        tuning_report.to_csv(tuning_report_path, index=False)
        _json_dump(best_params_path, tuned_model_params)
        _write_addon_checkpoint(
            addon_report_path=addon_report_path,
            selected_lag_features_path=selected_lag_features_path,
            report_rows=[],
            selected_features=[],
        )
        selected_lag_features, addon_report, base_model_report = run_single_addon_lag_selection_memmap(
            base_frame=selection_train,
            folds=folds,
            base_feature_cols=non_lag_feature_cols,
            candidate_cols=filtered_lag_cols,
            candidate_matrix=candidate_matrix,
            candidate_to_index=candidate_to_index,
            target_col=target_col,
            model_params=tuned_model_params,
            num_workers=num_workers,
            checkpoint_every=DEFAULT_PROGRESS_LOG_EVERY,
            checkpoint_callback=lambda report_rows, selected_features: _write_addon_checkpoint(
                addon_report_path=addon_report_path,
                selected_lag_features_path=selected_lag_features_path,
                report_rows=report_rows,
                selected_features=selected_features,
            ),
        )

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
            *[column for column in required_scoring_context_cols if column not in {*non_lag_feature_cols, *selected_lag_features}],
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
                "base_model_report": base_model_report,
                "best_non_lag_model_params": tuned_model_params,
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
                "folds": [
                    {
                        key: value
                        for key, value in fold.items()
                        if key not in {"train_idx", "valid_idx"}
                    }
                    for fold in folds
                ],
            },
        )
        logger.info(
            "Lag feature selection artifacts written: train=%s tuning=%s selected_lags=%s",
            train_output_path,
            tuning_output_path,
            len(selected_lag_features),
        )

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
    finally:
        if candidate_matrix is not None:
            candidate_matrix.flush()
            del candidate_matrix
        if candidate_memmap_path is not None and candidate_memmap_path.exists():
            candidate_memmap_path.unlink()
