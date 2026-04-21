from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd

from praedixa.demand_forecast.feature_screening.constants import DEFAULT_LAG_PATTERNS
from praedixa.demand_forecast.feature_screening.correlation import get_lag_candidate_columns_from_names
from praedixa.demand_forecast.feature_screening.orchestrator_models import PreparedLagSelectionInputs
from praedixa.platform.utils.memory import get_parquet_columns, read_parquet_projected


def json_dump(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_addon_checkpoint(
    addon_report_path: Path,
    selected_lag_features_path: Path,
    report_rows: list[dict[str, object]],
    selected_features: list[str],
) -> None:
    if report_rows:
        pd.DataFrame(report_rows).sort_values(by="wape_improvement_pct", ascending=False).to_csv(addon_report_path, index=False)
    json_dump(
        selected_lag_features_path,
        {
            "selected_lag_features": selected_features,
            "selected_lag_feature_count": len(selected_features),
            "evaluated_feature_count": len(report_rows),
        },
    )


def prepare_lag_selection_inputs(
    *,
    input_path: str | Path | None,
    output_dir: str | Path,
    duckdb_path: str | Path,
    gold_table: str,
    date_col: str,
    target_col: str,
    train_fraction: float,
    logger: logging.Logger,
) -> PreparedLagSelectionInputs:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve_feature_selection_source_path(
        input_path=input_path,
        target_dir=target_dir,
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        logger=logger,
    )
    logger.info("Starting lag feature selection: input=%s output_dir=%s", source_path, target_dir)
    all_columns = get_parquet_columns(source_path)
    lag_candidate_cols = get_lag_candidate_columns_from_names(all_columns, lag_patterns=DEFAULT_LAG_PATTERNS)
    split_metadata, selection_mask, holdout_mask, selection_train, base_frame = build_split_artifacts(
        source_path=source_path,
        date_col=date_col,
        target_col=target_col,
        train_fraction=train_fraction,
    )
    work_dir = target_dir / "_cache"
    work_dir.mkdir(parents=True, exist_ok=True)
    return PreparedLagSelectionInputs(
        target_dir=target_dir,
        source_path=source_path,
        all_columns=all_columns,
        lag_candidate_cols=lag_candidate_cols,
        split_metadata=split_metadata,
        selection_mask=selection_mask,
        holdout_mask=holdout_mask,
        selection_train=selection_train,
        base_frame=base_frame,
        work_dir=work_dir,
        output_paths=build_lag_selection_output_paths(target_dir),
    )


def resolve_feature_selection_source_path(
    *,
    input_path: str | Path | None,
    target_dir: Path,
    duckdb_path: str | Path,
    gold_table: str,
    logger: logging.Logger,
) -> Path:
    if input_path is not None:
        return Path(input_path)
    return materialize_feature_selection_source_from_gold(
        target_dir=target_dir,
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        logger=logger,
    )


def materialize_feature_selection_source_from_gold(
    *,
    target_dir: Path,
    duckdb_path: str | Path,
    gold_table: str,
    logger: logging.Logger,
) -> Path:
    cache_dir = target_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    materialized_path = cache_dir / "feature_selection_source_train.parquet"
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        connection.execute(
            f"""
            copy (
                select *
                from {gold_table}
                where split_bucket = 'train'
            ) to '{materialized_path.as_posix()}' (format parquet)
            """
        )
    finally:
        connection.close()
    logger.info(
        "Materialized feature-selection source from gold: duckdb_path=%s gold_table=%s output=%s",
        duckdb_path,
        gold_table,
        materialized_path,
    )
    return materialized_path


def build_split_artifacts(
    *,
    source_path: Path,
    date_col: str,
    target_col: str,
    train_fraction: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    all_columns = get_parquet_columns(source_path)
    lag_candidate_cols = get_lag_candidate_columns_from_names(all_columns, lag_patterns=DEFAULT_LAG_PATTERNS)
    non_lag_feature_cols = [column for column in all_columns if column not in {target_col, date_col} and column not in lag_candidate_cols]
    date_frame = read_parquet_projected(source_path, columns=[date_col])
    date_frame[date_col] = pd.to_datetime(date_frame[date_col])
    unique_dates = sorted_unique_dates(date_frame, date_col)
    train_dates, holdout_dates = split_unique_dates(unique_dates, train_fraction)
    selection_mask = date_frame[date_col].isin(train_dates.tolist()).to_numpy()
    holdout_mask = date_frame[date_col].isin(holdout_dates.tolist()).to_numpy()
    split_metadata = split_metadata_payload(date_frame, unique_dates, train_dates, holdout_dates, train_fraction)
    base_cols = [date_col, target_col, *non_lag_feature_cols]
    base_frame = read_parquet_projected(source_path, columns=base_cols)
    base_frame[date_col] = pd.to_datetime(base_frame[date_col])
    selection_train = base_frame.loc[selection_mask, base_cols].copy().reset_index(drop=True)
    return split_metadata, selection_mask, holdout_mask, selection_train, base_frame


def build_lag_selection_output_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "train_selected": target_dir / "train_selection_70_selected.parquet",
        "tuning_selected": target_dir / "train_tuning_30_selected.parquet",
        "selected_lag_features": target_dir / "selected_lag_features.json",
        "lag_correlation_report": target_dir / "lag_correlation_filter_report.csv",
        "non_lag_model_tuning_report": target_dir / "non_lag_model_tuning_report.csv",
        "best_non_lag_model_params": target_dir / "best_non_lag_model_params.json",
        "lag_addon_importance_report": target_dir / "lag_addon_importance_report.csv",
        "baseline_report": target_dir / "baseline_report.json",
        "split_metadata": target_dir / "split_metadata.json",
    }


def required_scoring_context_columns(all_columns: list[str]) -> list[str]:
    candidates = ("target_lag_7", "sale_amount_lag_7", "lag_7", "target_same_dow_mean_4w", "same_dow_mean_4w")
    return [column for column in candidates if column in all_columns]


def build_final_output_columns(
    *,
    date_col: str,
    target_col: str,
    non_lag_feature_cols: list[str],
    selected_lag_features: list[str],
    required_scoring_context_cols: list[str],
) -> list[str]:
    extra_context_cols = [column for column in required_scoring_context_cols if column not in {*non_lag_feature_cols, *selected_lag_features}]
    return [date_col, target_col, *non_lag_feature_cols, *selected_lag_features, *extra_context_cols]


def materialize_selected_frames(
    *,
    source_path: Path,
    final_cols: list[str],
    date_col: str,
    selection_mask: np.ndarray,
    holdout_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_frame = read_parquet_projected(source_path, columns=final_cols)
    final_frame[date_col] = pd.to_datetime(final_frame[date_col])
    train_selected = final_frame.loc[selection_mask, final_cols].copy().sort_values([date_col]).reset_index(drop=True)
    tuning_selected = final_frame.loc[holdout_mask, final_cols].copy().sort_values([date_col]).reset_index(drop=True)
    return train_selected, tuning_selected


def write_selected_frames(
    *,
    train_selected: pd.DataFrame,
    tuning_selected: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    train_selected.to_parquet(output_paths["train_selected"], index=False)
    tuning_selected.to_parquet(output_paths["tuning_selected"], index=False)


def sorted_unique_dates(frame: pd.DataFrame, date_col: str) -> pd.DatetimeIndex:
    date_values = pd.Series(pd.to_datetime(frame[date_col], errors="raise"), copy=False)
    return pd.DatetimeIndex(date_values.drop_duplicates().sort_values())


def split_unique_dates(
    unique_dates: pd.DatetimeIndex,
    train_fraction: float,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    split_idx = max(1, int(len(unique_dates) * train_fraction))
    split_idx = min(split_idx, len(unique_dates) - 1)
    return unique_dates[:split_idx], unique_dates[split_idx:]


def split_metadata_payload(
    date_frame: pd.DataFrame,
    unique_dates: pd.DatetimeIndex,
    train_dates: pd.DatetimeIndex,
    holdout_dates: pd.DatetimeIndex,
    train_fraction: float,
) -> dict[str, object]:
    selection_mask = date_frame.iloc[:, 0].isin(train_dates.tolist()).to_numpy()
    holdout_mask = date_frame.iloc[:, 0].isin(holdout_dates.tolist()).to_numpy()
    return {
        "train_fraction": train_fraction,
        "total_rows": int(len(date_frame)),
        "train_rows": int(selection_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "total_unique_dates": int(len(unique_dates)),
        "train_unique_dates": int(len(train_dates)),
        "holdout_unique_dates": int(len(holdout_dates)),
        "train_start_date": format_date_bound(train_dates.min()),
        "train_end_date": format_date_bound(train_dates.max()),
        "holdout_start_date": format_date_bound(holdout_dates.min()),
        "holdout_end_date": format_date_bound(holdout_dates.max()),
    }


def format_date_bound(value: object) -> str:
    return pd.Timestamp(cast(Any, value)).strftime("%Y-%m-%d")
