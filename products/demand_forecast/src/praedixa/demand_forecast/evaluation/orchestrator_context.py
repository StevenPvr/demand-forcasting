from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_mapping import resolve_explicit_tft_layout
from praedixa.demand_forecast.contracts.targets import (
    TargetContract,
    build_target_contract_metadata,
    resolve_target_contract,
)
from praedixa.demand_forecast.evaluation.modeling import select_feature_columns
from praedixa.demand_forecast.evaluation.reference_mode import (
    load_gold_reference_mode_frames,
    load_local_mode_frames,
    prepare_scored_test_frame,
    prepare_training_target_frame,
)
from praedixa.demand_forecast.evaluation.reporting import load_best_params


@dataclass(frozen=True)
class LoadedEvaluationFrames:
    evaluation_mode: str
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame
    history_reference: pd.DataFrame
    scored_reference_test: pd.DataFrame
    overlap_metadata: dict[str, object] | None


@dataclass(frozen=True)
class EvaluationPreparedContext:
    best_params: dict[str, Any]
    evaluation_mode: str
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame
    history_reference: pd.DataFrame
    scored_reference_test: pd.DataFrame
    target_contract: TargetContract
    feature_cols: list[str]
    constant_feature_cols: list[str]
    identifier_feature_cols: list[str]
    missing_in_test_feature_cols: list[str]
    dropped_test_rows: int
    overlap_metadata: dict[str, object] | None


def ensure_evaluation_target_dir(output_dir: str | Path) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _load_local_evaluation_frames(
    *,
    train_selection_input_path: str | Path,
    train_tuning_input_path: str | Path,
    val_input_path: str | Path,
    requested_target_col: str,
) -> LoadedEvaluationFrames:
    loaded = load_local_mode_frames(
        train_selection_input_path=train_selection_input_path,
        train_tuning_input_path=train_tuning_input_path,
        val_input_path=val_input_path,
        requested_target_col=requested_target_col,
    )
    return LoadedEvaluationFrames("local_parquet", *loaded)


def _load_gold_evaluation_frames(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
) -> LoadedEvaluationFrames:
    loaded = load_gold_reference_mode_frames(
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
    )
    return LoadedEvaluationFrames("bakery_reference_overlap", *loaded)


def load_evaluation_frames(
    *,
    train_selection_input_path: str | Path | None,
    train_tuning_input_path: str | Path | None,
    val_input_path: str | Path | None,
    requested_target_col: str,
    duckdb_path: str | Path,
    gold_table: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    logger: logging.Logger,
) -> LoadedEvaluationFrames:
    explicit_local_mode = (
        train_selection_input_path is not None
        and train_tuning_input_path is not None
        and val_input_path is not None
    )
    if explicit_local_mode:
        logger.info(
            "Running evaluation in local parquet mode: train=%s valid=%s test=%s",
            train_selection_input_path,
            train_tuning_input_path,
            val_input_path,
        )
        return _load_local_evaluation_frames(
            train_selection_input_path=cast(str | Path, train_selection_input_path),
            train_tuning_input_path=cast(str | Path, train_tuning_input_path),
            val_input_path=cast(str | Path, val_input_path),
            requested_target_col=requested_target_col,
        )
    logger.info(
        "Running evaluation in bakery reference mode: duckdb_path=%s gold_table=%s train_sample_fraction=%.4f tuning_sample_fraction=%.4f bakery_test=unsampled_reference_split reference_protocol=bakery_product_arima_equivalent_from_gold",
        duckdb_path,
        gold_table,
        train_sample_fraction,
        tuning_sample_fraction,
    )
    return _load_gold_evaluation_frames(
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
    )


def _resolve_feature_context(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_contract: TargetContract,
) -> tuple[list[str], list[str], list[str], list[str]]:
    combined_pretest_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    feature_cols, constant_feature_cols, identifier_feature_cols = select_feature_columns(
        combined_pretest_frame,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
    )
    missing_in_test_feature_cols = [column for column in feature_cols if column not in test_frame.columns]
    filtered_feature_cols = [column for column in feature_cols if column in test_frame.columns]
    return filtered_feature_cols, constant_feature_cols, identifier_feature_cols, missing_in_test_feature_cols


def _real_feature_columns(
    feature_cols: list[str],
) -> list[str]:
    layout = resolve_explicit_tft_layout(feature_cols)
    ordered_reals: list[str] = []
    for column in [*layout["static_reals"], *layout["time_varying_known_reals"], *layout["time_varying_unknown_reals"]]:
        if column not in ordered_reals:
            ordered_reals.append(column)
    return ordered_reals


def _feature_cols_all_null_in_test(
    test_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> list[str]:
    return [column for column in feature_cols if column in test_frame.columns and bool(test_frame[column].isna().all())]


def _train_real_feature_fill_values(
    train_frame: pd.DataFrame,
    *,
    real_feature_cols: list[str],
) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for column in real_feature_cols:
        numeric_values = pd.to_numeric(train_frame[column], errors="coerce").dropna()
        fill_values[column] = float(numeric_values.median()) if not numeric_values.empty else 0.0
    return fill_values


def _impute_real_feature_values(
    frame: pd.DataFrame,
    *,
    real_feature_fill_values: dict[str, float],
) -> pd.DataFrame:
    imputed = frame.copy()
    for column, fill_value in real_feature_fill_values.items():
        if column not in imputed.columns:
            continue
        imputed[column] = pd.to_numeric(imputed[column], errors="coerce").fillna(fill_value)
    return imputed


def _sanitize_evaluation_feature_frames(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    missing_in_test_feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    fully_missing_in_test_feature_cols = _feature_cols_all_null_in_test(test_frame, feature_cols=feature_cols)
    resolved_feature_cols = [
        column
        for column in feature_cols
        if column not in fully_missing_in_test_feature_cols
    ]
    real_feature_fill_values = _train_real_feature_fill_values(
        train_frame,
        real_feature_cols=_real_feature_columns(resolved_feature_cols),
    )
    return (
        _impute_real_feature_values(train_frame, real_feature_fill_values=real_feature_fill_values),
        _impute_real_feature_values(valid_frame, real_feature_fill_values=real_feature_fill_values),
        _impute_real_feature_values(test_frame, real_feature_fill_values=real_feature_fill_values),
        resolved_feature_cols,
        [*missing_in_test_feature_cols, *fully_missing_in_test_feature_cols],
    )


def _log_evaluation_dataset_summary(
    *,
    loaded: LoadedEvaluationFrames,
    target_contract: TargetContract,
    feature_cols: list[str],
    constant_feature_cols: list[str],
    identifier_feature_cols: list[str],
    missing_in_test_feature_cols: list[str],
    dropped_test_rows: int,
    logger: logging.Logger,
) -> dict[str, object] | None:
    logger.info(
        "Loaded evaluation datasets: mode=%s train_rows=%s valid_rows=%s test_rows=%s feature_count=%s learning_target_col=%s absolute_target_col=%s",
        loaded.evaluation_mode,
        len(loaded.train_frame),
        len(loaded.valid_frame),
        len(loaded.test_frame),
        len(feature_cols),
        target_contract.learning_target_col,
        target_contract.absolute_target_col,
    )
    logger.info("Target contract for evaluation: %s", build_target_contract_metadata(target_contract))
    if identifier_feature_cols:
        logger.info("Identifier features included in evaluation: %s", identifier_feature_cols)
    if constant_feature_cols:
        logger.info("Constant features dropped before evaluation: %s", constant_feature_cols)
    if missing_in_test_feature_cols:
        logger.info("Dropping evaluation features unavailable at inference time: %s", missing_in_test_feature_cols)
    if loaded.overlap_metadata is None:
        return None
    overlap_metadata = {
        **loaded.overlap_metadata,
        "scorable_test_rows": int(len(loaded.test_frame)),
        "dropped_test_rows_without_learning_target": int(dropped_test_rows),
    }
    logger.info("Bakery reference overlap: %s", overlap_metadata)
    return overlap_metadata


def _prepared_target_frames(
    *,
    loaded: LoadedEvaluationFrames,
    target_contract: TargetContract,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    train_frame = prepare_training_target_frame(loaded.train_frame, target_contract=target_contract)
    valid_frame = prepare_training_target_frame(loaded.valid_frame, target_contract=target_contract)
    test_frame, scored_reference_test, dropped_test_rows = prepare_scored_test_frame(
        loaded.test_frame,
        loaded.scored_reference_test,
        target_contract=target_contract,
    )
    return train_frame, valid_frame, test_frame, scored_reference_test, dropped_test_rows


def _resolved_target_contract(
    *,
    loaded: LoadedEvaluationFrames,
    requested_target_col: str,
) -> TargetContract:
    combined_pretest_frame = pd.concat([loaded.train_frame, loaded.valid_frame], ignore_index=True)
    return resolve_target_contract(
        combined_pretest_frame,
        loaded.test_frame,
        requested_target_col=requested_target_col,
    )


def _evaluation_context_payload(
    *,
    best_params: dict[str, Any],
    loaded: LoadedEvaluationFrames,
    target_contract: TargetContract,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    scored_reference_test: pd.DataFrame,
    feature_cols: list[str],
    constant_feature_cols: list[str],
    identifier_feature_cols: list[str],
    missing_in_test_feature_cols: list[str],
    dropped_test_rows: int,
    overlap_metadata: dict[str, object] | None,
) -> EvaluationPreparedContext:
    return EvaluationPreparedContext(
        best_params=best_params,
        evaluation_mode=loaded.evaluation_mode,
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        history_reference=loaded.history_reference,
        scored_reference_test=scored_reference_test,
        target_contract=target_contract,
        feature_cols=feature_cols,
        constant_feature_cols=constant_feature_cols,
        identifier_feature_cols=identifier_feature_cols,
        missing_in_test_feature_cols=missing_in_test_feature_cols,
        dropped_test_rows=dropped_test_rows,
        overlap_metadata=overlap_metadata,
    )


def prepare_evaluation_context(
    *,
    loaded: LoadedEvaluationFrames,
    requested_target_col: str,
    best_params_path: str | Path,
    logger: logging.Logger,
) -> EvaluationPreparedContext:
    best_params = load_best_params(best_params_path)
    target_contract = _resolved_target_contract(loaded=loaded, requested_target_col=requested_target_col)
    train_frame, valid_frame, test_frame, scored_reference_test, dropped_test_rows = (
        _prepared_target_frames(
            loaded=loaded,
            target_contract=target_contract,
        )
    )
    feature_cols, constant_feature_cols, identifier_feature_cols, missing_in_test_feature_cols = (
        _resolve_feature_context(
            train_frame=train_frame,
            valid_frame=valid_frame,
            test_frame=test_frame,
            target_contract=target_contract,
        )
    )
    train_frame, valid_frame, test_frame, feature_cols, missing_in_test_feature_cols = (
        _sanitize_evaluation_feature_frames(
            train_frame=train_frame,
            valid_frame=valid_frame,
            test_frame=test_frame,
            feature_cols=feature_cols,
            missing_in_test_feature_cols=missing_in_test_feature_cols,
        )
    )
    overlap_metadata = _log_evaluation_dataset_summary(
        loaded=loaded,
        target_contract=target_contract,
        feature_cols=feature_cols,
        constant_feature_cols=constant_feature_cols,
        identifier_feature_cols=identifier_feature_cols,
        missing_in_test_feature_cols=missing_in_test_feature_cols,
        dropped_test_rows=dropped_test_rows,
        logger=logger,
    )
    return _evaluation_context_payload(
        best_params=best_params,
        loaded=loaded,
        target_contract=target_contract,
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        scored_reference_test=scored_reference_test,
        feature_cols=feature_cols,
        constant_feature_cols=constant_feature_cols,
        identifier_feature_cols=identifier_feature_cols,
        missing_in_test_feature_cols=missing_in_test_feature_cols,
        dropped_test_rows=dropped_test_rows,
        overlap_metadata=overlap_metadata,
    )
