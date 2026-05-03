from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_mapping import (
    TFT_EXPLICIT_ROLE_BY_COLUMN,
)
from praedixa.demand_forecast.contracts.targets import (
    TargetContract,
    build_target_contract_metadata,
    resolve_target_contract,
)
from praedixa.demand_forecast.evaluation.constants import (
    DEFAULT_BAKERY_REFERENCE_DATASET_SOURCE,
    DEFAULT_TRAINING_FEATURE_MANIFEST_PATH,
)
from praedixa.demand_forecast.evaluation.modeling import select_feature_columns
from praedixa.demand_forecast.evaluation.reference_mode import (
    load_gold_reference_mode_frames,
    load_local_mode_frames,
    prepare_scored_test_frame,
    prepare_training_target_frame,
    uses_bakery_reference_transfer_holdout,
)
from praedixa.demand_forecast.evaluation.reporting import load_best_params
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS,
)


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
    model_backend: str
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
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    evaluation_dataset_source: str | None,
    logger: logging.Logger,
) -> LoadedEvaluationFrames:
    loaded = load_local_mode_frames(
        train_selection_input_path=train_selection_input_path,
        train_tuning_input_path=train_tuning_input_path,
        val_input_path=val_input_path,
        requested_target_col=requested_target_col,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        evaluation_dataset_source=evaluation_dataset_source,
        logger=logger,
    )
    return LoadedEvaluationFrames("local_parquet", *loaded)


def _load_gold_evaluation_frames(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    model_backend: str,
) -> LoadedEvaluationFrames:
    loaded = load_gold_reference_mode_frames(
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        model_backend=model_backend,
    )
    evaluation_mode = (
        "bakery_reference_transfer_holdout"
        if uses_bakery_reference_transfer_holdout(model_backend)
        else "bakery_reference_overlap"
    )
    return LoadedEvaluationFrames(evaluation_mode, *loaded)


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
    evaluation_dataset_source: str | None = None,
    model_backend: str = "tft",
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
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
            evaluation_dataset_source=evaluation_dataset_source,
            logger=logger,
        )
    logger.info(
        "Running evaluation in bakery reference mode: duckdb_path=%s gold_table=%s train_sample_fraction=%.4f tuning_sample_fraction=%.4f bakery_test=unsampled_reference_split model_backend=%s reference_protocol=bakery_product_arima_equivalent_from_gold",
        duckdb_path,
        gold_table,
        train_sample_fraction,
        tuning_sample_fraction,
        model_backend,
    )
    return _load_gold_evaluation_frames(
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        model_backend=model_backend,
    )


def _resolve_feature_context(
    *,
    evaluation_mode: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_contract: TargetContract,
    model_backend: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    combined_pretest_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    feature_cols, constant_feature_cols, identifier_feature_cols = select_feature_columns(
        combined_pretest_frame,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        model_backend=model_backend,
    )
    if model_backend == "xgboost" and evaluation_mode in {
        "bakery_reference_overlap",
        "bakery_reference_transfer_holdout",
    }:
        feature_cols, constant_feature_cols, identifier_feature_cols = (
            _restrict_xgboost_feature_context_to_manifest(
                train_frame=combined_pretest_frame,
                feature_cols=feature_cols,
                constant_feature_cols=constant_feature_cols,
            )
        )
    missing_in_test_feature_cols = [column for column in feature_cols if column not in test_frame.columns]
    filtered_feature_cols = [column for column in feature_cols if column in test_frame.columns]
    return filtered_feature_cols, constant_feature_cols, identifier_feature_cols, missing_in_test_feature_cols


def _real_feature_columns(
    feature_cols: list[str],
) -> list[str]:
    real_roles = {
        "static_real",
        "time_varying_known_real",
        "time_varying_unknown_real",
    }
    return [
        column
        for column in feature_cols
        if TFT_EXPLICIT_ROLE_BY_COLUMN.get(column) in real_roles
    ]


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


def _training_manifest_feature_columns(
    feature_manifest_path: Path = DEFAULT_TRAINING_FEATURE_MANIFEST_PATH,
) -> list[str] | None:
    if not feature_manifest_path.exists():
        return None
    payload = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    raw_feature_columns = payload.get("feature_columns")
    if not isinstance(raw_feature_columns, list):
        return None
    feature_columns = cast(list[object], raw_feature_columns)
    return [str(column) for column in feature_columns]


def _expected_xgboost_feature_columns(
    *,
    train_frame: pd.DataFrame,
) -> list[str] | None:
    expected = _training_manifest_feature_columns()
    if expected is None:
        return None
    resolved = list(expected)
    for column in DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS:
        if column in train_frame.columns and column not in resolved:
            resolved.append(column)
    return resolved


def _restrict_xgboost_feature_context_to_manifest(
    *,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    constant_feature_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    expected = _expected_xgboost_feature_columns(train_frame=train_frame)
    if expected is None:
        identifier_feature_cols = [
            column
            for column in DEFAULT_IDENTIFIER_FEATURE_COLS
            if column in train_frame.columns and column not in feature_cols
        ]
        return feature_cols, constant_feature_cols, identifier_feature_cols
    missing = [column for column in expected if column not in train_frame.columns]
    if missing:
        raise ValueError(
            "XGBoost evaluation feature contract mismatch with the optimisation bundle "
            f"(expected_count={len(expected)} actual_count={len(feature_cols)} "
            f"missing={missing} extra=[])."
        )
    expected_set = set(expected)
    resolved_feature_cols = [column for column in expected if column in expected_set]
    resolved_constant_feature_cols = [
        column for column in constant_feature_cols if column in expected_set
    ]
    identifier_feature_cols = [
        column
        for column in DEFAULT_IDENTIFIER_FEATURE_COLS
        if column in train_frame.columns and column not in resolved_feature_cols
    ]
    return (
        resolved_feature_cols,
        resolved_constant_feature_cols,
        identifier_feature_cols,
    )


def _validate_xgboost_feature_contract(
    *,
    evaluation_mode: str,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    if evaluation_mode not in {
        "bakery_reference_overlap",
        "bakery_reference_transfer_holdout",
    }:
        return
    expected = _expected_xgboost_feature_columns(train_frame=train_frame)
    if expected is None or set(feature_cols) == set(expected):
        return
    missing = [column for column in expected if column not in feature_cols]
    extra = [column for column in feature_cols if column not in expected]
    raise ValueError(
        "XGBoost evaluation feature contract mismatch with the optimisation bundle "
        f"(expected_count={len(expected)} actual_count={len(feature_cols)} "
        f"missing={missing} extra={extra})."
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


def _assert_transfer_holdout_training_sources(
    *,
    evaluation_mode: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
) -> None:
    if evaluation_mode != "bakery_reference_transfer_holdout":
        return
    for split_name, frame in (("train", train_frame), ("valid", valid_frame)):
        if "dataset_source" not in frame.columns:
            continue
        bakery_mask = (
            frame["dataset_source"].astype(str) == DEFAULT_BAKERY_REFERENCE_DATASET_SOURCE
        )
        if not bool(bakery_mask.any()):
            continue
        if (
            "bakery_refit_seed_flag" in frame.columns
            and bool(frame.loc[bakery_mask, "bakery_refit_seed_flag"].fillna(False).astype(bool).all())
        ):
            continue
        raise ValueError(
            "Transfer-holdout evaluation cannot train on bakery before the "
            f"test window (split={split_name})."
        )


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


def prepare_evaluation_context(
    *,
    loaded: LoadedEvaluationFrames,
    requested_target_col: str,
    best_params_path: str | Path,
    logger: logging.Logger,
    model_backend: str = "tft",
) -> EvaluationPreparedContext:
    best_params = load_best_params(best_params_path)
    if model_backend == "xgboost" and best_params.get("model_backend") != "xgboost":
        logger.warning(
            "XGBoost evaluation best params do not declare model_backend=xgboost: path=%s. "
            "XGBoost defaults will fill missing native parameters; rerun XGBoost optimisation "
            "before treating this as the tuned evaluation.",
            best_params_path,
        )
    target_contract = _resolved_target_contract(loaded=loaded, requested_target_col=requested_target_col)
    train_frame, valid_frame, test_frame, scored_reference_test, dropped_test_rows = (
        _prepared_target_frames(
            loaded=loaded,
            target_contract=target_contract,
        )
    )
    _assert_transfer_holdout_training_sources(
        evaluation_mode=loaded.evaluation_mode,
        train_frame=train_frame,
        valid_frame=valid_frame,
    )
    feature_cols, constant_feature_cols, identifier_feature_cols, missing_in_test_feature_cols = (
        _resolve_feature_context(
            train_frame=train_frame,
            valid_frame=valid_frame,
            test_frame=test_frame,
            evaluation_mode=loaded.evaluation_mode,
            target_contract=target_contract,
            model_backend=model_backend,
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
    if model_backend == "xgboost":
        _validate_xgboost_feature_contract(
            evaluation_mode=loaded.evaluation_mode,
            train_frame=train_frame,
            feature_cols=feature_cols,
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
    return EvaluationPreparedContext(
        best_params=best_params,
        model_backend=model_backend,
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
