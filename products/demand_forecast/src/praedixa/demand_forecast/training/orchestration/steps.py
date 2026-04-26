from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_mapping import (
    select_explicit_tft_feature_columns,
)
from praedixa.demand_forecast.contracts.targets import (
    TargetContract,
    build_target_contract_metadata,
    ensure_learning_target_column,
    resolve_target_contract,
)
from praedixa.demand_forecast.training.baselines import (
    evaluate_statistical_baselines_macro,
    summarize_dataset_weights,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_REMOVED_MODEL_INPUT_COLS,
    DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS,
)
from praedixa.demand_forecast.training.validation.feature_audit import (
    build_feature_audit_payload,
    drop_constant_feature_columns,
)
from praedixa.demand_forecast.training.validation.folds import (
    build_grouped_tuning_walk_forward_folds_by_dataset,
    build_tuning_walk_forward_folds_by_dataset,
)
from praedixa.demand_forecast.training.sampling import (
    log_sampling_summary,
    resolve_sampling_store_col,
)
from praedixa.demand_forecast.training.sampling.loaders import (
    load_gold_train_tuning_frames,
    load_parquet_train_tuning_frames,
)


@dataclass
class LoadedOptimisationFrames:
    train_frame: pd.DataFrame
    tuning_frame: pd.DataFrame
    sample_store_col: str
    train_sampling_metadata: dict[str, object]
    tuning_sampling_metadata: dict[str, object]
    train_path: Path | None
    tuning_path: Path | None


@dataclass
class PreparedOptimisationContext:
    train_frame: pd.DataFrame
    tuning_frame: pd.DataFrame
    sample_store_col: str
    target_contract: TargetContract
    resolved_target_col: str
    absolute_target_col: str
    feature_cols: list[str]
    raw_feature_cols: list[str]
    constant_feature_cols: list[str]
    identifier_feature_cols: list[str]


@dataclass(frozen=True)
class PersistedOptimisationArtifacts:
    best_params: dict[str, object]
    tuning_report: pd.DataFrame
    best_baseline: dict[str, object]
    baseline_rows: list[dict[str, object]]
    train_dataset_weights: dict[str, dict[str, float | int]]
    tuning_dataset_weights: dict[str, dict[str, float | int]]
    hpo_runtime_metadata: dict[str, object]
    baseline_folds: list[dict[str, object]]
    training_folds: list[dict[str, object]]


@dataclass(frozen=True)
class OptimisationRunConfig:
    bundle_dir: str | Path | None
    model_backend: str
    date_col: str
    n_folds: int
    tuning_trials: int
    tuning_random_seed: int
    resolved_model_params: dict[str, object]
    resolved_target_transform: str
    duckdb_path: str | Path
    gold_table: str


def _json_dump(
    path: Path, payload: dict[str, object] | list[dict[str, object]]
) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_gold_frames(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    logger: logging.Logger,
    date_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    included_dataset_sources: tuple[str, ...] | None,
) -> LoadedOptimisationFrames:
    loaded = load_gold_train_tuning_frames(
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        logger=logger,
        date_col=date_col,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        included_dataset_sources=included_dataset_sources,
    )
    return LoadedOptimisationFrames(*loaded, None, None)


def _load_parquet_frames(
    *,
    train_input_path: str | Path,
    tuning_input_path: str | Path,
    logger: logging.Logger,
    date_col: str,
    target_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    included_dataset_sources: tuple[str, ...] | None,
) -> LoadedOptimisationFrames:
    loaded = load_parquet_train_tuning_frames(
        train_input_path=train_input_path,
        tuning_input_path=tuning_input_path,
        logger=logger,
        date_col=date_col,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        target_col=target_col,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        included_dataset_sources=included_dataset_sources,
    )
    return LoadedOptimisationFrames(
        *loaded, Path(train_input_path), Path(tuning_input_path)
    )


def load_optimisation_frames(
    *,
    train_input_path: str | Path | None,
    tuning_input_path: str | Path | None,
    duckdb_path: str | Path,
    gold_table: str,
    logger: logging.Logger,
    date_col: str,
    target_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    included_dataset_sources: tuple[str, ...] | None = None,
) -> LoadedOptimisationFrames:
    if train_input_path is None or tuning_input_path is None:
        logger.info(
            "No parquet inputs provided; loading optimisation frames from gold."
        )
        return _load_gold_frames(
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            logger=logger,
            date_col=date_col,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
            included_dataset_sources=included_dataset_sources,
        )
    logger.info(
        "Loading optimisation frames from parquet inputs: train=%s tuning=%s",
        train_input_path,
        tuning_input_path,
    )
    return _load_parquet_frames(
        train_input_path=train_input_path,
        tuning_input_path=tuning_input_path,
        logger=logger,
        date_col=date_col,
        target_col=target_col,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        included_dataset_sources=included_dataset_sources,
    )


def _normalize_frames(
    *, train_frame: pd.DataFrame, tuning_frame: pd.DataFrame, date_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame[date_col] = pd.to_datetime(train_frame[date_col])
    tuning_frame[date_col] = pd.to_datetime(tuning_frame[date_col])
    if DEFAULT_DATASET_SOURCE_COL not in train_frame.columns:
        raise ValueError(f"Training frame is missing `{DEFAULT_DATASET_SOURCE_COL}`.")
    if DEFAULT_DATASET_SOURCE_COL not in tuning_frame.columns:
        raise ValueError(f"Tuning frame is missing `{DEFAULT_DATASET_SOURCE_COL}`.")
    train_frame = _sanitize_model_input_frame(train_frame)
    tuning_frame = _sanitize_model_input_frame(tuning_frame)
    return train_frame.sort_values(date_col).reset_index(
        drop=True
    ), tuning_frame.sort_values(date_col).reset_index(drop=True)


def _sanitize_model_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy()
    removed_columns = [
        column
        for column in DEFAULT_REMOVED_MODEL_INPUT_COLS
        if column in sanitized.columns
    ]
    if removed_columns:
        sanitized = sanitized.drop(columns=removed_columns)
    return sanitized


def _resolve_target_context(
    *,
    loaded: LoadedOptimisationFrames,
    target_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, str, TargetContract, str, str]:
    target_contract = resolve_target_contract(
        loaded.train_frame, loaded.tuning_frame, requested_target_col=target_col
    )
    train_frame = ensure_learning_target_column(loaded.train_frame, target_contract)
    tuning_frame = ensure_learning_target_column(loaded.tuning_frame, target_contract)
    resolved_target_col = target_contract.learning_target_col
    dropped_train_rows = int(train_frame[resolved_target_col].isna().sum())
    dropped_tuning_rows = int(tuning_frame[resolved_target_col].isna().sum())
    if dropped_train_rows or dropped_tuning_rows:
        logger.info(
            "Dropping rows without a usable learning target: train_rows_dropped=%s tuning_rows_dropped=%s target_col=%s",
            dropped_train_rows,
            dropped_tuning_rows,
            resolved_target_col,
        )
        train_frame = train_frame[train_frame[resolved_target_col].notna()].copy()
        tuning_frame = tuning_frame[tuning_frame[resolved_target_col].notna()].copy()
    sample_store_col = (
        resolve_sampling_store_col(train_frame)
        if loaded.train_path and loaded.tuning_path
        else loaded.sample_store_col
    )
    return (
        train_frame,
        tuning_frame,
        sample_store_col,
        target_contract,
        resolved_target_col,
        target_contract.absolute_target_col,
    )


def _resolve_feature_context(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    date_col: str,
    resolved_target_col: str,
    absolute_target_col: str,
    model_backend: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    identifier_exclusions = _identifier_feature_exclusions(model_backend)
    raw_feature_cols = select_explicit_tft_feature_columns(
        train_frame,
        excluded_cols={
            date_col,
            resolved_target_col,
            absolute_target_col,
            *identifier_exclusions,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
            *DEFAULT_REMOVED_MODEL_INPUT_COLS,
        },
    )
    raw_feature_cols = _with_xgboost_identifier_features(
        raw_feature_cols,
        train_frame=train_frame,
        model_backend=model_backend,
    )
    feature_cols, constant_feature_cols = drop_constant_feature_columns(
        train_frame,
        raw_feature_cols,
        tuning_frame=tuning_frame,
    )
    identifier_feature_cols = [
        col
        for col in DEFAULT_IDENTIFIER_FEATURE_COLS
        if col in train_frame.columns and col not in feature_cols
    ]
    return (
        feature_cols,
        raw_feature_cols,
        constant_feature_cols,
        identifier_feature_cols,
    )


def _identifier_feature_exclusions(model_backend: str) -> tuple[str, ...]:
    if model_backend == "xgboost":
        xgboost_identifiers = set(DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS)
        return tuple(
            column
            for column in DEFAULT_IDENTIFIER_FEATURE_COLS
            if column not in xgboost_identifiers
        )
    return DEFAULT_IDENTIFIER_FEATURE_COLS


def _with_xgboost_identifier_features(
    feature_cols: list[str],
    *,
    train_frame: pd.DataFrame,
    model_backend: str,
) -> list[str]:
    if model_backend != "xgboost":
        return feature_cols
    resolved = list(feature_cols)
    for column in DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS:
        if column in train_frame.columns and column not in resolved:
            resolved.append(column)
    return resolved


def prepare_optimisation_context(
    *,
    loaded: LoadedOptimisationFrames,
    date_col: str,
    target_col: str,
    model_backend: str,
    logger: logging.Logger,
) -> PreparedOptimisationContext:
    train_frame, tuning_frame = _normalize_frames(
        train_frame=loaded.train_frame,
        tuning_frame=loaded.tuning_frame,
        date_col=date_col,
    )
    normalized_loaded = LoadedOptimisationFrames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        sample_store_col=loaded.sample_store_col,
        train_sampling_metadata=loaded.train_sampling_metadata,
        tuning_sampling_metadata=loaded.tuning_sampling_metadata,
        train_path=loaded.train_path,
        tuning_path=loaded.tuning_path,
    )
    (
        train_frame,
        tuning_frame,
        sample_store_col,
        target_contract,
        resolved_target_col,
        absolute_target_col,
    ) = _resolve_target_context(
        loaded=normalized_loaded,
        target_col=target_col,
        logger=logger,
    )
    feature_cols, raw_feature_cols, constant_feature_cols, identifier_feature_cols = (
        _resolve_feature_context(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            date_col=date_col,
            resolved_target_col=resolved_target_col,
            absolute_target_col=absolute_target_col,
            model_backend=model_backend,
        )
    )
    return PreparedOptimisationContext(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        sample_store_col=sample_store_col,
        target_contract=target_contract,
        resolved_target_col=resolved_target_col,
        absolute_target_col=absolute_target_col,
        feature_cols=feature_cols,
        raw_feature_cols=raw_feature_cols,
        constant_feature_cols=constant_feature_cols,
        identifier_feature_cols=identifier_feature_cols,
    )


def _dataset_weight_summaries(
    context: PreparedOptimisationContext,
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, float | int]]]:
    return (
        summarize_dataset_weights(
            context.train_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL
        ),
        summarize_dataset_weights(
            context.tuning_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL
        ),
    )


def log_optimisation_context(
    *,
    loaded: LoadedOptimisationFrames,
    context: PreparedOptimisationContext,
    train_dataset_weights: dict[str, dict[str, float | int]],
    tuning_dataset_weights: dict[str, dict[str, float | int]],
    logger: logging.Logger,
) -> None:
    log_sampling_summary("Train", loaded.train_sampling_metadata, logger=logger)
    log_sampling_summary("Validation", loaded.tuning_sampling_metadata, logger=logger)
    logger.info(
        "Loaded optimisation datasets: train_rows=%s tuning_rows=%s feature_count=%s learning_target_col=%s absolute_target_col=%s sample_store_col=%s",
        len(context.train_frame),
        len(context.tuning_frame),
        len(context.feature_cols),
        context.resolved_target_col,
        context.absolute_target_col,
        context.sample_store_col,
    )
    logger.info(
        "Target contract for optimisation: %s",
        build_target_contract_metadata(context.target_contract),
    )
    logger.info(
        "Identifier features included in optimisation: %s",
        context.identifier_feature_cols,
    )
    logger.info(
        "Constant features dropped before optimisation: %s",
        context.constant_feature_cols,
    )
    logger.info("Train dataset weights summary: %s", train_dataset_weights)
    logger.info("Validation dataset weights summary: %s", tuning_dataset_weights)


def _metadata_payload(
    *,
    loaded: LoadedOptimisationFrames,
    context: PreparedOptimisationContext,
    run_config: OptimisationRunConfig,
    artifacts: PersistedOptimisationArtifacts,
) -> dict[str, object]:
    fold_payload = [
        {
            key: value
            for key, value in fold.items()
            if key not in {"train_idx", "valid_idx"}
        }
        for fold in artifacts.baseline_folds
    ]
    training_fold_payload = [
        {
            key: value
            for key, value in fold.items()
            if key not in {"train_idx", "valid_idx"}
        }
        for fold in artifacts.training_folds
    ]
    bundle_dir = None if run_config.bundle_dir is None else Path(run_config.bundle_dir)
    return {
        "train_input_path": str(loaded.train_path) if loaded.train_path else None,
        "tuning_input_path": str(loaded.tuning_path) if loaded.tuning_path else None,
        "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
        "bundle_manifest_path": str(bundle_dir / "bundle_manifest.json")
        if bundle_dir is not None
        else None,
        "bundle_feature_manifest_path": str(bundle_dir / "feature_manifest.json")
        if bundle_dir is not None
        else None,
        "bundle_feature_roles_path": str(bundle_dir / "feature_roles.json")
        if bundle_dir is not None
        else None,
        "bundle_split_manifest_path": str(bundle_dir / "split_manifest.json")
        if bundle_dir is not None
        else None,
        "bundle_target_contract_path": str(bundle_dir / "target_contract.json")
        if bundle_dir is not None
        else None,
        "bundle_optimisation_manifest_path": str(
            bundle_dir / "optimisation_manifest.json"
        )
        if bundle_dir is not None
        else None,
        "model_backend": run_config.model_backend,
        "duckdb_path": str(run_config.duckdb_path)
        if loaded.train_path is None
        else None,
        "gold_table": run_config.gold_table if loaded.train_path is None else None,
        "n_folds": run_config.n_folds,
        "tuning_trials": run_config.tuning_trials,
        "tuning_random_seed": run_config.tuning_random_seed,
        **build_target_contract_metadata(context.target_contract),
        "sample_store_col": context.sample_store_col,
        "identifier_feature_cols": context.identifier_feature_cols,
        "constant_feature_cols": context.constant_feature_cols,
        "excluded_risky_feature_cols": list(DEFAULT_EXCLUDED_RISKY_FEATURE_COLS),
        "train_sampling": loaded.train_sampling_metadata,
        "tuning_sampling": loaded.tuning_sampling_metadata,
        "train_dataset_weights": artifacts.train_dataset_weights,
        "tuning_dataset_weights": artifacts.tuning_dataset_weights,
        "hpo_runtime": artifacts.hpo_runtime_metadata,
        "feature_count": len(context.feature_cols),
        "train_rows": int(len(context.train_frame)),
        "tuning_rows": int(len(context.tuning_frame)),
        "folds": fold_payload,
        "training_folds": training_fold_payload,
    }


def _build_output_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "best_params": target_dir / "best_optuna_params.json",
        "optuna_trials_report": target_dir / "optuna_trials_report.csv",
        "baseline_report": target_dir / "baseline_report.json",
        "optimisation_metadata": target_dir / "optimisation_metadata.json",
        "optimisation_feature_audit": target_dir / "optimisation_feature_audit.json",
    }


def _persist_outputs(
    *,
    target_dir: Path,
    loaded: LoadedOptimisationFrames,
    context: PreparedOptimisationContext,
    run_config: OptimisationRunConfig,
    artifacts: PersistedOptimisationArtifacts,
) -> dict[str, Path]:
    output_paths = _build_output_paths(target_dir)
    _json_dump(output_paths["best_params"], artifacts.best_params)
    artifacts.tuning_report.to_csv(output_paths["optuna_trials_report"], index=False)
    _json_dump(
        output_paths["baseline_report"],
        {
            "best_baseline": artifacts.best_baseline,
            "all_baselines": artifacts.baseline_rows,
        },
    )
    _json_dump(
        output_paths["optimisation_feature_audit"],
        build_feature_audit_payload(
            train_frame=context.train_frame,
            tuning_frame=context.tuning_frame,
            feature_cols=context.feature_cols,
            raw_feature_cols=context.raw_feature_cols,
            constant_feature_cols=context.constant_feature_cols,
            identifier_feature_cols=context.identifier_feature_cols,
            folds=artifacts.baseline_folds,
            target_contract=context.target_contract,
        ),
    )
    _json_dump(
        output_paths["optimisation_metadata"],
        _metadata_payload(
            loaded=loaded,
            context=context,
            run_config=run_config,
            artifacts=artifacts,
        ),
    )
    return output_paths


def _baseline_and_hpo_outputs(
    *,
    context: PreparedOptimisationContext,
    date_col: str,
    n_folds: int,
    tuning_trials: int,
    tuning_random_seed: int,
    resolved_model_params: dict[str, object],
    resolved_target_transform: str,
    model_backend: str,
    logger: logging.Logger,
    optimize_fn: Callable[
        ..., tuple[dict[str, object], pd.DataFrame, dict[str, object]]
    ],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    dict[str, object],
]:
    baseline_folds = build_tuning_walk_forward_folds_by_dataset(
        context.tuning_frame,
        date_col=date_col,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        n_folds=n_folds,
        logger=logger,
    )
    training_folds = build_grouped_tuning_walk_forward_folds_by_dataset(
        context.tuning_frame,
        date_col=date_col,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        n_folds=n_folds,
        logger=logger,
    )
    logger.info(
        "Starting statistical baseline sweep before Optuna: baselines=%s fold_evaluations=%s tuning_rows=%s",
        5,
        len(baseline_folds),
        len(context.tuning_frame),
    )
    baseline_rows, best_baseline = evaluate_statistical_baselines_macro(
        context.tuning_frame,
        baseline_folds,
        absolute_target_col=context.absolute_target_col,
        target_contract=context.target_contract,
        logger=logger,
    )
    logger.info(
        "Best statistical baseline selected under dataset-macro WAPE: name=%s mean_wape=%.6f",
        best_baseline["baseline_name"],
        best_baseline["mean_wape"],
    )
    logger.info(
        "Starting model Optuna search after baseline sweep: backend=%s grouped_folds=%s train_rows=%s tuning_rows=%s",
        model_backend,
        len(training_folds),
        len(context.train_frame),
        len(context.tuning_frame),
    )
    best_params, tuning_report, hpo_runtime_metadata = optimize_fn(
        train_frame=context.train_frame,
        tuning_frame=context.tuning_frame,
        folds=training_folds,
        feature_cols=context.feature_cols,
        baseline_wape=float(cast(float, best_baseline["mean_wape"])),
        baseline_dataset_wape=cast(
            dict[str, float], best_baseline["dataset_mean_wape"]
        ),
        target_contract=context.target_contract,
        logger=logger,
        tuning_trials=tuning_trials,
        random_seed=tuning_random_seed,
        model_params=resolved_model_params,
        target_transform=resolved_target_transform,
    )
    return (
        baseline_folds,
        training_folds,
        baseline_rows,
        best_baseline,
        best_params,
        tuning_report,
        hpo_runtime_metadata,
    )


def run_optimisation_pipeline(
    *,
    target_dir: Path,
    loaded: LoadedOptimisationFrames,
    context: PreparedOptimisationContext,
    run_config: OptimisationRunConfig,
    logger: logging.Logger,
    optimize_fn: Callable[
        ..., tuple[dict[str, object], pd.DataFrame, dict[str, object]]
    ],
) -> dict[str, Path]:
    train_dataset_weights, tuning_dataset_weights = _dataset_weight_summaries(context)
    log_optimisation_context(
        loaded=loaded,
        context=context,
        train_dataset_weights=train_dataset_weights,
        tuning_dataset_weights=tuning_dataset_weights,
        logger=logger,
    )
    (
        baseline_folds,
        training_folds,
        baseline_rows,
        best_baseline,
        best_params,
        tuning_report,
        hpo_runtime_metadata,
    ) = _baseline_and_hpo_outputs(
        context=context,
        date_col=run_config.date_col,
        n_folds=run_config.n_folds,
        tuning_trials=run_config.tuning_trials,
        tuning_random_seed=run_config.tuning_random_seed,
        resolved_model_params=run_config.resolved_model_params,
        resolved_target_transform=run_config.resolved_target_transform,
        model_backend=run_config.model_backend,
        logger=logger,
        optimize_fn=optimize_fn,
    )
    artifacts = PersistedOptimisationArtifacts(
        best_params=best_params,
        tuning_report=tuning_report,
        best_baseline=best_baseline,
        baseline_rows=baseline_rows,
        train_dataset_weights=train_dataset_weights,
        tuning_dataset_weights=tuning_dataset_weights,
        hpo_runtime_metadata=hpo_runtime_metadata,
        baseline_folds=baseline_folds,
        training_folds=training_folds,
    )
    output_paths = _persist_outputs(
        target_dir=target_dir,
        loaded=loaded,
        context=context,
        run_config=run_config,
        artifacts=artifacts,
    )
    logger.info(
        "Optimisation artifacts written: best_params=%s trials_report=%s",
        output_paths["best_params"],
        output_paths["optuna_trials_report"],
    )
    return output_paths
