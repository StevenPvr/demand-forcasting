"""ElasticNet feature selection for pre-built training bundles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import cast

import pandas as pd

from praedixa.demand_forecast.feature_selection.artifacts import (
    FEATURE_MANIFEST,
    OPTIMISATION_TRAIN_SPLIT,
    OPTIMISATION_TUNING_SPLIT,
    OPTIMISATION_VALID_SPLIT,
    TARGET_CONTRACT,
    TRAIN_SPLIT,
    TUNING_SPLIT,
    VALID_SPLIT,
    copy_if_exists,
    feature_columns,
    feature_manifest,
    feature_roles,
    load_bundle_frame,
    selected_bundle_manifest,
    target_column,
    write_filtered_split,
    write_json,
    write_selected_feature_manifest,
    write_selected_feature_roles,
    write_selected_optimisation_manifest,
)
from praedixa.demand_forecast.feature_selection.constants import (
    CORRELATION_THRESHOLD,
    DEFAULT_FOLD_WORKERS,
    DEFAULT_FOLDS,
    DEFAULT_OPTUNA_TRIALS,
    MAX_SELECTED_MODEL_FEATURES,
)
from praedixa.demand_forecast.feature_selection.paths import (
    DEFAULT_BUNDLE_INPUT_DIR,
    DEFAULT_FEATURE_SELECTION_DIR,
)
from praedixa.demand_forecast.feature_selection.redundancy import (
    drop_linear_correlated_candidate_features,
    drop_nonlinear_correlated_candidate_features,
)
from praedixa.demand_forecast.feature_selection.selector import (
    drop_constant_candidate_features,
    importance_frame,
    selector_candidate_columns,
    selected_numeric_features,
    tune_selector_params,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Paths and counts produced by feature selection."""

    output_dir: Path
    selected_feature_count: int
    selected_numeric_feature_count: int
    passthrough_feature_count: int
    train_rows: int
    tuning_rows: int
    valid_rows: int


@dataclass(frozen=True)
class FeatureSelectionSummary:
    """JSON-serializable feature selection run summary."""

    selector_params: dict[str, object]
    target_column: str
    original_features: list[str]
    numeric_candidates: list[str]
    categorical_candidates: list[str]
    candidate_features: list[str]
    tuning_candidate_features: list[str]
    tuning_excluded_lag_features: list[str]
    passthrough_features: list[str]
    max_selected_model_features: int
    dropped_constant: list[str]
    dropped_linear_correlated: list[str]
    dropped_nonlinear_correlated: list[str]
    selected_numeric: list[str]
    selected_categorical: list[str]
    selected_features: list[str]
    train_mae: float
    tuning_mae: float | None
    n_trials: int
    n_folds: int
    n_fold_workers: int
    elapsed_seconds: float

    def to_json_payload(self) -> dict[str, object]:
        return {
            "selector_model_family": "ElasticNet",
            "selection_data_scope": "train_split_only",
            "selector_params": self.selector_params,
            "target_column": self.target_column,
            "original_feature_count": len(self.original_features),
            "numeric_candidate_features": self.numeric_candidates,
            "categorical_candidate_features": self.categorical_candidates,
            "candidate_feature_columns": self.candidate_features,
            "tuning_candidate_feature_columns": self.tuning_candidate_features,
            "tuning_excluded_lag_feature_columns": self.tuning_excluded_lag_features,
            "passthrough_feature_columns": self.passthrough_features,
            "max_selected_model_features": self.max_selected_model_features,
            "dropped_constant_features": self.dropped_constant,
            "dropped_linear_correlated_features": self.dropped_linear_correlated,
            "dropped_nonlinear_correlated_features": self.dropped_nonlinear_correlated,
            "selected_numeric_feature_columns": self.selected_numeric,
            "selected_categorical_feature_columns": self.selected_categorical,
            "selected_feature_columns": self.selected_features,
            "selected_feature_count": len(self.selected_features),
            "selector_train_mae": self.train_mae,
            "selector_tuning_mae": self.tuning_mae,
            "n_trials": self.n_trials,
            "n_folds": self.n_folds,
            "n_fold_workers": self.n_fold_workers,
            "elapsed_seconds": self.elapsed_seconds,
        }


def run_feature_selection(
    *,
    bundle_dir: str | Path = DEFAULT_BUNDLE_INPUT_DIR,
    output_dir: str | Path = DEFAULT_FEATURE_SELECTION_DIR,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    n_fold_workers: int = DEFAULT_FOLD_WORKERS,
) -> FeatureSelectionResult:
    """Select model features from a training bundle and write a new bundle."""

    start_time = time.perf_counter()
    resolved_bundle_dir = Path(bundle_dir)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_bundle_frame(resolved_bundle_dir / TRAIN_SPLIT)
    manifest = feature_manifest(resolved_bundle_dir)
    target = target_column(resolved_bundle_dir)
    original_features = feature_columns(manifest)
    roles = feature_roles(manifest)
    passthrough_features = _passthrough_feature_columns(
        manifest,
        original_features=original_features,
        feature_roles=roles,
    )
    selectable_features = [
        column
        for column in original_features
        if column not in set(passthrough_features)
    ]

    numeric_candidates, categorical_candidates = selector_candidate_columns(
        train_df,
        feature_columns=selectable_features,
        feature_roles=roles,
    )
    selector_candidates = [*numeric_candidates, *categorical_candidates]
    non_constant_candidates, dropped_constant = drop_constant_candidate_features(
        train_df,
        selector_candidates,
    )
    non_constant_numeric = [
        column for column in non_constant_candidates if column in numeric_candidates
    ]
    linear_candidate_features, dropped_linear_correlated = (
        drop_linear_correlated_candidate_features(
            train_df,
            non_constant_numeric,
            target,
            CORRELATION_THRESHOLD,
        )
    )
    numeric_candidate_features, dropped_nonlinear_correlated = (
        drop_nonlinear_correlated_candidate_features(
            train_df,
            linear_candidate_features,
            target,
            CORRELATION_THRESHOLD,
        )
    )
    candidate_features = numeric_candidate_features
    tuning_numeric_features = _non_lag_features(numeric_candidate_features)
    if not tuning_numeric_features:
        tuning_numeric_features = numeric_candidate_features
    tuning_candidate_features = tuning_numeric_features
    tuning_excluded_lag_features = [
        column
        for column in candidate_features
        if column not in set(tuning_candidate_features)
    ]
    selector_params, trials_df = tune_selector_params(
        train_df,
        numeric_feature_columns=tuning_numeric_features,
        target_column=target,
        n_trials=n_trials,
        n_folds=n_folds,
        n_fold_workers=n_fold_workers,
    )
    importance_df, train_mae, tuning_mae = importance_frame(
        train_df,
        numeric_feature_columns=numeric_candidate_features,
        target_column=target,
        selector_params=selector_params,
    )
    selected_model_features = selected_numeric_features(importance_df)
    selected_model_features = _limit_selected_model_features(
        importance_df,
        selected_model_features,
        max_selected_features=MAX_SELECTED_MODEL_FEATURES,
    )
    selected_numeric = [
        column for column in selected_model_features if column in numeric_candidates
    ]
    selected_categorical: list[str] = []
    selected_features = [
        column
        for column in original_features
        if column in set(selected_model_features) or column in set(passthrough_features)
    ]

    train_rows, tuning_rows, valid_rows = _write_selected_splits(
        bundle_dir=resolved_bundle_dir,
        output_dir=resolved_output_dir,
        original_features=original_features,
        selected_features=selected_features,
    )
    _write_selected_contracts(
        bundle_dir=resolved_bundle_dir,
        output_dir=resolved_output_dir,
        input_manifest=manifest,
        selected_features=selected_features,
        include_valid=valid_rows > 0,
    )
    _copy_passthrough_metadata(resolved_bundle_dir, resolved_output_dir)

    importance_path = resolved_output_dir / "feature_selection_importances.csv"
    trials_path = resolved_output_dir / "feature_selection_optuna_trials.csv"
    summary_path = resolved_output_dir / "feature_selection_summary.json"
    importance_df.to_csv(importance_path, index=False)
    trials_df.to_csv(trials_path, index=False)
    _write_summary(
        summary_path,
        FeatureSelectionSummary(
            selector_params=selector_params,
            target_column=target,
            original_features=original_features,
            numeric_candidates=numeric_candidates,
            categorical_candidates=categorical_candidates,
            candidate_features=candidate_features,
            tuning_candidate_features=tuning_candidate_features,
            tuning_excluded_lag_features=tuning_excluded_lag_features,
            passthrough_features=passthrough_features,
            max_selected_model_features=MAX_SELECTED_MODEL_FEATURES,
            dropped_constant=dropped_constant,
            dropped_linear_correlated=dropped_linear_correlated,
            dropped_nonlinear_correlated=dropped_nonlinear_correlated,
            selected_numeric=selected_numeric,
            selected_categorical=selected_categorical,
            selected_features=selected_features,
            train_mae=train_mae,
            tuning_mae=tuning_mae,
            n_trials=n_trials,
            n_folds=n_folds,
            n_fold_workers=n_fold_workers,
            elapsed_seconds=time.perf_counter() - start_time,
        ),
    )
    write_json(
        resolved_output_dir / "bundle_manifest.json",
        selected_bundle_manifest(
            source_bundle_dir=resolved_bundle_dir,
            output_dir=resolved_output_dir,
            train_rows=train_rows,
            tuning_rows=tuning_rows,
            valid_rows=valid_rows,
            selected_feature_count=len(selected_features),
            summary_path=summary_path,
            importance_path=importance_path,
            trials_path=trials_path,
        ),
    )
    LOGGER.info(
        "Selected %d/%d bundle features in %.3f seconds",
        len(selected_features),
        len(original_features),
        time.perf_counter() - start_time,
    )
    return FeatureSelectionResult(
        output_dir=resolved_output_dir,
        selected_feature_count=len(selected_features),
        selected_numeric_feature_count=len(selected_numeric),
        passthrough_feature_count=len(passthrough_features),
        train_rows=train_rows,
        tuning_rows=tuning_rows,
        valid_rows=valid_rows,
    )


def _non_lag_features(columns: list[str]) -> list[str]:
    return [column for column in columns if not _is_lag_feature(column)]


def _is_lag_feature(column: str) -> bool:
    return column.startswith("lag_") or "_lag_" in column


def _limit_selected_model_features(
    importance_df: pd.DataFrame,
    selected_model_features: list[str],
    *,
    max_selected_features: int,
) -> list[str]:
    if len(selected_model_features) <= max_selected_features:
        return selected_model_features
    selected_set = set(selected_model_features)
    ranked = importance_df.loc[
        importance_df["feature"].isin(selected_set),
        ["feature", "importance"],
    ].sort_values(["importance", "feature"], ascending=[False, True])
    return [str(value) for value in ranked["feature"].head(max_selected_features)]


def _passthrough_feature_columns(
    manifest: dict[str, object],
    *,
    original_features: list[str],
    feature_roles: dict[str, str],
) -> list[str]:
    return [
        column
        for column in original_features
        if _is_model_identity_feature(column, manifest, feature_roles)
    ]


def _is_model_identity_feature(
    column: str,
    manifest: dict[str, object],
    feature_roles: dict[str, str],
) -> bool:
    if feature_roles.get(column) in {"static_categorical", "group_id"}:
        return True
    contract_raw = manifest.get("feature_contract", {})
    if not isinstance(contract_raw, Mapping):
        return False
    contract = cast(Mapping[str, object], contract_raw)
    entry = contract.get(column)
    if not isinstance(entry, Mapping):
        return False
    typed_entry = cast(Mapping[str, object], entry)
    source_system = typed_entry.get("source_system")
    role = typed_entry.get("role")
    return source_system in {"metadata", "pipeline_metadata"} or role in {
        "static_categorical",
        "group_id",
    }


def _write_selected_splits(
    *,
    bundle_dir: Path,
    output_dir: Path,
    original_features: list[str],
    selected_features: list[str],
) -> tuple[int, int, int]:
    train_rows = _write_split(
        bundle_dir / TRAIN_SPLIT,
        output_dir / TRAIN_SPLIT,
        original_features,
        selected_features,
    )
    tuning_rows = _write_split(
        bundle_dir / TUNING_SPLIT,
        output_dir / TUNING_SPLIT,
        original_features,
        selected_features,
    )
    valid_rows = _write_split(
        bundle_dir / VALID_SPLIT,
        output_dir / VALID_SPLIT,
        original_features,
        selected_features,
    )
    for split_name in (
        OPTIMISATION_TRAIN_SPLIT,
        OPTIMISATION_TUNING_SPLIT,
        OPTIMISATION_VALID_SPLIT,
    ):
        _write_split(
            bundle_dir / split_name,
            output_dir / split_name,
            original_features,
            selected_features,
        )
    return train_rows, tuning_rows, valid_rows


def _write_split(
    input_path: Path,
    output_path: Path,
    original_features: list[str],
    selected_features: list[str],
) -> int:
    return write_filtered_split(
        input_path=input_path,
        output_path=output_path,
        original_feature_columns=original_features,
        selected_feature_columns=selected_features,
    )


def _write_selected_contracts(
    *,
    bundle_dir: Path,
    output_dir: Path,
    input_manifest: dict[str, object],
    selected_features: list[str],
    include_valid: bool,
) -> None:
    write_selected_feature_manifest(
        input_manifest=input_manifest,
        output_path=output_dir / FEATURE_MANIFEST,
        selected_feature_columns=selected_features,
    )
    selected_manifest = feature_manifest(output_dir)
    write_selected_feature_roles(
        output_path=output_dir / "feature_roles.json",
        selected_feature_columns=selected_features,
    )
    write_selected_optimisation_manifest(
        input_manifest_path=bundle_dir / "optimisation_manifest.json",
        output_path=output_dir / "optimisation_manifest.json",
        selected_feature_manifest=selected_manifest,
        selected_feature_columns=selected_features,
        include_valid=include_valid,
    )


def _copy_passthrough_metadata(bundle_dir: Path, output_dir: Path) -> None:
    for manifest_name in (
        TARGET_CONTRACT,
        "split_manifest.json",
        "training_exclusion_report.json",
    ):
        copy_if_exists(bundle_dir / manifest_name, output_dir / manifest_name)


def _write_summary(path: Path, summary: FeatureSelectionSummary) -> None:
    write_json(path, summary.to_json_payload())
