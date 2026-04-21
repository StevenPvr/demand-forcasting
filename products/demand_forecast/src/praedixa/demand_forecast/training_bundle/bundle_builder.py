from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_mapping import TFT_EXPLICIT_ROLE_BY_COLUMN
from praedixa.demand_forecast.backends.tft.feature_mapping import select_explicit_tft_group_id_columns
from praedixa.demand_forecast.backends.tft.model_utils import select_tft_feature_columns
from praedixa.demand_forecast.contracts.targets import (
    DEFAULT_VARIATION_TARGET_COL,
    TargetContract,
    build_target_contract_metadata,
    ensure_learning_target_column,
    resolve_target_contract,
)
from praedixa.demand_forecast.training.constants import (
    DEFAULT_DATE_COL,
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
)
from praedixa.platform.runtime.paths import FEATURE_SELECTION_DIR, TRAINING_BUNDLE_DIR
from praedixa.demand_forecast.training_bundle.manifest import sha256_file


DEFAULT_TRAIN_INPUT_PATH = FEATURE_SELECTION_DIR / "train_selection_70_selected.parquet"
DEFAULT_TUNING_INPUT_PATH = FEATURE_SELECTION_DIR / "train_tuning_30_selected.parquet"
DEFAULT_VALID_INPUT_PATH: Path | None = None
DEFAULT_OUTPUT_DIR = TRAINING_BUNDLE_DIR


def _json_dump(path: Path, payload: Mapping[str, object | None]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame[DEFAULT_DATE_COL] = pd.to_datetime(frame[DEFAULT_DATE_COL])
    return frame.sort_values(DEFAULT_DATE_COL).reset_index(drop=True)


def _bundle_output_paths(output_dir: Path, *, include_valid: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "train": output_dir / "train.parquet",
        "tuning": output_dir / "tuning.parquet",
        "feature_manifest": output_dir / "feature_manifest.json",
        "target_contract": output_dir / "target_contract.json",
        "bundle_manifest": output_dir / "bundle_manifest.json",
    }
    if include_valid:
        paths["valid"] = output_dir / "valid.parquet"
    return paths


def _bundle_projection_columns(
    feature_cols: list[str],
    *,
    group_id_cols: list[str],
    learning_target_col: str,
    absolute_target_col: str,
    reconstruction_anchor_col: str | None,
    available_columns: set[str],
) -> list[str]:
    ordered: list[str] = [DEFAULT_DATE_COL]
    for column in [*group_id_cols, absolute_target_col, learning_target_col, reconstruction_anchor_col, *feature_cols]:
        if column is None or column not in available_columns or column in ordered:
            continue
        ordered.append(column)
    return ordered


def _optimize_projection_frame(
    projected: pd.DataFrame,
    *,
    feature_cols: list[str],
    group_id_columns: list[str],
    learning_target_col: str,
    absolute_target_col: str,
    reconstruction_anchor_col: str | None,
) -> pd.DataFrame:
    float32_columns = {
        column
        for column in [learning_target_col, absolute_target_col, reconstruction_anchor_col, *feature_cols]
        if column is not None
        and (
            column in {learning_target_col, absolute_target_col, reconstruction_anchor_col}
            or TFT_EXPLICIT_ROLE_BY_COLUMN.get(column, "").endswith("_real")
        )
    }
    categorical_columns = set(group_id_columns).union(
        {
            column
            for column in feature_cols
            if not TFT_EXPLICIT_ROLE_BY_COLUMN.get(column, "").endswith("_real")
        }
    )
    optimized = projected.copy()
    for column in optimized.columns:
        if column == DEFAULT_DATE_COL:
            continue
        if column in float32_columns:
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype("float32")
            continue
        if pd.api.types.is_bool_dtype(optimized[column]):
            optimized[column] = optimized[column].astype("int8")
            continue
        if column.endswith("_status") and pd.api.types.is_numeric_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype("Int8")
            continue
        if column in categorical_columns:
            optimized[column] = optimized[column].astype("string")
    return optimized


def _write_projected_frame(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    output_path: Path,
    feature_cols: list[str],
    group_id_columns: list[str],
    learning_target_col: str,
    absolute_target_col: str,
    reconstruction_anchor_col: str | None,
) -> int:
    projected = frame.loc[:, columns].copy().sort_values(DEFAULT_DATE_COL).reset_index(drop=True)
    projected = _optimize_projection_frame(
        projected,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        learning_target_col=learning_target_col,
        absolute_target_col=absolute_target_col,
        reconstruction_anchor_col=reconstruction_anchor_col,
    )
    projected.to_parquet(output_path, index=False)
    return int(len(projected))


def _feature_manifest_payload(
    feature_cols: list[str],
    *,
    group_id_columns: list[str],
    projection_columns: list[str],
    projection_dtypes: dict[str, str],
) -> dict[str, object]:
    return {
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "group_id_columns": group_id_columns,
        "projection_columns": projection_columns,
        "projection_dtypes": projection_dtypes,
        "feature_roles": {
            column: TFT_EXPLICIT_ROLE_BY_COLUMN[column]
            for column in feature_cols
        },
    }


def _load_bundle_frames(
    *,
    train_input_path: str | Path,
    tuning_input_path: str | Path,
    valid_input_path: str | Path | None,
) -> tuple[Path, Path, Path | None, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_path = Path(train_input_path)
    tuning_path = Path(tuning_input_path)
    valid_path = None if valid_input_path is None else Path(valid_input_path)
    train_frame = _read_frame(train_path)
    tuning_frame = _read_frame(tuning_path)
    valid_frame = None if valid_path is None else _read_frame(valid_path)
    return train_path, tuning_path, valid_path, train_frame, tuning_frame, valid_frame


def _resolve_bundle_contract(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, TargetContract]:
    target_contract = resolve_target_contract(train_frame, tuning_frame, requested_target_col=target_col)
    train_frame = ensure_learning_target_column(train_frame, target_contract)
    tuning_frame = ensure_learning_target_column(tuning_frame, target_contract)
    if valid_frame is not None:
        valid_frame = ensure_learning_target_column(valid_frame, target_contract)
    return train_frame, tuning_frame, valid_frame, target_contract


def _resolve_bundle_features(
    *,
    train_frame: pd.DataFrame,
    target_contract: TargetContract,
) -> tuple[list[str], list[str], list[str]]:
    feature_cols = select_tft_feature_columns(
        train_frame,
        excluded_cols={
            DEFAULT_DATE_COL,
            target_contract.learning_target_col,
            target_contract.absolute_target_col,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
        },
    )
    group_id_columns = select_explicit_tft_group_id_columns(train_frame)
    projection_columns = _bundle_projection_columns(
        feature_cols,
        group_id_cols=group_id_columns,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
        available_columns=set(train_frame.columns),
    )
    return feature_cols, group_id_columns, projection_columns


def _write_bundle_frames(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    output_paths: dict[str, Path],
    feature_cols: list[str],
    group_id_columns: list[str],
    projection_columns: list[str],
    target_contract: TargetContract,
) -> tuple[int, int, int, dict[str, str]]:
    train_rows = _write_projected_frame(
        train_frame,
        columns=projection_columns,
        output_path=output_paths["train"],
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
    )
    tuning_rows = _write_projected_frame(
        tuning_frame,
        columns=projection_columns,
        output_path=output_paths["tuning"],
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
    )
    valid_rows = 0
    if valid_frame is not None:
        valid_rows = _write_projected_frame(
            valid_frame,
            columns=projection_columns,
            output_path=output_paths["valid"],
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            learning_target_col=target_contract.learning_target_col,
            absolute_target_col=target_contract.absolute_target_col,
            reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
        )
    projection_dtypes = {
        str(column): str(dtype)
        for column, dtype in pd.read_parquet(output_paths["train"]).dtypes.items()
    }
    return train_rows, tuning_rows, valid_rows, projection_dtypes


def _bundle_manifest_payload(
    *,
    train_path: Path,
    tuning_path: Path,
    valid_path: Path | None,
    output_paths: dict[str, Path],
    train_rows: int,
    tuning_rows: int,
    valid_rows: int,
    feature_count: int,
) -> dict[str, object]:
    bundle_manifest: dict[str, object] = {
        "train_input_path": str(train_path),
        "tuning_input_path": str(tuning_path),
        "valid_input_path": str(valid_path) if valid_path is not None else None,
        "train_rows": train_rows,
        "tuning_rows": tuning_rows,
        "valid_rows": valid_rows,
        "feature_count": feature_count,
        "feature_manifest_path": str(output_paths["feature_manifest"]),
        "target_contract_path": str(output_paths["target_contract"]),
        "train_sha256": sha256_file(output_paths["train"]),
        "tuning_sha256": sha256_file(output_paths["tuning"]),
    }
    if valid_path is not None:
        bundle_manifest["valid_sha256"] = sha256_file(output_paths["valid"])
    return bundle_manifest


def _persist_bundle_metadata(
    *,
    output_paths: dict[str, Path],
    feature_cols: list[str],
    group_id_columns: list[str],
    projection_columns: list[str],
    projection_dtypes: dict[str, str],
    target_contract: TargetContract,
    train_path: Path,
    tuning_path: Path,
    valid_path: Path | None,
    train_rows: int,
    tuning_rows: int,
    valid_rows: int,
) -> None:
    _json_dump(
        output_paths["feature_manifest"],
        _feature_manifest_payload(
            feature_cols,
            group_id_columns=group_id_columns,
            projection_columns=projection_columns,
            projection_dtypes=projection_dtypes,
        ),
    )
    _json_dump(output_paths["target_contract"], build_target_contract_metadata(target_contract))
    _json_dump(
        output_paths["bundle_manifest"],
        _bundle_manifest_payload(
            train_path=train_path,
            tuning_path=tuning_path,
            valid_path=valid_path,
            output_paths=output_paths,
            train_rows=train_rows,
            tuning_rows=tuning_rows,
            valid_rows=valid_rows,
            feature_count=len(feature_cols),
        ),
    )


def build_training_bundle(
    *,
    train_input_path: str | Path = DEFAULT_TRAIN_INPUT_PATH,
    tuning_input_path: str | Path = DEFAULT_TUNING_INPUT_PATH,
    valid_input_path: str | Path | None = DEFAULT_VALID_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target_col: str = DEFAULT_VARIATION_TARGET_COL,
) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    train_path, tuning_path, valid_path, train_frame, tuning_frame, valid_frame = _load_bundle_frames(train_input_path=train_input_path, tuning_input_path=tuning_input_path, valid_input_path=valid_input_path)
    train_frame, tuning_frame, valid_frame, target_contract = _resolve_bundle_contract(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        valid_frame=valid_frame,
        target_col=target_col,
    )
    feature_cols, group_id_columns, projection_columns = _resolve_bundle_features(train_frame=train_frame, target_contract=target_contract)
    output_paths = _bundle_output_paths(resolved_output_dir, include_valid=valid_frame is not None)
    train_rows, tuning_rows, valid_rows, projection_dtypes = _write_bundle_frames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        valid_frame=valid_frame,
        output_paths=output_paths,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        projection_columns=projection_columns,
        target_contract=target_contract,
    )
    _persist_bundle_metadata(
        output_paths=output_paths,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        projection_columns=projection_columns,
        projection_dtypes=projection_dtypes,
        target_contract=target_contract,
        train_path=train_path,
        tuning_path=tuning_path,
        valid_path=valid_path,
        train_rows=train_rows,
        tuning_rows=tuning_rows,
        valid_rows=valid_rows,
    )
    return output_paths
