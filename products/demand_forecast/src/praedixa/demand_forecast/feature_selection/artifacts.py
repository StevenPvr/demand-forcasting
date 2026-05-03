"""Bundle artifact helpers for feature-selected training bundles."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping, cast

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_contract import (
    build_feature_contract,
)
from praedixa.demand_forecast.training_bundle.manifest import sha256_file


DATE_COL = "dt"
FEATURE_MANIFEST = "feature_manifest.json"
TARGET_CONTRACT = "target_contract.json"
TRAIN_SPLIT = "train.parquet"
TUNING_SPLIT = "tuning.parquet"
VALID_SPLIT = "valid.parquet"
OPTIMISATION_TRAIN_SPLIT = "optimisation_train.parquet"
OPTIMISATION_TUNING_SPLIT = "optimisation_tuning.parquet"
OPTIMISATION_VALID_SPLIT = "optimisation_valid.parquet"


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: Mapping[str, object | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_bundle_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if DATE_COL in frame.columns:
        frame[DATE_COL] = pd.to_datetime(frame[DATE_COL])
        return frame.sort_values(DATE_COL).reset_index(drop=True)
    return frame.reset_index(drop=True)


def target_column(bundle_dir: Path) -> str:
    payload = read_json(bundle_dir / TARGET_CONTRACT)
    target = payload.get("learning_target_col") or payload.get("absolute_target_col")
    if not isinstance(target, str):
        raise ValueError(f"`{bundle_dir / TARGET_CONTRACT}` does not define a target.")
    return target


def feature_manifest(bundle_dir: Path) -> dict[str, Any]:
    payload = read_json(bundle_dir / FEATURE_MANIFEST)
    if not isinstance(payload.get("feature_columns"), list):
        raise ValueError(f"`{bundle_dir / FEATURE_MANIFEST}` has no feature_columns.")
    return payload


def feature_columns(manifest: Mapping[str, Any]) -> list[str]:
    return [str(column) for column in cast(list[object], manifest["feature_columns"])]


def feature_roles(manifest: Mapping[str, Any]) -> dict[str, str]:
    raw_roles = manifest.get("feature_roles", {})
    if not isinstance(raw_roles, dict):
        return {}
    typed_roles = cast(Mapping[object, object], raw_roles)
    return {str(key): str(value) for key, value in typed_roles.items()}


def write_filtered_split(
    *,
    input_path: Path,
    output_path: Path,
    original_feature_columns: list[str],
    selected_feature_columns: list[str],
) -> int:
    if not input_path.exists():
        return 0
    frame = load_bundle_frame(input_path)
    filtered = _filter_frame(
        frame,
        original_feature_columns=original_feature_columns,
        selected_feature_columns=selected_feature_columns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)
    return int(len(filtered))


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_selected_feature_manifest(
    *,
    input_manifest: Mapping[str, Any],
    output_path: Path,
    selected_feature_columns: list[str],
) -> None:
    roles = feature_roles(input_manifest)
    selected_contract = _selected_feature_contract(selected_feature_columns)
    payload: dict[str, object] = dict(input_manifest)
    payload["feature_columns"] = selected_feature_columns
    payload["feature_count"] = len(selected_feature_columns)
    payload["feature_roles"] = {
        column: roles[column] for column in selected_feature_columns if column in roles
    }
    payload["feature_contract"] = selected_contract
    projection_columns = payload.get("projection_columns")
    if isinstance(projection_columns, list):
        typed_projection_columns = cast(list[object], projection_columns)
        payload["projection_columns"] = [
            str(column)
            for column in typed_projection_columns
            if str(column) not in roles or str(column) in selected_feature_columns
        ]
    write_json(output_path, payload)


def write_selected_feature_roles(
    *,
    output_path: Path,
    selected_feature_columns: list[str],
) -> None:
    write_json(output_path, _selected_feature_contract(selected_feature_columns))


def write_selected_optimisation_manifest(
    *,
    input_manifest_path: Path,
    output_path: Path,
    selected_feature_manifest: Mapping[str, Any],
    selected_feature_columns: list[str],
    include_valid: bool,
) -> None:
    support_columns = ["__tft_group_id", "__tft_time_idx"]
    group_id_columns = selected_feature_manifest.get("group_id_columns", [])
    if not isinstance(group_id_columns, list):
        group_id_columns = []
    typed_group_id_columns = cast(list[object], group_id_columns)
    input_manifest = (
        read_json(input_manifest_path) if input_manifest_path.exists() else {}
    )
    projection_columns = _selected_optimisation_projection_columns(
        selected_feature_manifest,
        selected_feature_columns=selected_feature_columns,
    )
    payload: dict[str, object] = {
        "bundle_version": int(input_manifest.get("bundle_version", 1)),
        "train_path": str(output_path.parent / OPTIMISATION_TRAIN_SPLIT),
        "tuning_path": str(output_path.parent / OPTIMISATION_TUNING_SPLIT),
        "feature_manifest_path": str(output_path.parent / FEATURE_MANIFEST),
        "target_contract_path": str(output_path.parent / TARGET_CONTRACT),
        "projection_columns": [*projection_columns, *support_columns],
        "group_id_columns": [str(column) for column in typed_group_id_columns],
        "support_columns": support_columns,
        "train_group_count": input_manifest.get("train_group_count", 0),
        "precomputed_tft_support": True,
    }
    if include_valid:
        payload["valid_path"] = str(output_path.parent / OPTIMISATION_VALID_SPLIT)
    write_json(output_path, payload)


def selected_bundle_manifest(
    *,
    source_bundle_dir: Path,
    output_dir: Path,
    train_rows: int,
    tuning_rows: int,
    valid_rows: int,
    selected_feature_count: int,
    summary_path: Path,
    importance_path: Path,
    trials_path: Path,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "bundle_version": 3,
        "source_bundle_dir": str(source_bundle_dir),
        "feature_selection_summary_path": str(summary_path),
        "feature_selection_importances_path": str(importance_path),
        "feature_selection_trials_path": str(trials_path),
        "train_rows": train_rows,
        "tuning_rows": tuning_rows,
        "valid_rows": valid_rows,
        "feature_count": selected_feature_count,
        "train_path": str(output_dir / TRAIN_SPLIT),
        "tuning_path": str(output_dir / TUNING_SPLIT),
        "optimisation_train_path": str(output_dir / OPTIMISATION_TRAIN_SPLIT),
        "optimisation_tuning_path": str(output_dir / OPTIMISATION_TUNING_SPLIT),
        "optimisation_manifest_path": str(output_dir / "optimisation_manifest.json"),
        "feature_manifest_path": str(output_dir / FEATURE_MANIFEST),
        "feature_roles_path": str(output_dir / "feature_roles.json"),
        "split_manifest_path": str(output_dir / "split_manifest.json"),
        "target_contract_path": str(output_dir / TARGET_CONTRACT),
        "training_exclusion_report_path": str(
            output_dir / "training_exclusion_report.json"
        ),
        "train_sha256": sha256_file(output_dir / TRAIN_SPLIT),
        "tuning_sha256": sha256_file(output_dir / TUNING_SPLIT),
    }
    if valid_rows > 0:
        payload["valid_path"] = str(output_dir / VALID_SPLIT)
        payload["valid_sha256"] = sha256_file(output_dir / VALID_SPLIT)
        payload["optimisation_valid_path"] = str(output_dir / OPTIMISATION_VALID_SPLIT)
    return payload


def _selected_output_columns(
    frame: pd.DataFrame,
    *,
    original_feature_columns: list[str],
    selected_feature_columns: list[str],
) -> list[str]:
    selected_features = set(selected_feature_columns)
    original_features = set(original_feature_columns)
    return [
        column
        for column in frame.columns
        if column not in original_features or column in selected_features
    ]


def _filter_frame(
    frame: pd.DataFrame,
    *,
    original_feature_columns: list[str],
    selected_feature_columns: list[str],
) -> pd.DataFrame:
    columns = _selected_output_columns(
        frame,
        original_feature_columns=original_feature_columns,
        selected_feature_columns=selected_feature_columns,
    )
    return frame.loc[:, columns].copy()


def _selected_feature_contract(
    selected_feature_columns: list[str],
) -> dict[str, object]:
    return cast(dict[str, object], build_feature_contract(selected_feature_columns))


def _selected_optimisation_projection_columns(
    input_manifest: Mapping[str, Any],
    *,
    selected_feature_columns: list[str],
) -> list[str]:
    selected_features = set(selected_feature_columns)
    original_feature_columns = set(feature_columns(input_manifest))
    raw_projection_columns = input_manifest.get("projection_columns", [])
    if not isinstance(raw_projection_columns, list):
        return selected_feature_columns
    typed_projection_columns = cast(list[object], raw_projection_columns)
    return [
        str(column)
        for column in typed_projection_columns
        if str(column) not in original_feature_columns
        or str(column) in selected_features
    ]
