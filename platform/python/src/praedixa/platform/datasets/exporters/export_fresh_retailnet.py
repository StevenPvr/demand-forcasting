from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import datasets
from datasets import Dataset, DatasetDict


DATASETS_API: Any = datasets


DEFAULT_DATASET_NAME = "Dingdong-Inc/FreshRetailNet-50K"
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_CSV_FRACTION = 0.05
DEFAULT_SEED = 42
DEFAULT_EXCLUDED_COLUMNS: tuple[str, ...] = ()


@dataclass(frozen=True)
class FreshRetailNetExportConfig:
    dataset_name: str = DEFAULT_DATASET_NAME
    output_dir: str = "data"
    val_fraction: float = DEFAULT_VAL_FRACTION
    csv_fraction: float = DEFAULT_CSV_FRACTION
    seed: int = DEFAULT_SEED


def _disable_datasets_progress_bars() -> None:
    if hasattr(DATASETS_API, "disable_progress_bars"):
        DATASETS_API.disable_progress_bars()
        return
    if hasattr(DATASETS_API, "disable_progress_bar"):
        DATASETS_API.disable_progress_bar()


def fractional_count(total_rows: int, fraction: float) -> int:
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if total_rows == 0:
        return 0
    return max(1, int(total_rows * fraction))


def build_csv_preview(dataset: Dataset, csv_fraction: float = DEFAULT_CSV_FRACTION) -> Dataset:
    preview_rows = fractional_count(dataset.num_rows, csv_fraction)
    dataset_api: Any = dataset
    return cast(Dataset, dataset_api.select(range(preview_rows)))


def build_train_val_splits(
    dataset: Dataset,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Dataset]:
    split = dataset.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
    return {"train": split["train"], "val": split["test"]}


def _slugify_dataset_name(dataset_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")


def drop_columns_if_present(dataset: Dataset, columns: list[str] | tuple[str, ...]) -> Dataset:
    removable = [column for column in columns if column in dataset.column_names]
    if not removable:
        return dataset
    return dataset.remove_columns(removable)


def add_is_censored_flag(dataset: Dataset) -> Dataset:
    if "stock_hour6_22_cnt" not in dataset.column_names:
        return dataset
    _disable_datasets_progress_bars()

    def censor_batch(batch: dict[str, list[Any]]) -> dict[str, list[bool]]:
        stock_counts = [int(cast(int | float | bool, value)) for value in batch["stock_hour6_22_cnt"]]
        return {"is_censored": [value > 0 for value in stock_counts]}

    dataset_api: Any = dataset
    flagged_dataset = cast(Dataset, dataset_api.map(censor_batch, batched=True))
    return flagged_dataset


def flatten_hourly_columns(dataset: Dataset) -> Dataset:
    hourly_columns = [column for column in ("hours_sale", "hours_stock_status") if column in dataset.column_names]
    if not hourly_columns:
        return dataset
    _disable_datasets_progress_bars()

    def expand_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        expanded: dict[str, list[Any]] = {}
        for column in hourly_columns:
            for hour in range(24):
                expanded[f"{column}_{hour:02d}"] = [row[hour] for row in batch[column]]
        return expanded

    dataset_api: Any = dataset
    expanded_dataset = cast(Dataset, dataset_api.map(expand_batch, batched=True))
    return drop_columns_if_present(expanded_dataset, hourly_columns)


def _known_validation_splits(dataset_dict: DatasetDict) -> dict[str, Dataset] | None:
    for validation_name in ("val", "validation", "eval"):
        if "train" in dataset_dict and validation_name in dataset_dict:
            return {
                "train": dataset_dict["train"],
                "val": dataset_dict[validation_name],
            }
    return None


def _fallback_train_val_splits(dataset_dict: DatasetDict) -> tuple[Dataset, dict[str, Dataset]] | None:
    if "train" in dataset_dict:
        train_split = dataset_dict["train"]
        return train_split, build_train_val_splits(train_split)
    if len(dataset_dict) == 1:
        only_split = next(iter(dataset_dict.values()))
        return only_split, build_train_val_splits(only_split)
    return None


def resolve_splits(dataset_or_dict: Dataset | DatasetDict) -> tuple[Dataset, dict[str, Dataset]]:
    if isinstance(dataset_or_dict, Dataset):
        generated = build_train_val_splits(dataset_or_dict)
        return dataset_or_dict, generated

    known_splits = _known_validation_splits(dataset_or_dict)
    if known_splits is not None:
        return known_splits["train"], known_splits

    fallback_splits = _fallback_train_val_splits(dataset_or_dict)
    if fallback_splits is not None:
        return fallback_splits

    available = ", ".join(str(split_name) for split_name in dataset_or_dict.keys())
    raise ValueError(f"Unable to infer train/val splits from: {available}")


def _dataset_needs_generated_validation_split(dataset_or_dict: Dataset | DatasetDict) -> bool:
    if isinstance(dataset_or_dict, Dataset):
        return True
    return not any(split_name in dataset_or_dict for split_name in ("val", "validation", "eval"))


def _build_export_metadata(
    *,
    dataset_name: str,
    seed: int,
    val_fraction: float,
    csv_fraction: float,
    excluded_columns: list[str] | tuple[str, ...],
    preview: Dataset,
    split_row_counts: dict[str, int],
    parquet_paths: dict[str, str],
    csv_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "seed": seed,
        "val_fraction": val_fraction,
        "csv_fraction": csv_fraction,
        "excluded_columns": list(excluded_columns),
        "derived_columns": [
            "is_censored",
            "hours_sale_00..23",
            "hours_stock_status_00..23",
        ],
        "preview_rows": preview.num_rows,
        "split_rows": split_row_counts,
        "parquet_paths": parquet_paths,
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
    }


def _load_export_source(
    dataset_name: str,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[Dataset, dict[str, Dataset]]:
    dataset_or_dict = cast(Dataset | DatasetDict, DATASETS_API.load_dataset(dataset_name))
    preview_source, resolved_splits = resolve_splits(dataset_or_dict)
    if "train" not in resolved_splits or "val" not in resolved_splits:
        raise ValueError("Resolved splits must contain train and val")
    if _dataset_needs_generated_validation_split(dataset_or_dict):
        resolved_splits = build_train_val_splits(
            preview_source,
            val_fraction=val_fraction,
            seed=seed,
        )
    return preview_source, resolved_splits


def _prepare_export_preview(
    preview_source: Dataset,
    *,
    csv_fraction: float,
    excluded_columns: list[str] | tuple[str, ...],
) -> Dataset:
    return add_is_censored_flag(
        flatten_hourly_columns(
            drop_columns_if_present(
                build_csv_preview(preview_source, csv_fraction=csv_fraction),
                excluded_columns,
            )
        )
    )


def _write_preview_csv(preview: Dataset, *, output_path: Path, dataset_slug: str, csv_fraction: float) -> Path:
    csv_path = output_path / f"{dataset_slug}_first_{int(csv_fraction * 100)}pct.csv"
    cast(Any, preview).to_csv(str(csv_path), index=False)
    return csv_path


def _export_resolved_splits(
    resolved_splits: dict[str, Dataset],
    *,
    output_path: Path,
    dataset_slug: str,
    excluded_columns: list[str] | tuple[str, ...],
) -> tuple[dict[str, str], dict[str, int]]:
    parquet_paths: dict[str, str] = {}
    split_row_counts: dict[str, int] = {}
    for split_name, split_dataset in resolved_splits.items():
        exported_split = add_is_censored_flag(
            flatten_hourly_columns(
                drop_columns_if_present(split_dataset, excluded_columns)
            )
        )
        parquet_path = output_path / f"{dataset_slug}_{split_name}.parquet"
        cast(Any, exported_split).to_parquet(str(parquet_path))
        if split_name in {"train", "val"}:
            canonical_parquet_path = output_path / f"data_{split_name}.parquet"
            cast(Any, exported_split).to_parquet(str(canonical_parquet_path))
        parquet_paths[split_name] = str(parquet_path)
        split_row_counts[split_name] = exported_split.num_rows
    return parquet_paths, split_row_counts


def export_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str | Path = "data",
    val_fraction: float = DEFAULT_VAL_FRACTION,
    csv_fraction: float = DEFAULT_CSV_FRACTION,
    seed: int = DEFAULT_SEED,
    excluded_columns: list[str] | tuple[str, ...] = DEFAULT_EXCLUDED_COLUMNS,
) -> dict[str, Any]:
    _disable_datasets_progress_bars()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preview_source, resolved_splits = _load_export_source(
        dataset_name,
        val_fraction=val_fraction,
        seed=seed,
    )
    dataset_slug = _slugify_dataset_name(dataset_name)
    preview = _prepare_export_preview(
        preview_source,
        csv_fraction=csv_fraction,
        excluded_columns=excluded_columns,
    )
    csv_path = _write_preview_csv(
        preview,
        output_path=output_path,
        dataset_slug=dataset_slug,
        csv_fraction=csv_fraction,
    )
    parquet_paths, split_row_counts = _export_resolved_splits(
        resolved_splits,
        output_path=output_path,
        dataset_slug=dataset_slug,
        excluded_columns=excluded_columns,
    )
    metadata_path = output_path / f"{dataset_slug}_export_metadata.json"
    metadata = _build_export_metadata(
        dataset_name=dataset_name,
        seed=seed,
        val_fraction=val_fraction,
        csv_fraction=csv_fraction,
        excluded_columns=excluded_columns,
        preview=preview,
        split_row_counts=split_row_counts,
        parquet_paths=parquet_paths,
        csv_path=csv_path,
        metadata_path=metadata_path,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_default_export_config() -> FreshRetailNetExportConfig:
    return FreshRetailNetExportConfig()


def main() -> None:
    config = build_default_export_config()
    metadata = export_dataset(
        dataset_name=config.dataset_name,
        output_dir=config.output_dir,
        val_fraction=config.val_fraction,
        csv_fraction=config.csv_fraction,
        seed=config.seed,
        excluded_columns=DEFAULT_EXCLUDED_COLUMNS,
    )
    logging.getLogger(__name__).info(
        "FreshRetail export metadata: %s",
        json.dumps(metadata, indent=2),
    )


if __name__ == "__main__":
    main()
