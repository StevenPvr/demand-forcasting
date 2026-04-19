from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASET_NAME = "Dingdong-Inc/FreshRetailNet-50K"
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_CSV_FRACTION = 0.05
DEFAULT_SEED = 42
DEFAULT_EXCLUDED_COLUMNS: tuple[str, ...] = ()


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
    return dataset.select(range(preview_rows))


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

    flagged_dataset = dataset.map(
        lambda batch: {
            "is_censored": [value > 0 for value in batch["stock_hour6_22_cnt"]],
        },
        batched=True,
    )
    return flagged_dataset


def flatten_hourly_columns(dataset: Dataset) -> Dataset:
    hourly_columns = [column for column in ("hours_sale", "hours_stock_status") if column in dataset.column_names]
    if not hourly_columns:
        return dataset

    def expand_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        expanded: dict[str, list[Any]] = {}
        for column in hourly_columns:
            for hour in range(24):
                expanded[f"{column}_{hour:02d}"] = [row[hour] for row in batch[column]]
        return expanded

    expanded_dataset = dataset.map(expand_batch, batched=True)
    return drop_columns_if_present(expanded_dataset, hourly_columns)


def _resolve_splits(dataset_or_dict: Dataset | DatasetDict) -> tuple[Dataset, dict[str, Dataset]]:
    if isinstance(dataset_or_dict, Dataset):
        generated = build_train_val_splits(dataset_or_dict)
        return dataset_or_dict, generated

    if "train" in dataset_or_dict and "val" in dataset_or_dict:
        return dataset_or_dict["train"], {
            "train": dataset_or_dict["train"],
            "val": dataset_or_dict["val"],
        }

    if "train" in dataset_or_dict and "validation" in dataset_or_dict:
        return dataset_or_dict["train"], {
            "train": dataset_or_dict["train"],
            "val": dataset_or_dict["validation"],
        }

    if "train" in dataset_or_dict and "eval" in dataset_or_dict:
        return dataset_or_dict["train"], {
            "train": dataset_or_dict["train"],
            "val": dataset_or_dict["eval"],
        }

    if "train" in dataset_or_dict:
        train_split = dataset_or_dict["train"]
        return train_split, build_train_val_splits(train_split)

    if len(dataset_or_dict) == 1:
        only_split = next(iter(dataset_or_dict.values()))
        return only_split, build_train_val_splits(only_split)

    available = ", ".join(dataset_or_dict.keys())
    raise ValueError(f"Unable to infer train/val splits from: {available}")


def export_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str | Path = "data",
    val_fraction: float = DEFAULT_VAL_FRACTION,
    csv_fraction: float = DEFAULT_CSV_FRACTION,
    seed: int = DEFAULT_SEED,
    excluded_columns: list[str] | tuple[str, ...] = DEFAULT_EXCLUDED_COLUMNS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_or_dict = load_dataset(dataset_name)
    preview_source, resolved_splits = _resolve_splits(dataset_or_dict)
    if "train" not in resolved_splits or "val" not in resolved_splits:
        raise ValueError("Resolved splits must contain train and val")

    if isinstance(dataset_or_dict, Dataset) or (
        isinstance(dataset_or_dict, DatasetDict)
        and not ("val" in dataset_or_dict or "validation" in dataset_or_dict or "eval" in dataset_or_dict)
    ):
        resolved_splits = build_train_val_splits(
            preview_source,
            val_fraction=val_fraction,
            seed=seed,
        )

    dataset_slug = _slugify_dataset_name(dataset_name)
    preview = add_is_censored_flag(
        flatten_hourly_columns(
            drop_columns_if_present(
                build_csv_preview(preview_source, csv_fraction=csv_fraction),
                excluded_columns,
            )
        )
    )

    csv_path = output_path / f"{dataset_slug}_first_{int(csv_fraction * 100)}pct.csv"
    preview.to_csv(str(csv_path), index=False)

    parquet_paths: dict[str, str] = {}
    split_row_counts: dict[str, int] = {}
    for split_name, split_dataset in resolved_splits.items():
        split_dataset = add_is_censored_flag(
            flatten_hourly_columns(
                drop_columns_if_present(split_dataset, excluded_columns)
            )
        )
        parquet_path = output_path / f"{dataset_slug}_{split_name}.parquet"
        split_dataset.to_parquet(str(parquet_path))
        parquet_paths[split_name] = str(parquet_path)
        split_row_counts[split_name] = split_dataset.num_rows

    metadata = {
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
    }
    metadata_path = output_path / f"{dataset_slug}_export_metadata.json"
    metadata["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FreshRetailNet to parquet with a train/val split and a CSV preview."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--csv-fraction", type=float, default=DEFAULT_CSV_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        csv_fraction=args.csv_fraction,
        seed=args.seed,
        excluded_columns=DEFAULT_EXCLUDED_COLUMNS,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
