from pathlib import Path
import sys
from typing import Any, cast
import unittest
from unittest import mock

from datasets import Dataset, DatasetDict


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.exporters import export_fresh_retailnet as export_module  # noqa: E402


DATASET_API: Any = Dataset


fractional_count = export_module.fractional_count
build_csv_preview = export_module.build_csv_preview
build_train_val_splits = export_module.build_train_val_splits
drop_columns_if_present = export_module.drop_columns_if_present
add_is_censored_flag = export_module.add_is_censored_flag
flatten_hourly_columns = export_module.flatten_hourly_columns
export_dataset = export_module.export_dataset
resolve_splits = export_module.resolve_splits


def _dataset_from_dict(payload: dict[str, object]) -> Dataset:
    return cast(Dataset, DATASET_API.from_dict(payload))


def _dataset_column_values(dataset: Dataset, column: str) -> list[object]:
    return list(cast(Any, dataset[column]))


def _dataset_int_values(dataset: Dataset, column: str) -> list[int]:
    return [int(cast(int, value)) for value in _dataset_column_values(dataset, column)]


def _hourly_value_rows(value: float, row_count: int) -> list[list[float]]:
    return [[value] * 24] * row_count


def _hourly_status_rows(value: int, row_count: int) -> list[list[int]]:
    return [[value] * 24] * row_count


def _fake_export_writers(
    *,
    captured_csv_columns: list[str],
    captured_parquet_columns: list[list[str]],
    captured_csv_is_censored: list[bool],
    captured_parquet_is_censored: list[list[bool]],
) -> tuple[Any, Any]:
    def fake_to_csv(dataset_self: Dataset, *args: object, **kwargs: object) -> None:
        del args, kwargs
        captured_csv_columns.extend(dataset_self.column_names)
        captured_csv_is_censored.extend(bool(value) for value in _dataset_column_values(dataset_self, "is_censored"))

    def fake_to_parquet(dataset_self: Dataset, *args: object, **kwargs: object) -> None:
        del args, kwargs
        captured_parquet_columns.append(dataset_self.column_names)
        captured_parquet_is_censored.append([bool(value) for value in _dataset_column_values(dataset_self, "is_censored")])

    return fake_to_csv, fake_to_parquet


def _fresh_retailnet_existing_eval_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": _dataset_from_dict(
                {
                    "row_id": [0, 1, 2, 3],
                    "stock_hour6_22_cnt": [0, 1, 0, 2],
                    "hours_sale": _hourly_value_rows(0.1, 4),
                    "hours_stock_status": _hourly_status_rows(1, 4),
                }
            ),
            "eval": _dataset_from_dict(
                {
                    "row_id": [4, 5],
                    "stock_hour6_22_cnt": [0, 3],
                    "hours_sale": _hourly_value_rows(0.2, 2),
                    "hours_stock_status": _hourly_status_rows(0, 2),
                }
            ),
        }
    )


def _assert_export_dataset_metadata_and_columns(
    *,
    metadata: dict[str, object],
    captured_csv_columns: list[str],
    captured_parquet_columns: list[list[str]],
    captured_csv_is_censored: list[bool],
    captured_parquet_is_censored: list[list[bool]],
) -> None:
    _assert_export_split_rows(metadata)
    _assert_csv_flattened_columns(captured_csv_columns)
    _assert_parquet_flattened_columns(captured_parquet_columns)
    _assert_censoring_flags(captured_csv_is_censored, captured_parquet_is_censored)


def _expected_flattened_columns() -> set[str]:
    return {
        "hours_sale_00",
        "hours_sale_23",
        "hours_stock_status_00",
        "hours_stock_status_23",
        "is_censored",
    }


def _assert_export_split_rows(metadata: dict[str, object]) -> None:
    assert metadata["split_rows"] == {"train": 4, "val": 2}


def _assert_csv_flattened_columns(captured_csv_columns: list[str]) -> None:
    expected_flattened_columns = _expected_flattened_columns()
    assert expected_flattened_columns <= set(captured_csv_columns)
    assert "hours_sale" not in captured_csv_columns
    assert "hours_stock_status" not in captured_csv_columns


def _assert_parquet_flattened_columns(captured_parquet_columns: list[list[str]]) -> None:
    expected_flattened_columns = _expected_flattened_columns()
    assert len(captured_parquet_columns) == 2
    assert all(expected_flattened_columns <= set(columns) for columns in captured_parquet_columns)
    assert all("hours_sale" not in columns for columns in captured_parquet_columns)
    assert all("hours_stock_status" not in columns for columns in captured_parquet_columns)


def _assert_censoring_flags(
    captured_csv_is_censored: list[bool],
    captured_parquet_is_censored: list[list[bool]],
) -> None:
    assert captured_csv_is_censored == [False]
    assert captured_parquet_is_censored == [[False, True, False, True], [False, True]]


class ExportFreshRetailNetTests(unittest.TestCase):
    def test_fractional_count_uses_floor_with_minimum_one(self) -> None:
        self.assertEqual(fractional_count(50_000, 0.05), 2_500)
        self.assertEqual(fractional_count(1, 0.05), 1)

    def test_build_csv_preview_keeps_first_rows_in_order(self) -> None:
        dataset = _dataset_from_dict({"row_id": list(range(10))})

        preview = build_csv_preview(dataset, csv_fraction=0.3)

        self.assertEqual(preview["row_id"], [0, 1, 2])

    def test_build_train_val_splits_is_deterministic_and_exhaustive(self) -> None:
        dataset = _dataset_from_dict({"row_id": list(range(10))})

        split_once = build_train_val_splits(dataset, val_fraction=0.2, seed=7)
        split_twice = build_train_val_splits(dataset, val_fraction=0.2, seed=7)

        self.assertEqual(split_once["train"].num_rows, 8)
        self.assertEqual(split_once["val"].num_rows, 2)
        self.assertEqual(_dataset_column_values(split_once["train"], "row_id"), _dataset_column_values(split_twice["train"], "row_id"))
        self.assertEqual(_dataset_column_values(split_once["val"], "row_id"), _dataset_column_values(split_twice["val"], "row_id"))
        self.assertEqual(
            sorted(_dataset_int_values(split_once["train"], "row_id") + _dataset_int_values(split_once["val"], "row_id")),
            list(range(10)),
        )

    def test_drop_columns_if_present_removes_requested_columns(self) -> None:
        dataset = _dataset_from_dict(
            {
                "row_id": [0, 1],
                "hours_sale": [[0.1] * 24, [0.0] * 24],
                "hours_stock_status": [[1] * 24, [0] * 24],
                "sale_amount": [0.1, 0.0],
            }
        )

        trimmed = drop_columns_if_present(dataset, ["hours_sale", "hours_stock_status"])

        self.assertEqual(trimmed.column_names, ["row_id", "sale_amount"])

    def test_add_is_censored_flag_uses_positive_stockout_hours(self) -> None:
        dataset = _dataset_from_dict(
            {
                "row_id": [0, 1, 2],
                "stock_hour6_22_cnt": [0, 1, 4],
            }
        )

        flagged = add_is_censored_flag(dataset)

        self.assertEqual(flagged["is_censored"], [False, True, True])

    def test_flatten_hourly_columns_expands_24_hours_and_removes_source_columns(self) -> None:
        dataset = _dataset_from_dict(
            {
                "row_id": [0],
                "hours_sale": [[float(hour) for hour in range(24)]],
                "hours_stock_status": [[hour % 2 for hour in range(24)]],
            }
        )

        flattened = flatten_hourly_columns(dataset)

        self.assertNotIn("hours_sale", flattened.column_names)
        self.assertNotIn("hours_stock_status", flattened.column_names)
        self.assertEqual(flattened["hours_sale_00"], [0.0])
        self.assertEqual(flattened["hours_sale_23"], [23.0])
        self.assertEqual(flattened["hours_stock_status_00"], [0])
        self.assertEqual(flattened["hours_stock_status_23"], [1])

    def test_export_dataset_uses_existing_eval_as_val(self) -> None:
        dataset_dict = _fresh_retailnet_existing_eval_dataset()
        captured_csv_columns: list[str] = []
        captured_parquet_columns: list[list[str]] = []
        captured_csv_is_censored: list[bool] = []
        captured_parquet_is_censored: list[list[bool]] = []
        fake_to_csv, fake_to_parquet = _fake_export_writers(
            captured_csv_columns=captured_csv_columns,
            captured_parquet_columns=captured_parquet_columns,
            captured_csv_is_censored=captured_csv_is_censored,
            captured_parquet_is_censored=captured_parquet_is_censored,
        )

        with mock.patch(
            "praedixa.platform.datasets.exporters.export_fresh_retailnet.DATASETS_API.load_dataset",
            return_value=dataset_dict,
        ):
            with mock.patch.object(Dataset, "to_csv", new=fake_to_csv):
                with mock.patch.object(Dataset, "to_parquet", new=fake_to_parquet):
                    with mock.patch(
                        "pathlib.Path.write_text",
                        return_value=0,
                    ):
                        metadata = export_dataset(output_dir=PROJECT_ROOT / "var" / "sources")
        _assert_export_dataset_metadata_and_columns(
            metadata=cast(dict[str, object], metadata),
            captured_csv_columns=captured_csv_columns,
            captured_parquet_columns=captured_parquet_columns,
            captured_csv_is_censored=captured_csv_is_censored,
            captured_parquet_is_censored=captured_parquet_is_censored,
        )

    def test_resolve_splits_reports_available_split_names(self) -> None:
        dataset_dict = DatasetDict(
            {
                "foo": _dataset_from_dict({"row_id": [0]}),
                "bar": _dataset_from_dict({"row_id": [1]}),
            }
        )

        with self.assertRaisesRegex(ValueError, "foo, bar"):
            resolve_splits(dataset_dict)


if __name__ == "__main__":
    unittest.main()
