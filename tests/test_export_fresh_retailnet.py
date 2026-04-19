from pathlib import Path
import sys
import unittest

from datasets import Dataset, DatasetDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.export_fresh_retailnet import (  # noqa: E402
    add_is_censored_flag,
    build_csv_preview,
    build_train_val_splits,
    drop_columns_if_present,
    flatten_hourly_columns,
    fractional_count,
    export_dataset,
)


class ExportFreshRetailNetTests(unittest.TestCase):
    def test_fractional_count_uses_floor_with_minimum_one(self) -> None:
        self.assertEqual(fractional_count(50_000, 0.05), 2_500)
        self.assertEqual(fractional_count(1, 0.05), 1)

    def test_build_csv_preview_keeps_first_rows_in_order(self) -> None:
        dataset = Dataset.from_dict({"row_id": list(range(10))})

        preview = build_csv_preview(dataset, csv_fraction=0.3)

        self.assertEqual(preview["row_id"], [0, 1, 2])

    def test_build_train_val_splits_is_deterministic_and_exhaustive(self) -> None:
        dataset = Dataset.from_dict({"row_id": list(range(10))})

        split_once = build_train_val_splits(dataset, val_fraction=0.2, seed=7)
        split_twice = build_train_val_splits(dataset, val_fraction=0.2, seed=7)

        self.assertEqual(split_once["train"].num_rows, 8)
        self.assertEqual(split_once["val"].num_rows, 2)
        self.assertEqual(split_once["train"]["row_id"], split_twice["train"]["row_id"])
        self.assertEqual(split_once["val"]["row_id"], split_twice["val"]["row_id"])
        self.assertEqual(
            sorted(list(split_once["train"]["row_id"]) + list(split_once["val"]["row_id"])),
            list(range(10)),
        )

    def test_drop_columns_if_present_removes_requested_columns(self) -> None:
        dataset = Dataset.from_dict(
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
        dataset = Dataset.from_dict(
            {
                "row_id": [0, 1, 2],
                "stock_hour6_22_cnt": [0, 1, 4],
            }
        )

        flagged = add_is_censored_flag(dataset)

        self.assertEqual(flagged["is_censored"], [False, True, True])

    def test_flatten_hourly_columns_expands_24_hours_and_removes_source_columns(self) -> None:
        dataset = Dataset.from_dict(
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
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "row_id": [0, 1, 2, 3],
                        "stock_hour6_22_cnt": [0, 1, 0, 2],
                        "hours_sale": [[0.1] * 24] * 4,
                        "hours_stock_status": [[1] * 24] * 4,
                    }
                ),
                "eval": Dataset.from_dict(
                    {
                        "row_id": [4, 5],
                        "stock_hour6_22_cnt": [0, 3],
                        "hours_sale": [[0.2] * 24] * 2,
                        "hours_stock_status": [[0] * 24] * 2,
                    }
                ),
            }
        )

        captured_csv_columns: list[str] = []
        captured_parquet_columns: list[list[str]] = []
        captured_csv_is_censored: list[bool] = []
        captured_parquet_is_censored: list[list[bool]] = []

        def fake_to_csv(dataset_self: Dataset, *args: object, **kwargs: object) -> None:
            captured_csv_columns.extend(dataset_self.column_names)
            captured_csv_is_censored.extend(dataset_self["is_censored"])

        def fake_to_parquet(dataset_self: Dataset, *args: object, **kwargs: object) -> None:
            captured_parquet_columns.append(dataset_self.column_names)
            captured_parquet_is_censored.append(list(dataset_self["is_censored"]))

        with unittest.mock.patch(
            "research_praedixa.export_fresh_retailnet.load_dataset",
            return_value=dataset_dict,
        ):
            with unittest.mock.patch.object(Dataset, "to_csv", new=fake_to_csv):
                with unittest.mock.patch.object(Dataset, "to_parquet", new=fake_to_parquet):
                    with unittest.mock.patch(
                        "pathlib.Path.write_text",
                        return_value=0,
                    ):
                        metadata = export_dataset(output_dir=PROJECT_ROOT / "data")

        self.assertEqual(metadata["split_rows"], {"train": 4, "val": 2})
        self.assertEqual(len(captured_parquet_columns), 2)
        self.assertIn("is_censored", captured_csv_columns)
        self.assertIn("hours_sale_00", captured_csv_columns)
        self.assertIn("hours_sale_23", captured_csv_columns)
        self.assertIn("hours_stock_status_00", captured_csv_columns)
        self.assertIn("hours_stock_status_23", captured_csv_columns)
        self.assertNotIn("hours_sale", captured_csv_columns)
        self.assertNotIn("hours_stock_status", captured_csv_columns)
        self.assertTrue(all("hours_sale_00" in columns for columns in captured_parquet_columns))
        self.assertTrue(all("hours_sale_23" in columns for columns in captured_parquet_columns))
        self.assertTrue(all("hours_stock_status_00" in columns for columns in captured_parquet_columns))
        self.assertTrue(all("hours_stock_status_23" in columns for columns in captured_parquet_columns))
        self.assertTrue(all("hours_sale" not in columns for columns in captured_parquet_columns))
        self.assertTrue(all("hours_stock_status" not in columns for columns in captured_parquet_columns))
        self.assertTrue(all("is_censored" in columns for columns in captured_parquet_columns))
        self.assertEqual(captured_csv_is_censored, [False])
        self.assertEqual(captured_parquet_is_censored, [[False, True, False, True], [False, True]])


if __name__ == "__main__":
    unittest.main()
