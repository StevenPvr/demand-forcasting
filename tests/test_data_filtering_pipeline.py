from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.data_filtering.pipeline import (  # noqa: E402
    add_shifted_target_columns,
    build_filtered_datasets,
    drop_future_only_columns,
    filter_rows_for_recovered_modeling,
    filter_uncensored_rows,
)


class DataFilteringPipelineTests(unittest.TestCase):
    def test_add_shifted_target_columns_creates_next_day_target(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-04-01", "2024-04-02", "2024-04-03"],
                "store_id": [1, 1, 1],
                "product_id": [10, 10, 10],
                "stock_hour6_22_cnt": [0, 1, 0],
                "sale_amount": [0.1, 0.2, 0.3],
            }
        )

        shifted = add_shifted_target_columns(frame)

        self.assertEqual(list(shifted["target"][:-1]), [0.2, 0.3])
        self.assertEqual(list(shifted["target_stock_hour6_22_cnt_next_day"][:-1]), [1.0, 0.0])
        self.assertTrue(pd.isna(shifted.loc[2, "target"]))
        self.assertTrue(pd.isna(shifted.loc[2, "target_stock_hour6_22_cnt_next_day"]))

    def test_filter_uncensored_rows_keeps_only_non_censored_source_and_target_days(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"],
                "store_id": [1, 1, 1, 1],
                "product_id": [10, 10, 10, 10],
                "stock_hour6_22_cnt": [0, 1, 0, 0],
                "sale_amount": [0.1, 0.2, 0.3, 0.4],
            }
        )

        shifted = add_shifted_target_columns(frame)
        filtered = filter_uncensored_rows(shifted)

        self.assertEqual(list(filtered["dt"].dt.strftime("%Y-%m-%d")), ["2024-04-03"])
        self.assertEqual(list(filtered["stock_hour6_22_cnt"]), [0])
        self.assertEqual(list(filtered["target"]), [0.4])

    def test_drop_future_only_columns_removes_future_leakage_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-04-01"],
                "sale_amount": [0.1],
                "target": [0.2],
                "target_date_next_day": [pd.Timestamp("2024-04-02")],
                "target_stock_hour6_22_cnt_next_day": [0.0],
                "target_is_censored_next_day": [False],
            }
        )

        cleaned = drop_future_only_columns(frame)

        self.assertIn("sale_amount", cleaned.columns)
        self.assertIn("target", cleaned.columns)
        self.assertNotIn("target_date_next_day", cleaned.columns)
        self.assertNotIn("target_stock_hour6_22_cnt_next_day", cleaned.columns)
        self.assertNotIn("target_is_censored_next_day", cleaned.columns)

    def test_filter_rows_for_recovered_modeling_keeps_all_rows_with_available_target(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-04-01", "2024-04-02", "2024-04-03"],
                "stock_hour6_22_cnt": [0, 2, 0],
                "target": [0.2, 0.3, None],
            }
        )

        filtered = filter_rows_for_recovered_modeling(frame)

        self.assertEqual(list(filtered["dt"]), ["2024-04-01", "2024-04-02"])
        self.assertEqual(list(filtered["stock_hour6_22_cnt"]), [0, 2])

    def test_build_filtered_datasets_writes_filtered_parquets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            train_frame = pd.DataFrame(
                {
                    "dt": ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"],
                    "store_id": [1, 1, 1, 1],
                    "product_id": [10, 10, 10, 10],
                    "stock_hour6_22_cnt": [0, 1, 0, 0],
                    "sale_amount": [0.1, 0.2, 0.3, 0.4],
                }
            )
            val_frame = pd.DataFrame(
                {
                    "dt": ["2024-04-05", "2024-04-06", "2024-04-07"],
                    "store_id": [1, 1, 1],
                    "product_id": [10, 10, 10],
                    "stock_hour6_22_cnt": [0, 0, 1],
                    "sale_amount": [0.5, 0.6, 0.7],
                }
            )

            train_path = input_dir / "data_train.parquet"
            val_path = input_dir / "data_val.parquet"
            train_frame.to_parquet(train_path, index=False)
            val_frame.to_parquet(val_path, index=False)

            outputs = build_filtered_datasets(
                train_input_path=train_path,
                val_input_path=val_path,
                output_dir=output_dir,
                require_uncensored_days=True,
            )

            self.assertEqual(set(outputs.keys()), {"train", "val"})
            train_filtered = pd.read_parquet(outputs["train"])
            val_filtered = pd.read_parquet(outputs["val"])

            self.assertEqual(list(train_filtered["dt"].dt.strftime("%Y-%m-%d")), ["2024-04-03"])
            self.assertAlmostEqual(float(train_filtered["target"].iloc[0]), 0.4, places=6)
            self.assertNotIn("target_date_next_day", train_filtered.columns)
            self.assertNotIn("target_stock_hour6_22_cnt_next_day", train_filtered.columns)
            self.assertNotIn("target_is_censored_next_day", train_filtered.columns)
            self.assertEqual(list(val_filtered["dt"].dt.strftime("%Y-%m-%d")), ["2024-04-05"])
            self.assertAlmostEqual(float(val_filtered["target"].iloc[0]), 0.6, places=6)
            self.assertNotIn("target_date_next_day", val_filtered.columns)
            self.assertNotIn("target_stock_hour6_22_cnt_next_day", val_filtered.columns)
            self.assertNotIn("target_is_censored_next_day", val_filtered.columns)

    def test_build_filtered_datasets_defaults_to_recovered_flow_without_dropping_censored_days(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            train_frame = pd.DataFrame(
                {
                    "dt": ["2024-04-01", "2024-04-02", "2024-04-03"],
                    "store_id": [1, 1, 1],
                    "product_id": [10, 10, 10],
                    "stock_hour6_22_cnt": [0, 2, 0],
                    "sale_amount": [0.1, 0.2, 0.3],
                }
            )
            val_frame = pd.DataFrame(
                {
                    "dt": ["2024-04-04", "2024-04-05"],
                    "store_id": [1, 1],
                    "product_id": [10, 10],
                    "stock_hour6_22_cnt": [1, 0],
                    "sale_amount": [0.4, 0.5],
                }
            )

            train_path = input_dir / "data_train.parquet"
            val_path = input_dir / "data_val.parquet"
            train_frame.to_parquet(train_path, index=False)
            val_frame.to_parquet(val_path, index=False)

            outputs = build_filtered_datasets(
                train_input_path=train_path,
                val_input_path=val_path,
                output_dir=output_dir,
            )

            train_filtered = pd.read_parquet(outputs["train"])
            val_filtered = pd.read_parquet(outputs["val"])

            self.assertEqual(list(train_filtered["dt"].dt.strftime("%Y-%m-%d")), ["2024-04-01", "2024-04-02"])
            self.assertEqual(len(train_filtered["target"]), 2)
            self.assertAlmostEqual(float(train_filtered["target"].iloc[0]), 0.2, places=6)
            self.assertAlmostEqual(float(train_filtered["target"].iloc[1]), 0.3, places=6)
            self.assertEqual(list(val_filtered["dt"].dt.strftime("%Y-%m-%d")), ["2024-04-04"])
            self.assertAlmostEqual(float(val_filtered["target"].iloc[0]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
