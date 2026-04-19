from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.feature_engineering.pipeline import (  # noqa: E402
    build_feature_engineering_outputs,
)


class FeatureEngineeringPipelineTests(unittest.TestCase):
    def test_build_feature_engineering_outputs_writes_augmented_parquets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            train_frame = pd.DataFrame(
                {
                    "dt": ["2024-03-30", "2024-03-31", "2024-04-01", "2024-04-02"],
                    "store_id": [1, 1, 1, 1],
                    "product_id": [10, 10, 10, 10],
                    "stock_hour6_22_cnt": [0, 1, 0, 2],
                    "is_censored": [False, True, False, True],
                    "holiday_flag": [0, 1, 0, 0],
                    "sale_amount": [0.1, 0.2, 0.3, 0.4],
                    "target": [0.2, 0.3, 0.4, 0.5],
                    "hours_sale_00": [0.00, 0.10, 0.20, 0.30],
                    "hours_sale_01": [0.01, 0.11, 0.21, 0.31],
                    "hours_stock_status_00": [0, 1, 0, 1],
                    "hours_stock_status_01": [0, 0, 1, 1],
                }
            )
            val_frame = pd.DataFrame(
                {
                    "dt": ["2024-04-03", "2024-04-04"],
                    "store_id": [1, 1],
                    "product_id": [10, 10],
                    "stock_hour6_22_cnt": [0, 0],
                    "is_censored": [False, False],
                    "holiday_flag": [0, 0],
                    "sale_amount": [0.5, 0.6],
                    "target": [0.6, 0.7],
                    "hours_sale_00": [0.40, 0.50],
                    "hours_sale_01": [0.41, 0.51],
                    "hours_stock_status_00": [0, 0],
                    "hours_stock_status_01": [1, 0],
                }
            )

            train_path = input_dir / "data_train.parquet"
            val_path = input_dir / "data_val.parquet"
            train_frame.to_parquet(train_path, index=False)
            val_frame.to_parquet(val_path, index=False)

            outputs = build_feature_engineering_outputs(
                train_input_path=train_path,
                val_input_path=val_path,
                output_dir=output_dir,
                num_workers=1,
            )

            self.assertEqual(
                set(outputs.keys()),
                {"train", "val", "train_csv_sample", "val_csv_sample"},
            )
            self.assertTrue(outputs["train"].exists())
            self.assertTrue(outputs["val"].exists())
            self.assertTrue(outputs["train_csv_sample"].exists())
            self.assertTrue(outputs["val_csv_sample"].exists())

            train_featured = pd.read_parquet(outputs["train"])
            val_featured = pd.read_parquet(outputs["val"])
            train_sample = pd.read_csv(outputs["train_csv_sample"])
            val_sample = pd.read_csv(outputs["val_csv_sample"])

            self.assertIn("target", train_featured.columns)
            self.assertIn("target", val_featured.columns)
            self.assertNotIn("sale_amount", train_featured.columns)
            self.assertNotIn("sale_amount", val_featured.columns)
            self.assertIn("day_of_week", train_featured.columns)
            self.assertIn("is_post_holiday", train_featured.columns)
            self.assertIn("month_sin", train_featured.columns)
            self.assertIn("is_month_end", val_featured.columns)
            self.assertIn("sale_amount_lag_1", train_featured.columns)
            self.assertIn("sale_amount_rolling_mean_3", train_featured.columns)
            self.assertIn("stock_hour6_22_cnt_lag_1", train_featured.columns)
            self.assertIn("is_censored_rolling_sum_3", train_featured.columns)
            self.assertIn("hours_sale_00_lag_1", train_featured.columns)
            self.assertIn("hours_sale_00_lag_2", train_featured.columns)
            self.assertIn("hours_sale_00_lag_3", train_featured.columns)
            self.assertIn("hours_sale_00_lag_4", train_featured.columns)
            self.assertIn("hours_sale_00_lag_5", train_featured.columns)
            self.assertIn("hours_sale_00_lag_6", train_featured.columns)
            self.assertIn("hours_sale_00_lag_7", train_featured.columns)
            self.assertIn("hours_sale_00_rolling_mean_7", train_featured.columns)
            self.assertIn("hours_stock_status_00_lag_1", train_featured.columns)
            self.assertIn("hours_stock_status_00_lag_6", train_featured.columns)
            self.assertIn("hours_stock_status_00_rolling_mean_28", train_featured.columns)
            self.assertEqual(train_featured.loc[2, "is_post_holiday"], 1)
            self.assertEqual(val_featured.loc[val_featured.index[0], "sale_amount_lag_1"], 0.4)
            self.assertEqual(val_featured.loc[val_featured.index[0], "sale_amount_lag_2"], 0.3)
            self.assertEqual(val_featured.loc[val_featured.index[0], "stock_hour6_22_cnt_lag_1"], 2)
            self.assertEqual(val_featured.loc[val_featured.index[0], "hours_sale_00_lag_1"], 0.3)
            self.assertEqual(val_featured.loc[val_featured.index[0], "hours_sale_00_lag_2"], 0.2)
            self.assertEqual(val_featured.loc[val_featured.index[1], "hours_stock_status_01_lag_1"], 1)
            self.assertEqual(len(train_sample), 1)
            self.assertEqual(len(val_sample), 1)
            self.assertIn("target", train_sample.columns)
            self.assertIn("target", val_sample.columns)
            self.assertNotIn("sale_amount", train_sample.columns)
            self.assertNotIn("sale_amount", val_sample.columns)
            self.assertIn("hours_sale_00_lag_1", train_sample.columns)


if __name__ == "__main__":
    unittest.main()
