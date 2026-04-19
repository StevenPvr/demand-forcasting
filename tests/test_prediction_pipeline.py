from pathlib import Path
import json
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.prediction.pipeline import build_recovered_datasets  # noqa: E402


class PredictionPipelineTests(unittest.TestCase):
    def test_build_recovered_datasets_substitutes_censored_rows_and_writes_drop_in_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_raw_path = root / "data_train.parquet"
            val_raw_path = root / "data_val.parquet"
            selected_train_path = root / "train_selection_70_selected.parquet"
            best_params_path = root / "best_optuna_params.json"
            output_dir = root / "prediction"

            base_columns = {
                "city_id": [1, 1, 1, 1, 1],
                "store_id": [1, 1, 1, 1, 1],
                "management_group_id": [1, 1, 1, 1, 1],
                "first_category_id": [1, 1, 1, 1, 1],
                "second_category_id": [1, 1, 1, 1, 1],
                "third_category_id": [1, 1, 1, 1, 1],
                "product_id": [10, 10, 10, 10, 10],
                "stock_hour6_22_cnt": [0, 0, 0, 2, 0],
                "discount": [1.0, 1.0, 1.0, 0.9, 1.0],
                "holiday_flag": [0, 0, 0, 0, 0],
                "activity_flag": [0, 0, 0, 0, 0],
                "precpt": [0.0, 0.0, 0.0, 0.0, 0.0],
                "avg_temperature": [20.0, 20.0, 20.0, 20.0, 20.0],
                "avg_humidity": [50.0, 50.0, 50.0, 50.0, 50.0],
                "avg_wind_level": [1.0, 1.0, 1.0, 1.0, 1.0],
                "is_censored": [False, False, False, True, False],
                "sale_amount": [1.0, 1.1, 1.2, 0.3, 1.4],
            }
            for hour in range(24):
                base_columns[f"hours_sale_{hour:02d}"] = [0.1, 0.1, 0.1, 0.05, 0.1]
                base_columns[f"hours_stock_status_{hour:02d}"] = [0, 0, 0, 1 if hour >= 12 else 0, 0]

            train_frame = pd.DataFrame(
                {
                    **base_columns,
                    "dt": pd.date_range("2024-01-01", periods=5, freq="D"),
                }
            )
            val_frame = pd.DataFrame(
                {
                    **{
                        key: values[:2]
                        for key, values in base_columns.items()
                    },
                    "dt": pd.date_range("2024-01-06", periods=2, freq="D"),
                    "stock_hour6_22_cnt": [1, 0],
                    "is_censored": [True, False],
                    "sale_amount": [0.2, 1.5],
                    "discount": [0.85, 1.0],
                }
            )

            train_frame.to_parquet(train_raw_path, index=False)
            val_frame.to_parquet(val_raw_path, index=False)

            selected_train = pd.DataFrame(
                {
                    "dt": pd.date_range("2024-01-10", periods=2, freq="D"),
                    "target": [1.0, 1.1],
                    "city_id": [1, 1],
                    "store_id": [1, 1],
                    "product_id": [10, 10],
                    "discount": [1.0, 1.0],
                    "sale_amount_lag_1": [1.2, 1.3],
                    "hours_sale_00": [0.1, 0.1],
                }
            )
            selected_train.to_parquet(selected_train_path, index=False)
            best_params_path.write_text(
                json.dumps(
                    {
                        "objective": "reg:squarederror",
                        "eval_metric": "rmse",
                        "n_estimators": 25,
                        "learning_rate": 0.1,
                        "max_depth": 3,
                        "min_child_weight": 1.0,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                        "reg_lambda": 1.0,
                        "random_state": 42,
                        "n_jobs": 1,
                        "verbosity": 0,
                    }
                ),
                encoding="utf-8",
            )

            outputs = build_recovered_datasets(
                train_raw_input_path=train_raw_path,
                val_raw_input_path=val_raw_path,
                selected_train_input_path=selected_train_path,
                best_params_path=best_params_path,
                output_dir=output_dir,
            )

            self.assertEqual(set(outputs.keys()), {"train_recovered", "val_recovered", "recovery_metadata"})
            train_recovered = pd.read_parquet(outputs["train_recovered"])
            val_recovered = pd.read_parquet(outputs["val_recovered"])
            metadata = json.loads(outputs["recovery_metadata"].read_text(encoding="utf-8"))

            self.assertNotIn("observed_sale_amount", train_recovered.columns)
            self.assertNotIn("recovered_sale_amount", train_recovered.columns)
            self.assertNotIn("recovery_prediction", train_recovered.columns)
            self.assertNotIn("is_recovered_substituted", train_recovered.columns)
            self.assertEqual(len(train_recovered), len(train_frame))
            self.assertEqual(len(val_recovered), len(val_frame))
            self.assertNotEqual(train_recovered.loc[3, "sale_amount"], 0.3)
            self.assertNotEqual(val_recovered.loc[0, "sale_amount"], 0.2)

            self.assertGreater(metadata["train_summary"]["substituted_rows"], 0)
            self.assertGreater(metadata["val_summary"]["substituted_rows"], 0)


if __name__ == "__main__":
    unittest.main()
