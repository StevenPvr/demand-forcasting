from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.data_cleaning.pipeline import (  # noqa: E402
    build_cleaned_datasets,
    drop_rows_with_missing_values,
    get_required_non_null_columns,
)


class DataCleaningPipelineTests(unittest.TestCase):
    def test_get_required_non_null_columns_targets_sale_amount_history_only(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [0.2],
                "sale_amount_lag_1": [0.1],
                "sale_amount_rolling_mean_7": [0.1],
                "sale_amount_ewm_mean_7": [0.1],
                "hours_sale_00_lag_7": [None],
            }
        )

        required = get_required_non_null_columns(frame)

        self.assertEqual(
            required,
            ["sale_amount_lag_1", "sale_amount_rolling_mean_7", "sale_amount_ewm_mean_7"],
        )

    def test_drop_rows_with_missing_values_only_removes_rows_missing_target_history(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-04-01", "2024-04-02", "2024-04-03"],
                "target": [0.2, 0.3, 0.4],
                "sale_amount_lag_1": [None, 0.2, 0.3],
                "hours_sale_00_lag_7": [None, None, 0.1],
            }
        )

        cleaned = drop_rows_with_missing_values(frame)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(list(cleaned["dt"]), ["2024-04-02", "2024-04-03"])
        self.assertEqual(list(cleaned["hours_sale_00_lag_7"].isna()), [True, False])
        self.assertFalse(cleaned["sale_amount_lag_1"].isna().any())

    def test_build_cleaned_datasets_writes_cleaned_parquets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            train_frame = pd.DataFrame(
                {
                    "dt": ["2024-04-01", "2024-04-02", "2024-04-03"],
                    "target": [0.2, 0.3, 0.4],
                    "sale_amount_lag_1": [None, 0.2, 0.3],
                    "hours_sale_00_lag_7": [None, None, 0.1],
                }
            )
            val_frame = pd.DataFrame(
                {
                    "dt": ["2024-04-04", "2024-04-05"],
                    "target": [0.5, 0.6],
                    "sale_amount_lag_1": [0.4, None],
                    "hours_sale_00_lag_7": [0.2, 0.3],
                }
            )

            train_path = input_dir / "data_train_features.parquet"
            val_path = input_dir / "data_val_features.parquet"
            train_frame.to_parquet(train_path, index=False)
            val_frame.to_parquet(val_path, index=False)

            outputs = build_cleaned_datasets(
                train_input_path=train_path,
                val_input_path=val_path,
                output_dir=output_dir,
            )

            self.assertEqual(set(outputs.keys()), {"train", "val"})

            train_cleaned = pd.read_parquet(outputs["train"])
            val_cleaned = pd.read_parquet(outputs["val"])

            self.assertEqual(list(train_cleaned["dt"]), ["2024-04-02", "2024-04-03"])
            self.assertEqual(list(val_cleaned["dt"]), ["2024-04-04"])
            self.assertFalse(train_cleaned["sale_amount_lag_1"].isna().any())
            self.assertFalse(val_cleaned["sale_amount_lag_1"].isna().any())
            self.assertEqual(int(train_cleaned["hours_sale_00_lag_7"].isna().sum()), 1)


if __name__ == "__main__":
    unittest.main()
