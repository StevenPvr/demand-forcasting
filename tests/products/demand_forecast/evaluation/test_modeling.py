from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.evaluation.modeling import select_feature_columns  # noqa: E402


class EvaluationModelingTests(unittest.TestCase):
    def test_foundation_backends_use_multivariate_feature_selection(self) -> None:
        train_frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "client_id": ["client_1"] * 3,
                "location_id": ["store_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "gold_run_id": ["run_1"] * 3,
                "target": [1.0, 2.0, 3.0],
                "target_demand_qty_d_plus_1": [1.0, 2.0, 3.0],
                "target_day_of_week": [0, 1, 2],
                "rolling_mean_7": [1.0, 1.5, 2.0],
                "country_code": ["FR", "FR", "FR"],
            }
        )

        for model_backend in ("moirai", "timesfm"):
            with self.subTest(model_backend=model_backend):
                feature_cols, constant_cols, identifier_cols = select_feature_columns(
                    train_frame,
                    learning_target_col="target",
                    absolute_target_col="target_demand_qty_d_plus_1",
                    model_backend=model_backend,
                )

                self.assertIn("target_day_of_week", feature_cols)
                self.assertIn("rolling_mean_7", feature_cols)
                self.assertIn("country_code", feature_cols)
                self.assertNotIn("client_id", feature_cols)
                self.assertNotIn("location_id", feature_cols)
                self.assertNotIn("product_id", feature_cols)
                self.assertEqual(constant_cols, [])
                self.assertEqual(
                    identifier_cols,
                    ["client_id", "location_id", "product_id", "gold_run_id"],
                )


if __name__ == "__main__":
    unittest.main()
