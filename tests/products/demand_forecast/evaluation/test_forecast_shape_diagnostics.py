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

from praedixa.demand_forecast.evaluation.forecast_shape_diagnostics import (  # noqa: E402
    build_forecast_shape_diagnostics,
)


class ForecastShapeDiagnosticsTests(unittest.TestCase):
    def test_shape_diagnostics_detects_constant_product_forecast(self) -> None:
        predictions_df = pd.DataFrame(
            {
                "product": ["A", "A", "B", "B"],
                "target_date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
                "actual": [10.0, 20.0, 10.0, 30.0],
                "prediction_raw": [15.0, 15.0, 10.0, 30.0],
                "best_statistical_baseline_prediction": [10.0, 20.0, 15.0, 15.0],
            }
        )

        payload = build_forecast_shape_diagnostics(predictions_df)

        overall = payload["overall"]
        self.assertEqual(overall["product_count"], 2)
        self.assertEqual(overall["products_low_variance_std_ratio_count"], 1)
        self.assertAlmostEqual(
            overall["product_std_ratio"]["min"],
            0.0,
        )
        self.assertAlmostEqual(payload["per_product"]["A"]["prediction_actual_std_ratio"], 0.0)
        self.assertAlmostEqual(payload["per_product"]["B"]["prediction_actual_std_ratio"], 1.0)
        self.assertAlmostEqual(payload["per_product"]["B"]["actual_prediction_correlation"], 1.0)
        self.assertEqual(payload["daily"]["day_count"], 2)
        self.assertIn("high_actual_days", payload["actual_regime"])


if __name__ == "__main__":
    unittest.main()
