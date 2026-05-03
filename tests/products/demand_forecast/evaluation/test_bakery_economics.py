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

from praedixa.demand_forecast.evaluation.bakery_economics import (  # noqa: E402
    build_simple_economic_gain_payload,
)
from praedixa.demand_forecast.evaluation.bakery_substitution import (  # noqa: E402
    substitution_family,
)


class BakeryEconomicsSubstitutionTests(unittest.TestCase):
    def test_underproduction_loss_uses_margin_after_bread_substitution(self) -> None:
        predictions_df = pd.DataFrame(
            {
                "product": ["BAGUETTE"],
                "target_date": ["2022-06-26"],
                "actual": [10.0],
                "prediction_raw": [8.0],
                "prediction_rounded": [8.0],
                "best_statistical_baseline_name": ["blend_lag_1_lag_7_50_50"],
                "best_statistical_baseline_prediction": [10.0],
                "best_statistical_baseline_abs_error": [0.0],
                "best_statistical_baseline_product_mae": [0.0],
            }
        )
        metrics_payload = {
            "per_product_metrics": {
                "BAGUETTE": {
                    "mae": 2.0,
                    "test_rows": 1,
                }
            }
        }

        payload = build_simple_economic_gain_payload(
            predictions_df,
            metrics_payload,
            model_family="unit_test",
            production_cost_ratio=0.35,
        )

        product_gain = payload["per_product_gain"]["BAGUETTE"]
        self.assertEqual(product_gain["substitution_family"], "bread")
        self.assertAlmostEqual(product_gain["substitution_recovery_rate"], 0.75)
        self.assertAlmostEqual(product_gain["unit_margin_eur"], 0.65)
        self.assertAlmostEqual(product_gain["model_underproduction_loss_eur"], 0.325)
        self.assertAlmostEqual(product_gain["model_total_loss_eur"], 0.325)
        self.assertAlmostEqual(payload["total_estimated_savings_eur_vs_best_baselines"], -0.325)

    def test_family_detection_prioritizes_sandwich_and_viennoiserie_before_bread(self) -> None:
        self.assertEqual(substitution_family("FORMULE SANDWICH"), "sandwich")
        self.assertEqual(substitution_family("PAIN AU CHOCOLAT"), "viennoiserie")
        self.assertEqual(substitution_family("PAIN BANETTE"), "bread")


if __name__ == "__main__":
    unittest.main()
