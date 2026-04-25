from __future__ import annotations

import logging
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.evaluation.orchestrator_artifacts import (  # noqa: E402
    log_evaluation_completion,
)


class EvaluationLogCompletionTests(unittest.TestCase):
    def test_logs_final_global_mae_and_savings_after_product_details(self) -> None:
        logger = logging.getLogger("test_evaluation_log_completion")
        metrics_payload = {
            "overall_metrics": {
                "test_rows": 2,
                "mae": 5.25,
                "wape": 0.12,
                "bias": -0.5,
                "rmse": 7.0,
                "smape": 0.2,
            }
        }
        economic_gain_payload = {
            "total_estimated_savings_eur_vs_best_baselines": 123.45,
            "per_product_gain": {
                "BAGUETTE": {
                    "best_baseline_name": "blend_lag_1_lag_7_50_50",
                    "model_mae": 5.25,
                    "best_baseline_mae": 6.0,
                    "absolute_mae_saved_vs_best_baseline": 0.75,
                    "estimated_savings_eur_vs_best_baseline": 123.45,
                }
            },
        }

        with self.assertLogs(logger, level="INFO") as captured:
            log_evaluation_completion(
                evaluation_mode="bakery_reference_transfer_holdout",
                metrics_payload=metrics_payload,
                baseline_savings_payload={
                    "best_statistical_baseline_name": "blend_lag_1_lag_7_50_50"
                },
                economic_gain_payload=economic_gain_payload,
                logger=logger,
            )

        self.assertIn("global_mae=5.250000", captured.output[-1])
        self.assertIn("global_estimated_savings_eur=123.45", captured.output[-1])


if __name__ == "__main__":
    unittest.main()
