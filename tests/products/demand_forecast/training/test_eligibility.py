from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.training.validation.eligibility import (  # noqa: E402
    filter_training_eligible_rows,
)


class TrainingEligibilityTests(unittest.TestCase):
    def test_filter_training_eligible_rows_rejects_censored_low_quality_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "usable_for_training_flag": [True, True, True, False],
                "censor_flag": [False, True, False, False],
                "label_quality_score": [1.0, 1.0, 0.4, 1.0],
            }
        )

        filtered = filter_training_eligible_rows(frame, label="unit_test")

        self.assertEqual(filtered["row_id"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
