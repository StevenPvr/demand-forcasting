from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.evaluation.bakery_metrics import (  # noqa: E402
    bias_score,
    mae_score,
)


class BakeryMetricTests(unittest.TestCase):
    def test_mae_does_not_cancel_symmetric_forecast_errors(self) -> None:
        actual = np.array([10.0, 10.0])
        predicted = np.array([0.0, 20.0])

        self.assertEqual(mae_score(actual, predicted), 10.0)
        self.assertEqual(bias_score(actual, predicted), 0.0)


if __name__ == "__main__":
    unittest.main()
