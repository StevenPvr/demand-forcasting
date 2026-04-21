from __future__ import annotations

from pathlib import Path
import sys
import unittest
from typing import Any, cast

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.training.baselines import evaluate_statistical_baselines_macro  # noqa: E402
from praedixa.demand_forecast.training.folds import build_tuning_walk_forward_folds_by_dataset  # noqa: E402


class OptimisationBaselinesTests(unittest.TestCase):
    def test_evaluate_statistical_baselines_macro_ignores_partial_nan_predictions(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 6 + ["b"] * 6,
                "dt": list(pd.date_range("2024-06-01", periods=6, freq="D")) * 2,
                "target": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 2,
                "lag_1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0] * 2,
                "lag_7": [1.0] * 12,
                "rolling_mean_7": [1.0] * 12,
                "rolling_mean_28": [1.0] * 12,
                "same_dow_mean_4w": [1.0] * 12,
            }
        )

        folds = build_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=2)
        baseline_rows, best_baseline = evaluate_statistical_baselines_macro(
            tuning_frame,
            folds,
            absolute_target_col="target",
        )

        self.assertTrue(baseline_rows)
        self.assertEqual(best_baseline["baseline_name"], "naive_lag_1")
        self.assertTrue(bool(np.isfinite(float(cast(Any, best_baseline["mean_wape"])))))
        naive_row = next(row for row in baseline_rows if row["baseline_name"] == "naive_lag_1")
        self.assertTrue(
            all(
                cast(int, result["rows_scored"]) > 0
                for result in cast(list[dict[str, object]], naive_row["fold_wape_scores"])
            )
        )


if __name__ == "__main__":
    unittest.main()
