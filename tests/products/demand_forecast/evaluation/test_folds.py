from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.evaluation.folds import build_daily_walk_forward_folds  # noqa: E402


class EvaluationFoldsTests(unittest.TestCase):
    def test_build_daily_walk_forward_folds_creates_one_fold_per_day(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(
                    ["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-03"]
                ),
                "target": [1.0, 2.0, 3.0, 4.0],
            }
        )

        folds = build_daily_walk_forward_folds(frame)

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0]["eval_date"], "2024-04-01")
        self.assertEqual(folds[0]["history_rows"], 0)
        self.assertEqual(folds[1]["history_rows"], 2)
        self.assertEqual(folds[2]["valid_rows"], 1)

    def test_build_daily_walk_forward_folds_preserves_original_row_order(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": [
                    "2024-04-02",
                    "2024-04-01",
                    "2024-04-03",
                    "2024-04-01",
                    "2024-04-02",
                ],
                "target": [3.0, 1.0, 5.0, 2.0, 4.0],
            }
        )

        folds = build_daily_walk_forward_folds(frame)

        np.testing.assert_array_equal(
            folds[0]["valid_idx"], np.array([1, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            folds[1]["history_idx"], np.array([1, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            folds[1]["valid_idx"], np.array([0, 4], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            folds[2]["history_idx"],
            np.array([0, 1, 3, 4], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
