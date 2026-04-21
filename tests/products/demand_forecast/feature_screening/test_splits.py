from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.feature_screening.splits import (  # noqa: E402
    build_walk_forward_folds,
    split_chronological_train_tuning,
)


class FeatureSelectionLagSplitsTests(unittest.TestCase):
    def test_split_chronological_train_tuning_uses_unique_dates(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(
                    [
                        "2024-04-01",
                        "2024-04-01",
                        "2024-04-02",
                        "2024-04-03",
                        "2024-04-04",
                        "2024-04-05",
                        "2024-04-06",
                        "2024-04-07",
                        "2024-04-08",
                        "2024-04-09",
                    ]
                ),
                "target": np.arange(10),
            }
        )

        selection_train, tuning_holdout, metadata = split_chronological_train_tuning(frame, train_fraction=0.7)

        self.assertEqual(selection_train["dt"].nunique(), 6)
        self.assertEqual(tuning_holdout["dt"].nunique(), 3)
        self.assertLess(selection_train["dt"].max(), tuning_holdout["dt"].min())
        self.assertEqual(metadata["train_unique_dates"], 6)
        self.assertEqual(metadata["holdout_unique_dates"], 3)

    def test_build_walk_forward_folds_returns_expanding_windows(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-04-01", periods=42, freq="D"),
                "target": np.arange(42, dtype=float),
            }
        )

        folds = build_walk_forward_folds(frame, n_folds=5)

        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0]["train_dates"], 7)
        self.assertEqual(folds[0]["valid_dates"], 7)
        self.assertEqual(folds[-1]["train_dates"], 35)
        self.assertEqual(folds[-1]["valid_dates"], 7)


if __name__ == "__main__":
    unittest.main()
