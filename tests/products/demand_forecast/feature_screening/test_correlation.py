from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.feature_screening.correlation import (  # noqa: E402
    filter_correlated_lag_features,
    get_lag_candidate_columns,
)


class FeatureSelectionLagCorrelationTests(unittest.TestCase):
    def test_get_lag_candidate_columns_keeps_temporal_candidates(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-04-01"]),
                "target": [0.2],
                "city_id": [1],
                "sale_amount_lag_1": [0.1],
                "sale_amount_rolling_mean_7": [0.1],
                "sale_amount_ewm_mean_7": [0.1],
            }
        )

        candidates = get_lag_candidate_columns(frame, lag_patterns=("_lag_", "_rolling_", "_ewm_"))

        self.assertEqual(
            candidates,
            [
                "sale_amount_lag_1",
                "sale_amount_rolling_mean_7",
                "sale_amount_ewm_mean_7",
            ],
        )

    def test_filter_correlated_lag_features_applies_pearson_then_spearman(self) -> None:
        target = pd.Series(np.arange(1, 51), dtype=float)
        frame = pd.DataFrame(
            {
                "target": target,
                "lag_anchor": target,
                "lag_dup": target + 0.001,
                "lag_cube": target**3,
                "lag_noise": np.where((target % 2) == 0, 1.0, 0.0),
            }
        )

        selected, report = filter_correlated_lag_features(
            frame=frame,
            candidate_cols=["lag_anchor", "lag_dup", "lag_cube", "lag_noise"],
            target_col="target",
            pearson_threshold=0.95,
            spearman_threshold=0.95,
        )

        self.assertIn("lag_anchor", selected)
        self.assertIn("lag_noise", selected)
        self.assertNotIn("lag_dup", selected)
        self.assertNotIn("lag_cube", selected)

        dropped_by = dict(zip(report["feature"], report["dropped_by"], strict=False))
        self.assertEqual(dropped_by["lag_dup"], "pearson")
        self.assertEqual(dropped_by["lag_cube"], "spearman")


if __name__ == "__main__":
    unittest.main()
