from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.training.validation.folds import (  # noqa: E402
    build_grouped_tuning_walk_forward_folds_by_dataset,
    build_tuning_walk_forward_folds,
    build_tuning_walk_forward_folds_by_dataset,
)


class OptimisationFoldsTests(unittest.TestCase):
    def test_build_tuning_walk_forward_folds_uses_expanding_train_on_tuning_block(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-06-01", periods=18, freq="D"),
                "target": np.arange(18, dtype=float),
            }
        )

        folds = build_tuning_walk_forward_folds(tuning_frame, n_folds=5)

        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0]["train_dates"], 3)
        self.assertEqual(folds[0]["valid_dates"], 3)
        self.assertEqual(folds[-1]["train_dates"], 15)
        self.assertEqual(folds[-1]["valid_dates"], 3)

    def test_build_tuning_walk_forward_folds_by_dataset_returns_positional_indices(self) -> None:
        base_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 8 + ["b"] * 8,
                "dt": list(pd.date_range("2024-06-01", periods=8, freq="D")) * 2,
                "target": np.arange(16, dtype=float),
            }
        )
        tuning_frame = base_frame.drop(index=[1, 3, 10]).copy()

        folds = build_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=2)

        for fold in folds:
            valid_frame = cast(
                pd.DataFrame,
                tuning_frame.loc[tuning_frame["dataset_source"] == fold["dataset_source"]]
                .sort_values("dt")
                .reset_index(drop=True)
                .iloc[cast(Any, fold["valid_idx"])]
            )
            self.assertFalse(valid_frame.empty)
            self.assertTrue((valid_frame["dataset_source"] == fold["dataset_source"]).all())

    def test_build_grouped_tuning_walk_forward_folds_by_dataset_returns_fold_unions(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 8 + ["b"] * 8,
                "dt": list(pd.date_range("2024-06-01", periods=8, freq="D")) * 2,
                "target": np.arange(16, dtype=float),
            }
        ).sort_values("dt").reset_index(drop=True)

        folds = build_grouped_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=2)

        self.assertEqual(len(folds), 2)
        for fold in folds:
            valid_frame = cast(pd.DataFrame, tuning_frame.iloc[cast(Any, fold["valid_idx"])])
            self.assertEqual(set(valid_frame["dataset_source"]), {"a", "b"})
            self.assertEqual(len(cast(dict[str, object], fold["per_dataset"])), 2)


if __name__ == "__main__":
    unittest.main()
