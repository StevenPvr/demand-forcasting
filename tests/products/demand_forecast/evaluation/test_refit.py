from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import cast
import unittest
import warnings

import numpy as np
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.evaluation import refit as evaluation_refit  # noqa: E402


_concat_refit_train_with_history = cast(
    Callable[..., pd.DataFrame],
    getattr(evaluation_refit, "_concat_refit_train_with_history"),
)


class EvaluationRefitTests(unittest.TestCase):
    def test_refit_train_history_concat_drops_all_na_history_columns_before_concat(
        self,
    ) -> None:
        train_frame = pd.DataFrame(
            {
                "target": [1.0, 2.0],
                "feature": [0.1, 0.2],
                "all_na_train_col": [np.nan, np.nan],
                "metadata": ["train_a", "train_b"],
            }
        )
        eligible_history_frame = pd.DataFrame(
            {
                "target": [3.0],
                "feature": [0.3],
                "all_na_train_col": [np.nan],
                "metadata": [None],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            refit_frame = _concat_refit_train_with_history(
                train_frame=train_frame,
                eligible_history_frame=eligible_history_frame,
            )

        self.assertEqual(refit_frame.columns.tolist(), train_frame.columns.tolist())
        self.assertEqual(len(refit_frame), 3)
        self.assertEqual(refit_frame["target"].tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(pd.isna(refit_frame.loc[2, "all_na_train_col"]))
        self.assertTrue(pd.isna(refit_frame.loc[2, "metadata"]))


if __name__ == "__main__":
    unittest.main()
