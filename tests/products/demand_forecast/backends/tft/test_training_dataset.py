from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.frame_utils import (  # noqa: E402
    GROUP_COL,
    SPLIT_COL,
    TIME_IDX_COL,
)
from praedixa.demand_forecast.backends.tft.training_dataset import _build_validation_frame  # noqa: E402


def _prepared_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(10):
        rows.append(
            {
                GROUP_COL: "store_1__sku_1",
                SPLIT_COL: "train" if index < 7 else "valid",
                TIME_IDX_COL: index,
                "dt": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "target": float(index + 1),
            }
        )
    for index in range(14):
        rows.append(
            {
                GROUP_COL: "store_2__sku_2",
                SPLIT_COL: "train" if index < 11 else "valid",
                TIME_IDX_COL: index,
                "dt": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "target": float(index + 1),
            }
        )
    return pd.DataFrame(rows)


class TrainingDatasetTests(unittest.TestCase):
    def test_build_validation_frame_keeps_only_encoder_tail_and_validation_rows(self) -> None:
        validation_frame = _build_validation_frame(
            prepared_frame=_prepared_frame(),
            max_encoder_length=4,
        )

        grouped = {
            group_key: group_frame.reset_index(drop=True)
            for group_key, group_frame in validation_frame.groupby(GROUP_COL, sort=False)
        }
        self.assertEqual(len(grouped["store_1__sku_1"]), 7)
        self.assertEqual(len(grouped["store_2__sku_2"]), 7)
        self.assertEqual(grouped["store_1__sku_1"][SPLIT_COL].tolist()[:4], ["train"] * 4)
        self.assertEqual(grouped["store_1__sku_1"][SPLIT_COL].tolist()[4:], ["valid"] * 3)
        self.assertEqual(grouped["store_2__sku_2"][SPLIT_COL].tolist()[:4], ["train"] * 4)
        self.assertEqual(grouped["store_2__sku_2"][SPLIT_COL].tolist()[4:], ["valid"] * 3)
        self.assertEqual(grouped["store_1__sku_1"][TIME_IDX_COL].tolist(), list(range(7)))
        self.assertEqual(grouped["store_2__sku_2"][TIME_IDX_COL].tolist(), list(range(7)))

    def test_build_validation_frame_skips_groups_without_enough_history(self) -> None:
        short_history = _prepared_frame().loc[lambda frame: frame[GROUP_COL] == "store_1__sku_1"].copy()

        validation_frame = _build_validation_frame(
            prepared_frame=short_history,
            max_encoder_length=8,
        )

        self.assertTrue(validation_frame.empty)


if __name__ == "__main__":
    unittest.main()
