from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, cast
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
from praedixa.demand_forecast.backends.tft.model_common import lazy_import_tft_dependencies  # noqa: E402
from praedixa.demand_forecast.backends.tft import training_dataset as training_dataset_module  # noqa: E402
from praedixa.demand_forecast.backends.tft.training_dataset import build_training_dataset_artifacts  # noqa: E402


_BUILD_VALIDATION_FRAME = cast(
    Callable[..., pd.DataFrame],
    getattr(training_dataset_module, "_build_validation_frame"),
)


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
        validation_frame = _BUILD_VALIDATION_FRAME(
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
        full_frame = _prepared_frame()
        short_history = full_frame.loc[full_frame[GROUP_COL] == "store_1__sku_1"].copy()

        validation_frame = _BUILD_VALIDATION_FRAME(
            prepared_frame=short_history,
            max_encoder_length=8,
        )

        self.assertTrue(validation_frame.empty)

    def test_build_training_dataset_artifacts_rebases_train_time_idx_after_valid_gaps(self) -> None:
        prepared_frame = pd.DataFrame(
            {
                GROUP_COL: ["store_1__sku_1"] * 7,
                SPLIT_COL: ["train", "train", "valid", "valid", "valid", "train", "train"],
                TIME_IDX_COL: [0, 1, 2, 3, 4, 5, 6],
                "dt": pd.date_range("2024-01-01", periods=7, freq="D"),
                "location_id": ["store_1"] * 7,
                "product_id": ["sku_1"] * 7,
                "rolling_mean_7": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            }
        )

        artifacts = build_training_dataset_artifacts(
            lazy_import_tft_dependencies(),
            prepared_frame=prepared_frame,
            feature_cols=["location_id", "product_id", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            resolved_params={"max_encoder_length": 2},
        )

        self.assertEqual(artifacts.training_slice[TIME_IDX_COL].tolist(), [0, 1, 2, 3])
        self.assertEqual(len(artifacts.training_slice), 4)
        self.assertGreater(len(artifacts.training_dataset.index), 0)


if __name__ == "__main__":
    unittest.main()
