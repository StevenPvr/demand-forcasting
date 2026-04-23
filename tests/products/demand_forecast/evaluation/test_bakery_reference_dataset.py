from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.evaluation.bakery_reference_dataset import (  # noqa: E402
    build_bakery_reference_splits_from_source_frame,
)


class BakeryReferenceDatasetTests(unittest.TestCase):
    def test_build_reference_splits_matches_dense_filter_and_chronological_split(self) -> None:
        source_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-04",
                        "2024-01-06",
                        "2024-01-07",
                        "2024-01-08",
                        "2024-01-09",
                        "2024-01-10",
                        "2024-01-03",
                    ]
                ),
                "product": [
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                ],
                "quantity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0],
            }
        )

        bundle = build_bakery_reference_splits_from_source_frame(source_frame)

        self.assertEqual(bundle.metadata["reference_product_count"], 1)
        self.assertEqual(bundle.metadata["reference_full_date_count"], 10)
        self.assertEqual(len(bundle.train_df), 7)
        self.assertEqual(len(bundle.val_df), 1)
        self.assertEqual(len(bundle.test_df), 2)
        self.assertEqual(bundle.train_df["product"].unique().tolist(), ["A"])
        self.assertEqual(bundle.train_df["date"].min().strftime("%Y-%m-%d"), "2024-01-01")
        self.assertEqual(bundle.train_df["date"].max().strftime("%Y-%m-%d"), "2024-01-07")
        self.assertEqual(bundle.val_df["date"].tolist()[0].strftime("%Y-%m-%d"), "2024-01-08")
        self.assertEqual(
            [timestamp.strftime("%Y-%m-%d") for timestamp in bundle.test_df["date"].tolist()],
            ["2024-01-09", "2024-01-10"],
        )
        missing_day = bundle.train_df.loc[bundle.train_df["date"] == pd.Timestamp("2024-01-05"), "is_missing_day"]
        self.assertEqual(missing_day.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
