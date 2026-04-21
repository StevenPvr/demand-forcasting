from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import cast
import unittest

import pandas as pd
import polars as pl


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.bakery import build_bakery_standardized_dataset, parse_bakery_unit_price  # noqa: E402


class GlobalDatasetBakeryTests(unittest.TestCase):
    def test_parse_bakery_unit_price_parses_french_money_strings(self) -> None:
        parsed_price = parse_bakery_unit_price("1,20 €")
        self.assertIsNotNone(parsed_price)
        self.assertAlmostEqual(cast(float, parsed_price), 1.2, places=6)
        self.assertIsNone(parse_bakery_unit_price(""))

    def test_build_bakery_standardized_dataset_aggregates_ticket_lines_daily(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "bakery.csv"
            output_path = root / "bakery.parquet"
            pd.DataFrame(
                {
                    "Unnamed: 0": [0, 1],
                    "date": ["2021-01-02", "2021-01-02"],
                    "time": ["08:38", "09:14"],
                    "ticket_number": [1, 2],
                    "article": ["BAGUETTE", "BAGUETTE"],
                    "Quantity": [1.0, 2.0],
                    "unit_price": ["0,90 €", "0,90 €"],
                }
            ).to_csv(input_path, index=False)

            build_bakery_standardized_dataset(output_path=output_path, input_path=input_path, location_id="bakery_42")
            record = pl.read_parquet(output_path).to_dicts()[0]

        self.assertEqual(record["dataset_source"], "bakery")
        self.assertEqual(record["series_id"], "bakery_42__BAGUETTE")
        self.assertEqual(record["observed_demand_qty"], 3.0)
        self.assertAlmostEqual(record["observed_revenue_net"], 2.7, places=6)


if __name__ == "__main__":
    unittest.main()
