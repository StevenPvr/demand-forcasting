from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.global_dataset.m5 import (  # noqa: E402
    load_m5_calendar_frame,
    load_m5_sell_prices_frame,
    standardize_m5_sales_chunk,
)


def _write_m5_reference_files(root: Path) -> tuple[Path, Path]:
    calendar_path = root / "calendar.csv"
    sell_prices_path = root / "sell_prices.csv"
    calendar_path.write_text(
        "\n".join(
            [
                "date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,event_name_2,event_type_2,snap_CA,snap_TX,snap_WI",
                "2011-01-29,11101,Saturday,1,1,2011,d_1,Holiday,Cultural,,,1,0,0",
                "2011-01-30,11101,Sunday,2,1,2011,d_2,,,,,0,0,0",
            ]
        ),
        encoding="utf-8",
    )
    sell_prices_path.write_text(
        "\n".join(
            [
                "store_id,item_id,wm_yr_wk,sell_price",
                "CA_1,sku_1,11101,3.5",
            ]
        ),
        encoding="utf-8",
    )
    return calendar_path, sell_prices_path


class GlobalDatasetM5Tests(unittest.TestCase):
    def test_standardize_m5_sales_chunk_joins_calendar_and_prices(self) -> None:
        sales_chunk = pd.DataFrame(
            {
                "id": ["sku_store_validation"],
                "item_id": ["sku_1"],
                "dept_id": ["dept_1"],
                "cat_id": ["cat_1"],
                "store_id": ["CA_1"],
                "state_id": ["CA"],
                "d_1": [2],
                "d_2": [4],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            calendar_path, sell_prices_path = _write_m5_reference_files(Path(temp_dir))
            standardized = standardize_m5_sales_chunk(
                sales_chunk,
                calendar_frame=load_m5_calendar_frame(calendar_path),
                sell_prices_frame=load_m5_sell_prices_frame(sell_prices_path),
            )

        records = standardized.sort("dt").to_dicts()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["dataset_source"], "m5")
        self.assertEqual(records[0]["series_id"], "CA_1__sku_1")
        self.assertEqual(records[0]["observed_demand_qty"], 2.0)
        self.assertEqual(records[0]["avg_selling_price"], 3.5)
        self.assertTrue(records[0]["holiday_flag"])
        self.assertTrue(records[0]["snap_flag"])
        self.assertFalse(records[0]["observed_stockout_available"])


if __name__ == "__main__":
    unittest.main()
