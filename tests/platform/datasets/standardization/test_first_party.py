from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.first_party import (  # noqa: E402
    FIRST_PARTY_DAILY_TEMPLATE_COLUMNS,
    FIRST_PARTY_LOCATION_PROFILE_TEMPLATE_COLUMNS,
    FIRST_PARTY_PRODUCT_PROFILE_TEMPLATE_COLUMNS,
    build_first_party_onboarding_templates,
    standardize_first_party_daily_frame,
)


class FirstPartyTests(unittest.TestCase):
    def test_build_first_party_onboarding_templates_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)

            artifacts = build_first_party_onboarding_templates(tmp_dir)

            self.assertTrue(artifacts["daily_template"].exists())
            self.assertTrue(artifacts["location_profile_template"].exists())
            self.assertTrue(artifacts["product_profile_template"].exists())
            self.assertTrue(artifacts["readme"].exists())

            daily_columns = pd.read_csv(artifacts["daily_template"]).columns.tolist()
            location_columns = pd.read_csv(artifacts["location_profile_template"]).columns.tolist()
            product_columns = pd.read_csv(artifacts["product_profile_template"]).columns.tolist()

            self.assertEqual(daily_columns, FIRST_PARTY_DAILY_TEMPLATE_COLUMNS)
            self.assertEqual(location_columns, FIRST_PARTY_LOCATION_PROFILE_TEMPLATE_COLUMNS)
            self.assertEqual(product_columns, FIRST_PARTY_PRODUCT_PROFILE_TEMPLATE_COLUMNS)

    def test_standardize_first_party_daily_frame_fills_defaults_and_series_id(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-01-01", "2024-01-02"],
                "location_id": ["site_1", "site_1"],
                "product_id": ["sku_1", "sku_1"],
                "observed_demand_qty": [12, 15],
                "avg_selling_price": [4.5, 4.5],
                "promo_flag": [False, True],
            }
        )

        standardized = standardize_first_party_daily_frame(
            frame,
            source_partition="client_onboarding",
            source_run_id="client_a",
            silver_run_id="silver_test",
        ).sort("dt")
        records = standardized.to_dicts()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["dataset_source"], "first_party_daily")
        self.assertEqual(records[0]["series_id"], "site_1__sku_1")
        self.assertIsNone(records[0]["location_open_flag"])
        self.assertIsNone(records[0]["day_complete_flag"])
        self.assertEqual(records[0]["target_semantics"], "observed_sales")
        self.assertEqual(records[0]["censor_flag"], False)
        self.assertEqual(records[0]["calendar_month"], 1)
        self.assertEqual(records[1]["promo_flag"], True)


if __name__ == "__main__":
    unittest.main()
