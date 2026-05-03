from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.datasets.synthetic_foodservice.config import (  # noqa: E402
    build_smoke_synthetic_foodservice_config,
)
from praedixa.platform.datasets.synthetic_foodservice.demand import (  # noqa: E402
    ORACLE_COLUMNS,
)
from praedixa.platform.datasets.synthetic_foodservice.exporter import (  # noqa: E402
    generate_synthetic_foodservice_dataset,
)
from praedixa.platform.datasets.synthetic_foodservice.taxonomy import (  # noqa: E402
    SYNTHETIC_DATASET_SOURCES,
)


class SyntheticFoodserviceExporterTests(unittest.TestCase):
    def test_generator_writes_stable_csv_without_oracle_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_smoke_synthetic_foodservice_config(temp_dir)
            config.open_exogenous_location_metadata_csv_path.write_text(
                "dataset_source,location_id,country_code,region_code,city_name,"
                "latitude,longitude,school_zone,weather_location_label,"
                "assumption_source,drive_through_flag,delivery_flag,pickup_flag,"
                "mall_flag,transit_hub_flag,tourism_flag\n"
                "bakery,bakery_store_1,FR,IDF,Paris,48.8566,2.3522,C,manual,"
                "bakery_manual,False,False,True,False,False,False\n",
                encoding="utf-8",
            )
            artifacts = generate_synthetic_foodservice_dataset(config)
            assert artifacts.oracle_csv_path is not None
            daily = pd.read_csv(artifacts.daily_csv_path)
            oracle = pd.read_csv(artifacts.oracle_csv_path)
            metadata = pd.read_csv(artifacts.open_exogenous_location_metadata_csv_path)
            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            # Read granularity files inside with block before cleanup
            tickets_exists = config.tickets_csv_path.exists()
            lines_exists = config.lines_csv_path.exists()
            agg15_exists = config.agg_15min_csv_path.exists()
            hourly_exists = config.agg_hourly_csv_path.exists()
            halfday_exists = config.agg_halfday_csv_path.exists()
            weekly_exists = config.agg_weekly_csv_path.exists()
            monthly_exists = config.agg_monthly_csv_path.exists()
            snap_exists = config.stock_snapshots_csv_path.exists()
            inv_exists = config.inventory_movements_csv_path.exists()
            staff_exists = config.staff_schedules_csv_path.exists()
            tickets_df = (
                pd.read_csv(config.tickets_csv_path)
                if tickets_exists
                else pd.DataFrame()
            )
            lines_df = (
                pd.read_csv(config.lines_csv_path) if lines_exists else pd.DataFrame()
            )
            weekly_df = (
                pd.read_csv(config.agg_weekly_csv_path)
                if weekly_exists
                else pd.DataFrame()
            )
            staff_df = (
                pd.read_csv(config.staff_schedules_csv_path)
                if staff_exists
                else pd.DataFrame()
            )

        self.assertFalse(daily.empty)
        self.assertFalse(oracle.empty)
        self.assertTrue(
            set(SYNTHETIC_DATASET_SOURCES).issubset(
                set(daily["dataset_source"].unique())
            )
        )
        debug_columns = {
            column for column in ORACLE_COLUMNS if column.endswith("_debug")
        }
        self.assertFalse(debug_columns.intersection(daily.columns))
        self.assertTrue(set(ORACLE_COLUMNS).issubset(set(oracle.columns)))
        self.assertFalse(
            daily.duplicated(
                ["dataset_source", "dt", "location_id", "product_id"]
            ).any()
        )
        usable = daily["usable_for_training_flag"].fillna(False).astype(bool)
        missing_label = pd.to_numeric(
            daily["observed_demand_qty"], errors="coerce"
        ).isna()
        incomplete = ~daily["day_complete_flag"].fillna(False).astype(bool)
        self.assertFalse(
            bool((usable & missing_label).any()),
            "No missing label row may remain training-usable",
        )
        self.assertFalse(
            bool((usable & incomplete).any()),
            "No incomplete day may remain training-usable",
        )
        self.assertFalse(
            bool(missing_label.any()),
            "Default synthetic generation must not create corrupted missing labels",
        )
        self.assertNotIn(
            "partial_export_day",
            set(daily["target_source"].astype(str).unique()),
        )
        self.assertNotIn(
            "missing_observed_sales",
            set(daily["target_source"].astype(str).unique()),
        )
        self.assertEqual(int(daily["censor_flag"].sum()), 0)
        self.assertEqual(int(daily["observed_stockout_flag"].sum()), 0)
        self.assertEqual(
            float(pd.to_numeric(daily["observed_stockout_intensity"]).sum()),
            0.0,
        )
        self.assertIn("missing_label_rows", manifest["stats"])
        self.assertIn("training_usable_rows", manifest["stats"])
        self.assertEqual(manifest["stats"]["missing_label_rows"], 0)
        self.assertIn("bakery", set(metadata["dataset_source"].unique()))
        self.assertTrue(
            set(SYNTHETIC_DATASET_SOURCES).issubset(
                set(metadata["dataset_source"].unique())
            )
        )
        self.assertTrue(
            bool(manifest["model_facing_policy"]["daily_source_is_stable_one_shot_csv"])
        )
        self.assertEqual(manifest["config"]["target_contract"], "observed_sales")
        self.assertEqual(manifest["config"]["real_data_policy"], "none")
        self.assertEqual(manifest["config"]["panel_mode"], "complete_product_day")
        self.assertFalse(manifest["config"]["apply_corruption"])
        self.assertTrue(
            bool(
                manifest["model_facing_policy"]["medallion_generates_no_synthetic_data"]
            )
        )
        self.assertEqual(
            manifest["model_facing_policy"]["gold_split_policy"], "train_only"
        )
        self.assertEqual(
            manifest["model_facing_policy"]["allowed_gold_split_buckets"], ["train"]
        )
        self.assertEqual(
            manifest["model_facing_policy"]["forbidden_gold_split_buckets"],
            ["val", "test"],
        )

        # -- Granularity files --
        self.assertTrue(tickets_exists, "tickets CSV missing")
        self.assertTrue(lines_exists, "lines CSV missing")
        self.assertTrue(agg15_exists, "15min CSV missing")
        self.assertTrue(hourly_exists, "hourly CSV missing")
        self.assertTrue(halfday_exists, "halfday CSV missing")
        self.assertTrue(weekly_exists, "weekly CSV missing")
        self.assertTrue(monthly_exists, "monthly CSV missing")

        self.assertFalse(tickets_df.empty, "tickets should not be empty")
        self.assertFalse(lines_df.empty, "lines should not be empty")
        self.assertIn("ticket_id", tickets_df.columns)
        self.assertIn("sales_line_id", lines_df.columns)

        self.assertFalse(weekly_df.empty, "weekly should not be empty")
        self.assertLess(
            len(weekly_df), len(daily), "weekly should have fewer rows than daily"
        )

        # -- Operational files --
        self.assertTrue(snap_exists, "stock snapshots CSV missing")
        self.assertTrue(inv_exists, "inventory CSV missing")
        self.assertTrue(staff_exists, "staff schedules CSV missing")

        self.assertFalse(staff_df.empty, "staff schedules should not be empty")
        self.assertIn("staff_role", staff_df.columns)


if __name__ == "__main__":
    unittest.main()
