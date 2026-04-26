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
        self.assertGreater(int(daily["censor_flag"].sum()), 0)
        self.assertIn("bakery", set(metadata["dataset_source"].unique()))
        self.assertTrue(
            set(SYNTHETIC_DATASET_SOURCES).issubset(
                set(metadata["dataset_source"].unique())
            )
        )
        self.assertTrue(
            bool(manifest["model_facing_policy"]["daily_source_is_stable_one_shot_csv"])
        )
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


if __name__ == "__main__":
    unittest.main()
