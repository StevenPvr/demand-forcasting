from __future__ import annotations

from dataclasses import replace
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
from praedixa.platform.datasets.synthetic_foodservice.entities import (  # noqa: E402
    build_synthetic_foodservice_entities,
)
from praedixa.platform.datasets.synthetic_foodservice.taxonomy import (  # noqa: E402
    SYNTHETIC_BAKERY_SOURCE,
    SYNTHETIC_DATASET_SOURCES,
)


class SyntheticFoodserviceEntitiesTests(unittest.TestCase):
    def test_entities_include_all_requested_foodservice_verticals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_smoke_synthetic_foodservice_config(temp_dir)
            entities = build_synthetic_foodservice_entities(config)

        self.assertEqual(len(entities.cities), config.city_count)
        self.assertEqual(len(entities.locations), config.site_count)
        self.assertEqual(
            set(entities.locations["dataset_source"].unique()),
            set(SYNTHETIC_DATASET_SOURCES),
        )
        self.assertGreater(entities.assortments["product_id"].nunique(), 100)
        self.assertTrue(entities.location_metadata["city_name"].notna().all())

    def test_bakery_product_names_are_not_reused_as_synthetic_bakery_ids(
        self,
    ) -> None:
        bakery_products = ("BAGUETTE", "CROISSANT", "PAIN AU CHOCOLAT")
        with tempfile.TemporaryDirectory() as temp_dir:
            bakery_csv = Path(temp_dir) / "Bakery sales.csv"
            pd.DataFrame(
                {
                    "article": ["baguette", "croissant", "pain au chocolat"],
                    "date": ["2024-01-01"] * 3,
                    "Quantity": [1, 1, 1],
                    "unit_price": ["1,00"] * 3,
                }
            ).to_csv(bakery_csv, index=False)
            config = replace(
                build_smoke_synthetic_foodservice_config(temp_dir),
                bakery_product_names_csv_path=bakery_csv,
            )
            entities = build_synthetic_foodservice_entities(config)

        synthetic_bakery_products = set(
            entities.products.loc[
                entities.products["dataset_source"].eq(SYNTHETIC_BAKERY_SOURCE),
                "product_id",
            ].astype(str)
        )
        synthetic_bakery_assortment = set(
            entities.assortments.loc[
                entities.assortments["dataset_source"].eq(SYNTHETIC_BAKERY_SOURCE),
                "product_id",
            ].astype(str)
        )
        self.assertEqual(set(bakery_products) & synthetic_bakery_products, set())
        self.assertEqual(set(bakery_products) & synthetic_bakery_assortment, set())
        self.assertGreaterEqual(len(synthetic_bakery_products), len(bakery_products))
        self.assertTrue(
            all(
                product_id.startswith("syn_bakery_sku_")
                for product_id in synthetic_bakery_products
            )
        )


if __name__ == "__main__":
    unittest.main()
