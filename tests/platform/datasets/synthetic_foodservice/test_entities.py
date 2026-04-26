from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
