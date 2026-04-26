from __future__ import annotations

from datetime import date
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
    build_default_synthetic_foodservice_config,
    build_smoke_synthetic_foodservice_config,
)


class SyntheticFoodserviceConfigTests(unittest.TestCase):
    def test_default_config_matches_reduced_source_scale(self) -> None:
        config = build_default_synthetic_foodservice_config()

        self.assertEqual(config.site_count, 180)
        self.assertEqual(config.city_count, 180)
        self.assertEqual(config.start_date, date(2024, 7, 21))
        self.assertEqual(config.end_date, date(2025, 7, 20))
        self.assertEqual(config.daily_csv_path.name, "synthetic_foodservice_daily.csv")

    def test_smoke_config_keeps_one_shot_csv_paths_under_target_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_smoke_synthetic_foodservice_config(temp_dir)

            self.assertEqual(config.daily_csv_path.parent, Path(temp_dir))
            self.assertEqual(config.site_count, 50)
            self.assertLess(
                config.end_date, build_default_synthetic_foodservice_config().end_date
            )


if __name__ == "__main__":
    unittest.main()
