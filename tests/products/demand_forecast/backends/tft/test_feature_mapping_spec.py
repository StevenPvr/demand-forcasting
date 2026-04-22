from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.feature_mapping import TFT_EXPLICIT_ROLE_BY_COLUMN  # noqa: E402
from praedixa.demand_forecast.backends.tft.feature_mapping_spec import (  # noqa: E402
    EXCLUDED_FROM_TFT_LAG_COLUMNS,
    FEATURE_ROLE_ORDER,
)


class TFTFeatureMappingSpecTests(unittest.TestCase):
    def test_all_excluded_lag_columns_are_explicitly_marked_exclude(self) -> None:
        for column in EXCLUDED_FROM_TFT_LAG_COLUMNS:
            self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN[column], "exclude")

    def test_feature_role_order_excludes_non_feature_roles(self) -> None:
        self.assertEqual(
            FEATURE_ROLE_ORDER,
            (
                "static_categorical",
                "static_real",
                "time_varying_known_categorical",
                "time_varying_known_real",
                "time_varying_unknown_categorical",
                "time_varying_unknown_real",
            ),
        )


if __name__ == "__main__":
    unittest.main()
