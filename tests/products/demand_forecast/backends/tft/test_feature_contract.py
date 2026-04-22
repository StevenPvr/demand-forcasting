from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.feature_contract import (  # noqa: E402
    build_feature_contract,
    feature_available_at_prediction,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import TFT_EXPLICIT_ROLE_BY_COLUMN  # noqa: E402


class TFTFeatureContractTests(unittest.TestCase):
    def test_known_and_static_roles_are_available_at_prediction_for_all_mapped_columns(self) -> None:
        contract = build_feature_contract(list(TFT_EXPLICIT_ROLE_BY_COLUMN.keys()))

        for column, role in TFT_EXPLICIT_ROLE_BY_COLUMN.items():
            if role in {
                "static_categorical",
                "static_real",
                "time_varying_known_categorical",
                "time_varying_known_real",
            }:
                self.assertTrue(
                    contract[column]["available_at_prediction"],
                    msg=f"{column} should be available at prediction for role {role}",
                )
            elif role not in {"exclude", "date", "target", "group_id"}:
                self.assertEqual(
                    contract[column]["available_at_prediction"],
                    feature_available_at_prediction(role),
                )


if __name__ == "__main__":
    unittest.main()
