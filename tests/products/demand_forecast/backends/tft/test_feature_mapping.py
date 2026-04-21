from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.feature_mapping import (  # noqa: E402
    TFT_EXPLICIT_ROLE_BY_COLUMN,
    TFT_GROUP_ID_COLUMNS,
    resolve_explicit_tft_layout,
)
from praedixa.demand_forecast.backends.tft.feature_contract import (  # noqa: E402
    build_feature_contract,
    validate_feature_contract,
)


class TFTFeatureMappingTests(unittest.TestCase):
    def test_explicit_mapping_covers_gold_panel_and_runtime_helpers(self) -> None:
        self.assertEqual(len(TFT_EXPLICIT_ROLE_BY_COLUMN), 347)

    def test_group_id_is_explicit_and_stable(self) -> None:
        self.assertEqual(TFT_GROUP_ID_COLUMNS, ("series_id",))
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["series_id"], "group_id")

    def test_roles_cover_representative_columns(self) -> None:
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["location_id"], "static_categorical")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["history_available_days"], "time_varying_known_real")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["rolling_mean_7"], "time_varying_known_real")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["target_holiday_flag"], "time_varying_known_categorical")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["population_1km"], "static_real")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["target_demand_qty_d_plus_1"], "target")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["lag_1"], "exclude")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["split_bucket"], "exclude")

    def test_layout_resolution_uses_only_explicit_roles(self) -> None:
        layout = resolve_explicit_tft_layout(
            [
                "location_id",
                "population_1km",
                "target_holiday_flag",
                "rolling_mean_7",
            ]
        )

        self.assertEqual(layout["static_categoricals"], ["location_id"])
        self.assertEqual(layout["static_reals"], ["population_1km"])
        self.assertEqual(layout["time_varying_known_categoricals"], ["target_holiday_flag"])
        self.assertEqual(layout["time_varying_known_reals"], ["rolling_mean_7"])
        self.assertEqual(layout["time_varying_unknown_categoricals"], [])
        self.assertEqual(layout["time_varying_unknown_reals"], [])

    def test_feature_contract_marks_known_features_as_available(self) -> None:
        contract = build_feature_contract(["location_id", "rolling_mean_7", "target_holiday_flag"])

        self.assertEqual(contract["location_id"]["role"], "static_categorical")
        self.assertTrue(contract["location_id"]["available_at_prediction"])
        self.assertEqual(contract["rolling_mean_7"]["source_system"], "operations")
        self.assertTrue(contract["target_holiday_flag"]["available_at_prediction"])

    def test_validate_feature_contract_returns_contract_for_valid_mapping(self) -> None:
        contract = validate_feature_contract(["location_id", "population_1km", "target_holiday_flag"])

        self.assertEqual(sorted(contract.keys()), ["location_id", "population_1km", "target_holiday_flag"])


if __name__ == "__main__":
    unittest.main()
