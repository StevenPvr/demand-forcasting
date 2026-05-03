from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
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
        self.assertGreaterEqual(len(TFT_EXPLICIT_ROLE_BY_COLUMN), 252)

    def test_group_id_is_explicit_and_stable(self) -> None:
        self.assertEqual(TFT_GROUP_ID_COLUMNS, ("client_id",))
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["client_id"], "group_id")
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["series_id"], "exclude")

    def test_roles_cover_representative_columns(self) -> None:
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["location_id"], "static_categorical"
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["location_id_static_cat"], "static_categorical"
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["source_role_static_cat"], "static_categorical"
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["is_synthetic_source_static_cat"],
            "static_categorical",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["commerce_modality_static_cat"],
            "static_categorical",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["operation_type_static_cat"],
            "static_categorical",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["region_known_flag_static_cat"],
            "static_categorical",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["product_taxonomy_depth_static_real"],
            "static_real",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["history_available_days_known_real"],
            "time_varying_known_real",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN[
                "data_quality_history_missing_count_known_real"
            ],
            "time_varying_known_real",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["rolling_mean_7_known_real"],
            "time_varying_known_real",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN[
                "field_baseline_blend_lag_1_lag_7_d_plus_1_known_real"
            ],
            "time_varying_known_real",
        )
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["closure_minutes"], "exclude")
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["observed_revenue_net_rolling_mean_7"],
            "exclude",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["target_holiday_flag"],
            "time_varying_known_categorical",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["target_holiday_flag_known_cat"],
            "time_varying_known_categorical",
        )
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["site_format"], "exclude")
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["target_demand_qty_d_plus_1"], "target"
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN[
                "target_residual_field_blend_lag_1_lag_7_d_plus_1"
            ],
            "target",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["lag_1"], "time_varying_unknown_real"
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["lag_1_unknown_real"],
            "time_varying_unknown_real",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["target_lag_1_unknown_real"],
            "time_varying_unknown_real",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["promo_flag_lag_14_unknown_cat"],
            "time_varying_unknown_categorical",
        )
        self.assertEqual(
            TFT_EXPLICIT_ROLE_BY_COLUMN["observed_revenue_net_lag_1_unknown_real"],
            "time_varying_unknown_real",
        )
        self.assertEqual(TFT_EXPLICIT_ROLE_BY_COLUMN["split_bucket"], "exclude")

    def test_layout_resolution_uses_only_explicit_roles(self) -> None:
        layout = resolve_explicit_tft_layout(
            [
                "location_id_static_cat",
                "product_family_static_cat",
                "target_holiday_flag_known_cat",
                "rolling_mean_7_known_real",
            ]
        )

        self.assertEqual(
            layout["static_categoricals"],
            ["location_id_static_cat", "product_family_static_cat"],
        )
        self.assertEqual(layout["static_reals"], [])
        self.assertEqual(
            layout["time_varying_known_categoricals"],
            ["target_holiday_flag_known_cat"],
        )
        self.assertEqual(
            layout["time_varying_known_reals"], ["rolling_mean_7_known_real"]
        )
        self.assertEqual(layout["time_varying_unknown_categoricals"], [])
        self.assertEqual(layout["time_varying_unknown_reals"], [])

    def test_feature_contract_marks_known_features_as_available(self) -> None:
        contract = build_feature_contract(
            [
                "location_id_static_cat",
                "is_synthetic_source_static_cat",
                "rolling_mean_7_known_real",
                "target_holiday_flag_known_cat",
                "data_quality_history_missing_count_known_real",
                "field_baseline_blend_lag_1_lag_7_d_plus_1_known_real",
            ]
        )

        self.assertEqual(
            contract["location_id_static_cat"]["role"], "static_categorical"
        )
        self.assertTrue(contract["location_id_static_cat"]["available_at_prediction"])
        self.assertTrue(
            contract["is_synthetic_source_static_cat"]["available_at_prediction"]
        )
        self.assertEqual(
            contract["rolling_mean_7_known_real"]["source_system"], "operations"
        )
        self.assertEqual(
            contract["data_quality_history_missing_count_known_real"]["source_system"],
            "data_quality",
        )
        self.assertTrue(
            contract["target_holiday_flag_known_cat"]["available_at_prediction"]
        )
        self.assertEqual(
            contract["target_holiday_flag_known_cat"]["decision_time_scope"],
            "target_calendar_known",
        )
        self.assertEqual(
            contract["rolling_mean_7_known_real"]["max_publication_lag_hours"],
            0,
        )
        self.assertEqual(
            contract["rolling_mean_7_known_real"]["leakage_risk_level"], "low"
        )
        self.assertEqual(
            contract["field_baseline_blend_lag_1_lag_7_d_plus_1_known_real"][
                "decision_time_scope"
            ],
            "decision_day_history",
        )

    def test_validate_feature_contract_returns_contract_for_valid_mapping(self) -> None:
        contract = validate_feature_contract(
            [
                "location_id_static_cat",
                "product_family_static_cat",
                "target_holiday_flag_known_cat",
            ]
        )

        self.assertEqual(
            sorted(contract.keys()),
            [
                "location_id_static_cat",
                "product_family_static_cat",
                "target_holiday_flag_known_cat",
            ],
        )


if __name__ == "__main__":
    unittest.main()
