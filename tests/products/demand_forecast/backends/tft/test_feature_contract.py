from __future__ import annotations

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

from praedixa.demand_forecast.backends.tft.feature_contract import (  # noqa: E402
    build_feature_contract,
    feature_available_at_prediction,
    validate_feature_contract,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import (  # noqa: E402
    TFT_EXPLICIT_ROLE_BY_COLUMN,
)


class TFTFeatureContractTests(unittest.TestCase):
    def test_known_and_static_roles_are_available_at_prediction_for_all_mapped_columns(
        self,
    ) -> None:
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

    def test_pre_close_profile_blocks_decision_day_history_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "pre-close D\\+1"):
            validate_feature_contract(
                ["rolling_mean_7_known_real"],
                decision_profile="pre_close_d_plus_1",
            )

    def test_post_close_profile_allows_decision_day_history_features(self) -> None:
        contract = validate_feature_contract(
            ["rolling_mean_7_known_real"],
            decision_profile="post_close_d_plus_1",
        )

        self.assertEqual(
            contract["rolling_mean_7_known_real"]["decision_profile"],
            "post_close_d_plus_1",
        )

    def test_lagged_operational_features_keep_point_in_time_scope(self) -> None:
        contract = build_feature_contract(
            [
                "observed_discount_amount_rolling_mean_7_known_real",
                "censor_rate_7_known_real",
                "label_quality_score_rolling_mean_7_known_real",
            ]
        )

        self.assertEqual(
            contract["observed_discount_amount_rolling_mean_7_known_real"][
                "decision_time_scope"
            ],
            "decision_day_history",
        )
        self.assertEqual(
            contract["censor_rate_7_known_real"]["source_system"], "operations"
        )
        self.assertEqual(
            contract["label_quality_score_rolling_mean_7_known_real"]["source_system"],
            "data_quality",
        )


if __name__ == "__main__":
    unittest.main()
