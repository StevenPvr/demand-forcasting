from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.training.shared.economic_objective import (  # noqa: E402
    DEFAULT_ASYMMETRIC_ECONOMIC_OBJECTIVE_CONFIG,
    EconomicCostProfile,
    SegmentedEconomicObjectiveConfig,
    apply_economic_decision_calibration_to_predictions,
    fit_economic_decision_calibration,
    resolve_economic_segments,
    score_asymmetric_forecast_decision,
    score_segmented_asymmetric_forecast_decision,
)


class AsymmetricEconomicObjectiveTest(unittest.TestCase):
    def test_positive_bias_costs_more_than_mild_negative_bias(self) -> None:
        actual = np.asarray([100.0, 100.0, 100.0])
        over_forecast = np.asarray([110.0, 110.0, 110.0])
        mild_under_forecast = np.asarray([90.0, 90.0, 90.0])

        over_score = score_asymmetric_forecast_decision(actual, over_forecast)
        under_score = score_asymmetric_forecast_decision(actual, mild_under_forecast)

        self.assertGreater(over_score["decision_loss"], under_score["decision_loss"])
        self.assertGreater(over_score["positive_bias_penalty"], 0.0)
        self.assertEqual(under_score["positive_bias_penalty"], 0.0)

    def test_severe_negative_bias_is_penalized(self) -> None:
        actual = np.asarray([100.0, 100.0, 100.0])
        severe_under_forecast = np.asarray([50.0, 50.0, 50.0])

        score = score_asymmetric_forecast_decision(actual, severe_under_forecast)

        self.assertGreater(score["severe_underproduction_units"], 0.0)
        self.assertGreater(score["severe_negative_bias_penalty"], 0.0)

    def test_zero_actual_volume_stays_finite_and_deterministic(self) -> None:
        actual = np.asarray([0.0, 0.0, 0.0])
        prediction = np.asarray([2.0, 0.0, 1.0])

        first_score = score_asymmetric_forecast_decision(actual, prediction)
        second_score = score_asymmetric_forecast_decision(actual, prediction)

        self.assertEqual(first_score, second_score)
        self.assertTrue(np.isfinite(first_score["decision_loss"]))
        self.assertEqual(
            first_score["normalized_bias"],
            1.0,
        )

    def test_defaults_encode_prudent_asymmetric_costs(self) -> None:
        config = DEFAULT_ASYMMETRIC_ECONOMIC_OBJECTIVE_CONFIG

        self.assertGreater(
            config.overproduction_unit_cost,
            config.mild_underproduction_unit_cost,
        )
        self.assertGreater(
            config.severe_underproduction_unit_cost,
            config.mild_underproduction_unit_cost,
        )
        self.assertLess(config.max_allowed_negative_bias, 0.0)

    def test_segment_resolution_prefers_product_override_then_family(self) -> None:
        config = SegmentedEconomicObjectiveConfig(
            family_profiles={"bread": EconomicCostProfile(), "default": EconomicCostProfile()},
            product_profiles={"SKU_SPECIAL": EconomicCostProfile(severe_underproduction_unit_cost=4.0)},
        )
        frame = pd.DataFrame(
            {
                "product_id": ["SKU_SPECIAL", "SKU_BREAD", "UNKNOWN"],
                "product_family": ["bread", "bread", None],
                "product": ["Whatever", "Whatever", "BAGUETTE"],
            }
        )

        segments = resolve_economic_segments(frame, config=config)

        self.assertEqual(segments.tolist(), ["product:SKU_SPECIAL", "bread", "bread"])

    def test_segmented_objective_applies_family_specific_costs(self) -> None:
        frame = pd.DataFrame({"product_family": ["bread", "sandwich"]})
        actual = np.asarray([100.0, 100.0])
        prediction = np.asarray([90.0, 90.0])

        score = score_segmented_asymmetric_forecast_decision(
            actual,
            prediction,
            frame=frame,
        )
        segment_scores = cast(dict[str, dict[str, float]], score["segment_decision_loss"])

        self.assertGreater(
            segment_scores["sandwich"]["decision_loss"],
            segment_scores["bread"]["decision_loss"],
        )

    def test_economic_calibration_prefers_negative_shift_when_it_reduces_loss(self) -> None:
        frame = pd.DataFrame(
            {"product_family": ["bread"] * 80, "product": ["BAGUETTE"] * 80}
        )
        actual = np.full(80, 100.0)
        prediction = np.full(80, 102.0)

        calibration = fit_economic_decision_calibration(frame, actual, prediction)
        global_entry = cast(dict[str, Any], calibration["global"])

        self.assertTrue(calibration["applied"])
        self.assertTrue(global_entry["applied"])
        self.assertLess(global_entry["intercept"], 0.0)

        predictions_df = pd.DataFrame(
            {
                "product": ["BAGUETTE"],
                "product_family": ["bread"],
                "prediction_raw": [102.0],
                "prediction_rounded": [102.0],
            }
        )
        calibrated = apply_economic_decision_calibration_to_predictions(
            predictions_df,
            calibration=calibration,
        )
        self.assertLess(float(calibrated["prediction_raw"].iloc[0]), 102.0)


if __name__ == "__main__":
    unittest.main()
