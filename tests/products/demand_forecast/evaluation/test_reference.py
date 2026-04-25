from __future__ import annotations

from pathlib import Path
import sys
import unittest
import warnings

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.evaluation.reference import build_bakery_reference_feature_frame  # noqa: E402


def _build_reference_full_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "product": ["A"] * 10,
            "quantity": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        }
    )


def _reference_gold_base_identity_payload() -> dict[str, list[object]]:
    return {
        "dataset_source": ["bakery"] * 8,
        "source_partition": ["p"] * 8,
        "source_run_id": ["r"] * 8,
        "dt": list(pd.date_range("2024-01-01", periods=8, freq="D")),
        "target_dt": list(pd.date_range("2024-01-02", periods=8, freq="D")),
        "location_id": ["bakery_store_1"] * 8,
        "product_id": ["A"] * 8,
        "region_id": ["r1"] * 8,
        "org_group_id": ["g1"] * 8,
        "category_level_1": ["c1"] * 8,
        "category_level_2": ["c2"] * 8,
        "category_level_3": ["c3"] * 8,
        "client_id": ["public_bakery_sales"] * 8,
        "vertical_level_1": ["bakery"] * 8,
        "vertical_level_2": ["bakery_pastry"] * 8,
        "country_code": ["FR"] * 8,
        "region_code": ["FR-IDF"] * 8,
        "city_name": ["Paris"] * 8,
        "gold_run_id": ["manual"] * 8,
    }


def _reference_gold_base_signal_payload() -> dict[str, list[object]]:
    return {
        "is_observed_row": [True] * 8,
        "day_complete_flag": [True] * 8,
        "observed_stockout_flag": [False] * 8,
        "observed_stockout_available": [True] * 8,
        "current_day_demand_qty": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "true_zero_demand_flag": [False] * 8,
        "observed_revenue_net": [20.0] * 8,
        "observed_discount_amount": [0.0] * 8,
        "promo_flag": [False] * 8,
        "holiday_flag": [False] * 8,
        "activity_flag": [True] * 8,
        "holiday_name": [None] * 8,
        "school_holiday_flag": [False] * 8,
        "bridge_day_flag": [False] * 8,
        "pre_holiday_flag": [False] * 8,
        "post_holiday_flag": [False] * 8,
        "weather_temperature": [20.0] * 8,
        "weather_precipitation": [0.0] * 8,
        "weather_humidity": [0.5] * 8,
        "weather_wind_level": [1.0] * 8,
        "lending_interest_rate_latest": [1.0] * 8,
    }


def _build_reference_gold_base_frame() -> pd.DataFrame:
    payload = _reference_gold_base_identity_payload()
    payload.update(_reference_gold_base_signal_payload())
    return pd.DataFrame(payload)


def _build_reference_gold_feature_frame() -> pd.DataFrame:
    gold_feature = _build_reference_gold_base_frame()
    gold_feature["target_day_of_week"] = list(range(8))
    gold_feature["rolling_mean_7"] = [100.0 + i for i in range(8)]
    gold_feature["weather_temperature_lag_0"] = [200.0 + i for i in range(8)]
    gold_feature["weather_temperature_lag_1"] = [190.0 + i for i in range(8)]
    gold_feature["weather_precipitation_lag_0"] = [0.5] * 8
    return gold_feature.drop(columns=["weather_temperature", "weather_precipitation"])


def _build_reference_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_full = _build_reference_full_frame()
    reference_test = reference_full.iloc[-2:].reset_index(drop=True)
    return reference_full, reference_test, _build_reference_gold_base_frame()


class EvaluationReferenceTests(unittest.TestCase):
    def test_build_bakery_reference_feature_frame_keeps_exact_reference_universe(self) -> None:
        reference_full, reference_test, gold_base = _build_reference_inputs()

        feature_frame = build_bakery_reference_feature_frame(reference_full, reference_test, gold_base)

        self.assertEqual(len(feature_frame), len(reference_test))
        self.assertEqual(feature_frame["target_demand_qty_d_plus_1"].tolist(), [18.0, 19.0])
        self.assertTrue(feature_frame["target_lag_7"].notna().all())
        self.assertEqual(feature_frame["target_lag_7"].tolist(), [11.0, 12.0])
        self.assertEqual(feature_frame["weather_temperature_lag_0"].iloc[0], 20.0)
        self.assertTrue(pd.isna(feature_frame["weather_temperature_lag_0"].iloc[1]))
        self.assertNotIn("avg_selling_price_lag_1", feature_frame.columns)
        self.assertEqual(feature_frame["client_id"].tolist(), ["public_bakery_sales", "public_bakery_sales"])
        self.assertEqual(feature_frame["promo_rate_7"].tolist(), [0.0, 0.0])
        self.assertEqual(feature_frame["activity_rate_7"].tolist(), [1.0, 1.0])

    def test_build_bakery_reference_feature_frame_preserves_gold_feature_columns(self) -> None:
        reference_full = _build_reference_full_frame()
        reference_test = reference_full.iloc[-2:].reset_index(drop=True)
        gold_feature = _build_reference_gold_feature_frame()

        feature_frame = build_bakery_reference_feature_frame(reference_full, reference_test, gold_feature)

        self.assertIn("target_day_of_week", feature_frame.columns)
        self.assertIn("rolling_mean_7", feature_frame.columns)
        self.assertEqual(feature_frame["target_day_of_week"].iloc[0], 7)
        self.assertEqual(feature_frame["rolling_mean_7"].iloc[0], 107.0)
        self.assertEqual(feature_frame["weather_temperature_lag_0"].iloc[0], 207.0)
        self.assertTrue(pd.isna(feature_frame["weather_temperature_lag_0"].iloc[1]))

    def test_build_bakery_reference_feature_frame_synthesizes_missing_client_id(self) -> None:
        reference_full, reference_test, gold_base = _build_reference_inputs()
        gold_base = gold_base.drop(columns=["client_id"])

        feature_frame = build_bakery_reference_feature_frame(reference_full, reference_test, gold_base)

        self.assertEqual(feature_frame["client_id"].tolist(), ["bakery_store_1__A", "bakery_store_1__A"])

    def test_build_bakery_reference_feature_frame_avoids_fragmentation_warning(self) -> None:
        reference_full, reference_test, gold_base = _build_reference_inputs()

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            feature_frame = build_bakery_reference_feature_frame(reference_full, reference_test, gold_base)

        self.assertEqual(len(feature_frame), 2)
        self.assertIn("target_same_dow_mean_4w", feature_frame.columns)
        self.assertNotIn("target_delta_log_wow_d_plus_1", feature_frame.columns)


if __name__ == "__main__":
    unittest.main()
