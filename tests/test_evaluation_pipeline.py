from pathlib import Path
import json
import sys
import tempfile
import warnings
from typing import cast
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.evaluation.pipeline import (  # noqa: E402
    _build_bakery_reference_feature_frame,
    _evaluate_daily_refit_predictions,
    build_daily_walk_forward_folds,
    build_evaluation_outputs,
)
from research_praedixa.target_utils import TargetContract  # noqa: E402


class EvaluationPipelineTests(unittest.TestCase):
    def test_build_bakery_reference_feature_frame_keeps_exact_reference_universe(self) -> None:
        reference_full = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "product": ["A"] * 10,
                "quantity": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            }
        )
        reference_test = reference_full.iloc[-2:].reset_index(drop=True)
        gold_base = pd.DataFrame(
            {
                "dataset_source": ["bakery"] * 8,
                "source_partition": ["p"] * 8,
                "source_run_id": ["r"] * 8,
                "series_id": ["series_a"] * 8,
                "dt": pd.date_range("2024-01-01", periods=8, freq="D"),
                "target_dt": pd.date_range("2024-01-02", periods=8, freq="D"),
                "location_id": ["bakery_store_1"] * 8,
                "product_id": ["A"] * 8,
                "region_id": ["r1"] * 8,
                "org_group_id": ["g1"] * 8,
                "category_level_1": ["c1"] * 8,
                "category_level_2": ["c2"] * 8,
                "category_level_3": ["c3"] * 8,
                "is_observed_row": [True] * 8,
                "location_open_flag": [True] * 8,
                "product_active_flag": [True] * 8,
                "day_complete_flag": [True] * 8,
                "missing_sales_flag": [False] * 8,
                "observed_stockout_flag": [False] * 8,
                "observed_stockout_available": [True] * 8,
                "anomaly_flag": [False] * 8,
                "current_day_demand_qty": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                "true_zero_demand_flag": [False] * 8,
                "location_closed_flag": [False] * 8,
                "observed_revenue_net": [20.0] * 8,
                "avg_selling_price": [2.0] * 8,
                "observed_discount_amount": [0.0] * 8,
                "promo_flag": [False] * 8,
                "holiday_flag": [False] * 8,
                "activity_flag": [True] * 8,
                "holiday_name": [None] * 8,
                "school_holiday_flag": [False] * 8,
                "bridge_day_flag": [False] * 8,
                "pre_holiday_flag": [False] * 8,
                "post_holiday_flag": [False] * 8,
                "snap_flag": [False] * 8,
                "weather_temperature": [20.0] * 8,
                "weather_temperature_min": [18.0] * 8,
                "weather_temperature_max": [22.0] * 8,
                "weather_precipitation": [0.0] * 8,
                "weather_humidity": [0.5] * 8,
                "weather_wind_level": [1.0] * 8,
                "inflation_cpi_latest": [1.0] * 8,
                "food_cpi_latest": [1.0] * 8,
                "policy_rate_latest": [1.0] * 8,
                "gdp_growth_latest": [1.0] * 8,
                "gdp_quarterly_level_latest": [1.0] * 8,
                "gdp_current_usd_latest": [1.0] * 8,
                "unemployment_rate_latest": [1.0] * 8,
                "consumer_confidence_latest": [1.0] * 8,
                "retail_sales_index_latest": [1.0] * 8,
                "lending_interest_rate_latest": [1.0] * 8,
                "government_debt_pct_gdp_latest": [1.0] * 8,
                "client_id": ["public_bakery_sales"] * 8,
                "vertical_level_1": ["bakery"] * 8,
                "vertical_level_2": ["bakery_pastry"] * 8,
                "country_code": ["FR"] * 8,
                "region_code": ["FR-IDF"] * 8,
                "city_name": ["Paris"] * 8,
                "freshretail_rescaled_flag": [False] * 8,
                "target_scale_assumption": ["native_observed"] * 8,
                "gold_run_id": ["manual"] * 8,
                "price_available": [True] * 8,
                "weather_available": [True] * 8,
                "macro_available": [True] * 8,
            }
        )

        feature_frame = _build_bakery_reference_feature_frame(reference_full, reference_test, gold_base)

        self.assertEqual(len(feature_frame), len(reference_test))
        self.assertEqual(feature_frame["target_demand_qty_d_plus_1"].tolist(), [18.0, 19.0])
        self.assertTrue(feature_frame["target_lag_7"].notna().all())
        self.assertEqual(feature_frame["target_lag_7"].tolist(), [11.0, 12.0])
        self.assertEqual(feature_frame["weather_temperature_lag_0"].iloc[0], 20.0)
        self.assertTrue(pd.isna(feature_frame["weather_temperature_lag_0"].iloc[1]))
        self.assertEqual(feature_frame["avg_selling_price_lag_1"].tolist(), [2.0, 2.0])
        self.assertEqual(feature_frame["promo_rate_7"].tolist(), [0.0, 0.0])
        self.assertEqual(feature_frame["activity_rate_7"].tolist(), [1.0, 1.0])

    def test_build_bakery_reference_feature_frame_avoids_fragmentation_warning(self) -> None:
        reference_full = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "product": ["A"] * 10,
                "quantity": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            }
        )
        reference_test = reference_full.iloc[-2:].reset_index(drop=True)
        gold_base = pd.DataFrame(
            {
                "dataset_source": ["bakery"] * 8,
                "source_partition": ["p"] * 8,
                "source_run_id": ["r"] * 8,
                "series_id": ["series_a"] * 8,
                "dt": pd.date_range("2024-01-01", periods=8, freq="D"),
                "target_dt": pd.date_range("2024-01-02", periods=8, freq="D"),
                "location_id": ["bakery_store_1"] * 8,
                "product_id": ["A"] * 8,
                "region_id": ["r1"] * 8,
                "org_group_id": ["g1"] * 8,
                "category_level_1": ["c1"] * 8,
                "category_level_2": ["c2"] * 8,
                "category_level_3": ["c3"] * 8,
                "is_observed_row": [True] * 8,
                "location_open_flag": [True] * 8,
                "product_active_flag": [True] * 8,
                "day_complete_flag": [True] * 8,
                "missing_sales_flag": [False] * 8,
                "observed_stockout_flag": [False] * 8,
                "observed_stockout_available": [True] * 8,
                "anomaly_flag": [False] * 8,
                "current_day_demand_qty": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                "true_zero_demand_flag": [False] * 8,
                "location_closed_flag": [False] * 8,
                "observed_revenue_net": [20.0] * 8,
                "avg_selling_price": [2.0] * 8,
                "observed_discount_amount": [0.0] * 8,
                "promo_flag": [False] * 8,
                "holiday_flag": [False] * 8,
                "activity_flag": [True] * 8,
                "holiday_name": [None] * 8,
                "school_holiday_flag": [False] * 8,
                "bridge_day_flag": [False] * 8,
                "pre_holiday_flag": [False] * 8,
                "post_holiday_flag": [False] * 8,
                "snap_flag": [False] * 8,
                "weather_temperature": [20.0] * 8,
                "weather_temperature_min": [18.0] * 8,
                "weather_temperature_max": [22.0] * 8,
                "weather_precipitation": [0.0] * 8,
                "weather_humidity": [0.5] * 8,
                "weather_wind_level": [1.0] * 8,
                "inflation_cpi_latest": [1.0] * 8,
                "food_cpi_latest": [1.0] * 8,
                "policy_rate_latest": [1.0] * 8,
                "gdp_growth_latest": [1.0] * 8,
                "gdp_quarterly_level_latest": [1.0] * 8,
                "gdp_current_usd_latest": [1.0] * 8,
                "unemployment_rate_latest": [1.0] * 8,
                "consumer_confidence_latest": [1.0] * 8,
                "retail_sales_index_latest": [1.0] * 8,
                "lending_interest_rate_latest": [1.0] * 8,
                "government_debt_pct_gdp_latest": [1.0] * 8,
                "client_id": ["public_bakery_sales"] * 8,
                "vertical_level_1": ["bakery"] * 8,
                "vertical_level_2": ["bakery_pastry"] * 8,
                "country_code": ["FR"] * 8,
                "region_code": ["FR-IDF"] * 8,
                "city_name": ["Paris"] * 8,
                "freshretail_rescaled_flag": [False] * 8,
                "target_scale_assumption": ["native_observed"] * 8,
                "gold_run_id": ["manual"] * 8,
                "price_available": [True] * 8,
                "weather_available": [True] * 8,
                "macro_available": [True] * 8,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            feature_frame = _build_bakery_reference_feature_frame(reference_full, reference_test, gold_base)

        self.assertEqual(len(feature_frame), 2)
        self.assertIn("target_same_dow_mean_4w", feature_frame.columns)
        self.assertIn("target_delta_log_wow_d_plus_1", feature_frame.columns)

    def test_build_daily_walk_forward_folds_creates_one_fold_per_day(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-03"]),
                "target": [1.0, 2.0, 3.0, 4.0],
            }
        )

        folds = build_daily_walk_forward_folds(frame)

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0]["eval_date"], "2024-04-01")
        self.assertEqual(folds[0]["history_rows"], 0)
        self.assertEqual(folds[1]["history_rows"], 2)
        self.assertEqual(folds[2]["valid_rows"], 1)

    def test_build_evaluation_outputs_writes_metrics_predictions_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection_path = root / "train_selection_70_selected.parquet"
            tuning_path = root / "train_tuning_30_selected.parquet"
            val_path = root / "data_val_cleaned.parquet"
            params_path = root / "best_optuna_params.json"
            output_dir = root / "evaluation"

            train_selection = pd.DataFrame(
                {
                    "dt": pd.date_range("2024-01-01", periods=8, freq="D"),
                    "store_id": [1] * 8,
                    "product_id": [10] * 8,
                    "day_of_week": list(range(7)) + [0],
                    "sale_amount_lag_1": [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45],
                    "sale_amount_lag_7": [1.0] * 8,
                    "sale_amount_rolling_mean_7": [1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.22],
                    "target": [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5],
                }
            )
            train_tuning = pd.DataFrame(
                {
                    "dt": pd.date_range("2024-01-09", periods=4, freq="D"),
                    "store_id": [1] * 4,
                    "product_id": [10] * 4,
                    "day_of_week": [1, 2, 3, 4],
                    "sale_amount_lag_1": [1.5, 1.55, 1.6, 1.65],
                    "sale_amount_lag_7": [1.1, 1.2, 1.25, 1.3],
                    "sale_amount_rolling_mean_7": [1.28, 1.32, 1.36, 1.4],
                    "target": [1.55, 1.6, 1.65, 1.7],
                }
            )
            val_frame = pd.DataFrame(
                {
                    "dt": pd.date_range("2024-01-13", periods=3, freq="D"),
                    "store_id": [1] * 3,
                    "product_id": [10] * 3,
                    "day_of_week": [5, 6, 0],
                    "sale_amount_lag_1": [1.7, 1.75, 1.8],
                    "sale_amount_lag_7": [1.35, 1.4, 1.45],
                    "sale_amount_rolling_mean_7": [1.45, 1.5, 1.55],
                    "target": [1.75, 1.8, 1.85],
                    "extra_unused_feature": [10.0, 11.0, 12.0],
                }
            )

            train_selection.to_parquet(selection_path, index=False)
            train_tuning.to_parquet(tuning_path, index=False)
            val_frame.to_parquet(val_path, index=False)
            params_path.write_text(
                json.dumps(
                    {
                        "objective": "reg:squarederror",
                        "eval_metric": "rmse",
                        "n_estimators": 50,
                        "learning_rate": 0.1,
                        "max_depth": 3,
                        "min_child_weight": 1.0,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                        "reg_lambda": 1.0,
                        "random_state": 42,
                        "n_jobs": 1,
                        "verbosity": 0,
                    }
                ),
                encoding="utf-8",
            )

            outputs = build_evaluation_outputs(
                train_selection_input_path=selection_path,
                train_tuning_input_path=tuning_path,
                val_input_path=val_path,
                best_params_path=params_path,
                output_dir=output_dir,
            )

            self.assertEqual(
                set(outputs.keys()),
                {
                    "metrics_json",
                    "statistical_baselines_json",
                    "predictions_csv",
                    "diagnostics_json",
                    "model_card_json",
                    "final_model",
                    "evaluation_metadata",
                    "economic_gain_json",
                    "daily_refit_metrics_csv",
                    "actual_vs_predicted_plot",
                    "residuals_plot",
                },
            )
            predictions = pd.read_csv(outputs["predictions_csv"])
            daily_refit_metrics = pd.read_csv(outputs["daily_refit_metrics_csv"])
            summary = json.loads(outputs["metrics_json"].read_text(encoding="utf-8"))
            baseline_report = json.loads(outputs["statistical_baselines_json"].read_text(encoding="utf-8"))
            metadata = json.loads(outputs["evaluation_metadata"].read_text(encoding="utf-8"))
            diagnostics = json.loads(outputs["diagnostics_json"].read_text(encoding="utf-8"))
            economic_gain = json.loads(outputs["economic_gain_json"].read_text(encoding="utf-8"))

            self.assertEqual(len(predictions), 3)
            self.assertIn("prediction_raw", predictions.columns)
            self.assertIn("best_statistical_baseline_prediction", predictions.columns)
            self.assertEqual(len(daily_refit_metrics), 3)
            self.assertIn("best_iteration", daily_refit_metrics.columns)
            self.assertEqual(summary["model_family"], "FOUNDATION_XGBOOST")
            self.assertIn("overall_metrics", summary)
            self.assertIn("per_product_metrics", summary)
            self.assertIn("business_impact", summary)
            self.assertEqual(baseline_report["baseline_count"], 21)
            self.assertIn("best_baseline", baseline_report)
            self.assertEqual(
                len(baseline_report["best_baseline"]["prediction_rows"]),
                len(predictions),
            )
            self.assertEqual(metadata["target_transform"], "delta_log_wow")
            self.assertEqual(metadata["absolute_target_col"], "target")
            self.assertEqual(metadata["reconstruction_anchor_col"], "sale_amount_lag_7")
            self.assertEqual(diagnostics["evaluation_mode"], "local_parquet")
            self.assertTrue(diagnostics["daily_refit"])
            self.assertEqual(metadata["missing_in_test_feature_cols"], [])
            self.assertGreaterEqual(economic_gain["product_count"], 1)
            self.assertIn("total_model_loss_eur", economic_gain)
            self.assertIn("total_baseline_loss_eur", economic_gain)
            self.assertIn("total_estimated_savings_eur_vs_best_baselines", economic_gain)
            single_product_name = next(iter(economic_gain["per_product_gain"].keys()))
            self.assertAlmostEqual(
                economic_gain["per_product_gain"][single_product_name]["best_baseline_mae"],
                baseline_report["per_product_best_baselines"][single_product_name]["mae"],
            )
            self.assertTrue(outputs["final_model"].exists())
            self.assertTrue(outputs["actual_vs_predicted_plot"].exists())
            self.assertTrue(outputs["residuals_plot"].exists())

    def test_daily_refit_predictions_keep_actuals_aligned_with_product_and_target_date(self) -> None:
        train_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "product": ["A", "A"],
                "target": [1.0, 2.0],
                "target_abs": [1.0, 2.0],
                "feat": [0.1, 0.2],
            }
        )
        valid_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-03"]),
                "product": ["A"],
                "target": [3.0],
                "target_abs": [3.0],
                "feat": [0.3],
            }
        )
        test_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-02", "2024-02-02"]),
                "product": ["B", "A", "A", "B"],
                "target": [10.0, 20.0, 30.0, 40.0],
                "target_abs": [10.0, 20.0, 30.0, 40.0],
                "feat": [1.0, 2.0, 3.0, 4.0],
            }
        )
        target_contract = TargetContract(
            learning_target_col="target",
            absolute_target_col="target_abs",
            target_mode="identity",
            reconstruction_anchor_col=None,
        )
        train_row_counts: list[int] = []
        def _record_fit_call(**kwargs: object) -> tuple[str, int]:
            train_row_counts.append(len(cast(pd.DataFrame, kwargs["train_frame"])))
            return "model", 7

        with (
            patch("research_praedixa.evaluation.pipeline._fit_evaluation_model", side_effect=_record_fit_call),
            patch(
                "research_praedixa.evaluation.pipeline._predict_absolute",
                side_effect=lambda _model, frame, **_: frame["target_abs"].to_numpy(),
            ),
        ):
            predictions_df, daily_report_df, best_iteration = _evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={},
            )

        self.assertEqual(best_iteration, 7)
        self.assertEqual(len(daily_report_df), 2)
        self.assertEqual(train_row_counts, [2, 4])
        self.assertEqual(daily_report_df["bakery_history_rows"].tolist(), [0, 2])
        aligned_pairs = predictions_df.loc[:, ["product", "target_date", "actual"]].copy()
        expected_pairs = pd.DataFrame(
            {
                "product": ["A", "A", "B", "B"],
                "target_date": ["2024-02-01", "2024-02-02", "2024-02-01", "2024-02-02"],
                "actual": [20.0, 30.0, 10.0, 40.0],
            }
        )
        pd.testing.assert_frame_equal(aligned_pairs.reset_index(drop=True), expected_pairs)


if __name__ == "__main__":
    unittest.main()
