from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, patch

import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.evaluation.reference_mode import (  # noqa: E402
    _raise_if_refit_seed_not_trainable,
    _uses_foundation_hybrid_training,
    _uses_transfer_training,
    load_gold_reference_mode_frames,
    uses_bakery_reference_transfer_holdout,
)


class EvaluationReferenceModeTests(unittest.TestCase):
    def test_foundation_backends_use_xgboost_equivalent_transfer_holdout(self) -> None:
        for backend in ("xgboost", "chronos2", "moirai", "timesfm"):
            with self.subTest(backend=backend):
                self.assertTrue(uses_bakery_reference_transfer_holdout(backend))
                self.assertTrue(_uses_transfer_training(backend))
                self.assertFalse(_uses_foundation_hybrid_training(backend))

    def test_gold_reference_mode_samples_only_train_and_validation(self) -> None:
        train_frame = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-01-01"]), "series_id": ["a"]}
        )
        valid_frame = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-01-02"]), "series_id": ["a"]}
        )
        overlap_test_frame = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-04-01"]), "series_id": ["bakery_a"]}
        )
        reference_train = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-03-30"]),
                "product": ["croissant"],
                "quantity": [9.0],
                "is_missing_day": [0],
            }
        )
        reference_val = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-03-31"]),
                "product": ["croissant"],
                "quantity": [10.0],
                "is_missing_day": [0],
            }
        )
        scored_reference_test = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01"]),
                "product": ["croissant"],
                "quantity": [11.0],
                "is_missing_day": [0],
            }
        )
        gold_feature_df = pd.DataFrame({"dataset_source": ["bakery"]})
        refit_seed_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-03-30", "2024-03-31"]),
                "date": pd.to_datetime(["2024-03-30", "2024-03-31"]),
                "dataset_source": ["bakery", "bakery"],
                "product_id": ["croissant", "croissant"],
                "target_demand_qty_d_plus_1": [9.0, 10.0],
                "target_semantics": ["observed_sales", "observed_sales"],
                "censor_flag": [False, False],
                "target_source": ["observed_sales", "observed_sales"],
                "label_quality_score": [1.0, 1.0],
                "usable_for_training_flag": [True, True],
            }
        )
        scored_refit_seed = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-03-30", "2024-03-31"]),
                "product": ["croissant", "croissant"],
                "quantity": [9.0, 10.0],
            }
        )

        with (
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.load_gold_train_tuning_frames",
                return_value=(train_frame, valid_frame, "location_id", {}, {}),
            ) as mocked_load_gold,
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode._load_gold_bakery_feature_frame",
                return_value=gold_feature_df,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.build_bakery_reference_splits_from_gold",
                return_value=SimpleNamespace(
                    train_df=reference_train,
                    val_df=reference_val,
                    test_df=scored_reference_test,
                    metadata={
                        "reference_protocol": "bakery_product_arima_equivalent_from_gold",
                        "reference_test_rows": 1,
                    },
                ),
            ) as mocked_reference_bundle,
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.load_gold_bakery_overlap_test_frame",
                return_value=(
                    overlap_test_frame,
                    scored_reference_test,
                    {"reference_test_rows": 1, "overlap_test_rows": 1},
                ),
            ) as mocked_overlap,
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode._materialize_bakery_reference_feature_frame",
                return_value=(refit_seed_frame, scored_refit_seed),
            ) as mocked_materialize,
        ):
            (
                resolved_train_frame,
                resolved_valid_frame,
                resolved_test_frame,
                _,
                _,
                overlap_metadata,
            ) = load_gold_reference_mode_frames(
                duckdb_path="warehouse.duckdb",
                gold_table="gold.gold_training_matrix_d1",
                train_sample_fraction=0.05,
                tuning_sample_fraction=0.07,
                model_backend="xgboost",
            )

        mocked_load_gold.assert_called_once_with(
            duckdb_path="warehouse.duckdb",
            gold_table="gold.gold_training_matrix_d1",
            logger=ANY,
            date_col="dt",
            dataset_source_col="dataset_source",
            train_sample_fraction=0.05,
            tuning_sample_fraction=0.07,
            excluded_dataset_sources=("bakery",),
        )
        mocked_reference_bundle.assert_called_once_with(
            duckdb_path="warehouse.duckdb",
            logger=ANY,
        )
        mocked_overlap.assert_called_once_with(
            duckdb_path="warehouse.duckdb",
            reference_full_df=ANY,
            reference_test_df=scored_reference_test,
            gold_table="gold.gold_training_matrix_d1",
            gold_feature_df=gold_feature_df,
        )
        mocked_materialize.assert_called_once()
        self.assertEqual(len(resolved_train_frame), len(train_frame) + 2)
        self.assertEqual(
            int(resolved_train_frame["bakery_refit_seed_flag"].fillna(False).sum()),
            2,
        )
        self.assertIs(resolved_valid_frame, valid_frame)
        self.assertIs(resolved_test_frame, overlap_test_frame)
        self.assertEqual(overlap_metadata["reference_test_rows"], 1)
        self.assertEqual(overlap_metadata["overlap_test_rows"], 1)
        self.assertEqual(
            overlap_metadata["model_training_protocol"],
            "transfer_holdout_with_bakery_test_refits",
        )
        self.assertEqual(
            overlap_metadata["model_training_excluded_dataset_sources"], ["bakery"]
        )
        self.assertEqual(
            overlap_metadata["bakery_pretest_rows_used_for_model_training"], 2
        )
        self.assertEqual(
            overlap_metadata["bakery_pretest_seed_rows_used_for_evaluation_refits"],
            2,
        )
        self.assertEqual(
            overlap_metadata[
                "eligible_bakery_pretest_seed_rows_used_for_evaluation_refits"
            ],
            2,
        )
        self.assertEqual(
            overlap_metadata[
                "eligible_bakery_pretest_seed_days_used_for_evaluation_refits"
            ],
            2,
        )
        self.assertEqual(
            overlap_metadata["bakery_refit_policy"],
            "one_month_pretest_seed_plus_test_window_history",
        )

    def test_foundation_reference_mode_matches_xgboost_transfer_holdout_frames(
        self,
    ) -> None:
        train_frame = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-01-01"]), "series_id": ["a"]}
        )
        valid_frame = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-01-02"]), "series_id": ["a"]}
        )
        overlap_test_frame = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-04-01"]), "series_id": ["bakery_a"]}
        )
        reference_train = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-03-30"]),
                "product": ["croissant"],
                "quantity": [9.0],
                "is_missing_day": [0],
            }
        )
        reference_val = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-03-31"]),
                "product": ["croissant"],
                "quantity": [10.0],
                "is_missing_day": [0],
            }
        )
        scored_reference_test = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01"]),
                "product": ["croissant"],
                "quantity": [11.0],
                "is_missing_day": [0],
            }
        )
        gold_feature_df = pd.DataFrame({"dataset_source": ["bakery"]})
        refit_seed_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-03-30", "2024-03-31"]),
                "date": pd.to_datetime(["2024-03-30", "2024-03-31"]),
                "dataset_source": ["bakery", "bakery"],
                "product_id": ["croissant", "croissant"],
                "target_demand_qty_d_plus_1": [9.0, 10.0],
                "target_semantics": ["observed_sales", "observed_sales"],
                "censor_flag": [False, False],
                "target_source": ["observed_sales", "observed_sales"],
                "label_quality_score": [1.0, 1.0],
                "usable_for_training_flag": [True, True],
            }
        )
        scored_refit_seed = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-03-30", "2024-03-31"]),
                "product": ["croissant", "croissant"],
                "quantity": [9.0, 10.0],
            }
        )

        with (
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.load_gold_train_tuning_frames",
                return_value=(train_frame, valid_frame, "location_id", {}, {}),
            ),
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode._load_gold_bakery_feature_frame",
                return_value=gold_feature_df,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.build_bakery_reference_splits_from_gold",
                return_value=SimpleNamespace(
                    train_df=reference_train,
                    val_df=reference_val,
                    test_df=scored_reference_test,
                    metadata={
                        "reference_protocol": "bakery_product_arima_equivalent_from_gold",
                        "reference_test_rows": 1,
                    },
                ),
            ),
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.load_gold_bakery_overlap_test_frame",
                return_value=(
                    overlap_test_frame,
                    scored_reference_test,
                    {"reference_test_rows": 1, "overlap_test_rows": 1},
                ),
            ),
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode._materialize_bakery_reference_feature_frame",
                return_value=(refit_seed_frame, scored_refit_seed),
            ) as mocked_materialize,
        ):
            (
                resolved_train_frame,
                resolved_valid_frame,
                resolved_test_frame,
                _,
                _,
                overlap_metadata,
            ) = load_gold_reference_mode_frames(
                duckdb_path="warehouse.duckdb",
                gold_table="gold.gold_training_matrix_d1",
                train_sample_fraction=0.05,
                tuning_sample_fraction=0.07,
                model_backend="timesfm",
            )

        mocked_materialize.assert_called_once()
        self.assertEqual(len(resolved_train_frame), len(train_frame) + 2)
        self.assertEqual(
            int(resolved_train_frame["bakery_refit_seed_flag"].fillna(False).sum()),
            2,
        )
        self.assertIs(resolved_valid_frame, valid_frame)
        self.assertIs(resolved_test_frame, overlap_test_frame)
        self.assertEqual(
            overlap_metadata["model_training_protocol"],
            "transfer_holdout_with_bakery_test_refits",
        )
        self.assertEqual(
            overlap_metadata["model_training_excluded_dataset_sources"], ["bakery"]
        )
        self.assertEqual(
            overlap_metadata["bakery_pretest_rows_used_for_model_training"], 2
        )
        self.assertEqual(
            overlap_metadata["non_bakery_pretest_rows_used_for_model_training"], 0
        )

    def test_gold_reference_mode_rejects_untrainable_pretest_seed(self) -> None:
        seed_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-03-30"]),
                "dataset_source": ["bakery"],
                "target_demand_qty_d_plus_1": [9.0],
                "target_semantics": ["observed_sales"],
                "censor_flag": [False],
                "target_source": ["closed_or_missing_observation"],
                "label_quality_score": [0.0],
                "usable_for_training_flag": [False],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "no seed row is trainable"):
            _raise_if_refit_seed_not_trainable(seed_frame)


if __name__ == "__main__":
    unittest.main()
