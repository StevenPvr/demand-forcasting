from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import cast
import unittest

import duckdb
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

from praedixa.demand_forecast.training.sampling.metadata import (  # noqa: E402
    load_complete_series_sampling_metadata_from_relation,
    load_gold_split_sampling_metadata,
    load_sampling_metadata_from_relation,
    relation_sampling_requires_top_up,
    resolve_gold_projection_columns,
)
from praedixa.demand_forecast.training.sampling.execution import (  # noqa: E402
    sample_complete_series_relation_frame,
    sample_relation_frame,
)
from praedixa.demand_forecast.training.sampling.loaders import (  # noqa: E402
    load_parquet_train_tuning_frames,
)
from praedixa.demand_forecast.training.sampling.models import (  # noqa: E402
    GoldSplitSamplingSpec,
    RelationSamplingSpec,
)


class SamplingMetadataTests(unittest.TestCase):
    def test_gold_sampling_metadata_reports_complete_series_strategy(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        frame = pd.DataFrame(
            [
                {
                    "dataset_source": "synthetic_foodservice_qsr",
                    "split_bucket": "train",
                    "dt": date,
                    "location_id": "loc_a",
                    "product_id": product_id,
                    "client_id": f"loc_a__{product_id}",
                }
                for product_id in ("sku_1", "sku_2", "sku_3", "sku_4")
                for date in dates
            ]
        )
        connection = duckdb.connect()
        connection.register("gold_frame", frame)
        try:
            metadata = load_gold_split_sampling_metadata(
                connection,
                spec=GoldSplitSamplingSpec(
                    gold_table="gold_frame",
                    split_bucket="train",
                    date_col="dt",
                    dataset_source_col="dataset_source",
                    sample_store_col="location_id",
                    sample_fraction=0.5,
                    excluded_dataset_sources=(),
                ),
            )
        finally:
            connection.close()

        datasets = cast(dict[str, dict[str, object]], metadata["datasets"])

        self.assertEqual(
            metadata["sample_strategy"], "complete_series_by_dataset_store"
        )
        self.assertEqual(metadata["sampled_rows"], 6)
        self.assertEqual(datasets["synthetic_foodservice_qsr"]["series_count"], 4)
        self.assertEqual(
            datasets["synthetic_foodservice_qsr"]["sampled_series_count"], 2
        )

    def test_gold_projection_keeps_explicit_categorical_model_features(self) -> None:
        schema_preview = pd.DataFrame(
            {
                "dt": pd.Series([], dtype="datetime64[ns]"),
                "dataset_source": pd.Series([], dtype="object"),
                "location_id": pd.Series([], dtype="object"),
                "product_id": pd.Series([], dtype="object"),
                "client_id": pd.Series([], dtype="object"),
                "vertical_level_1": pd.Series([], dtype="object"),
                "product_family": pd.Series([], dtype="object"),
                "cold_start_bucket": pd.Series([], dtype="object"),
                "rolling_mean_7": pd.Series([], dtype="float64"),
                "free_text_note": pd.Series([], dtype="object"),
            }
        )

        projection_columns = resolve_gold_projection_columns(
            schema_preview,
            date_col="dt",
            dataset_source_col="dataset_source",
            sample_store_col="location_id",
        )

        self.assertIn("vertical_level_1", projection_columns)
        self.assertIn("product_family", projection_columns)
        self.assertIn("cold_start_bucket", projection_columns)
        self.assertIn("rolling_mean_7", projection_columns)
        self.assertNotIn("free_text_note", projection_columns)

    def test_complete_series_relation_metadata_matches_sampled_groups(self) -> None:
        frame = self._precomputed_tft_frame()
        spec = RelationSamplingSpec(
            relation_sql="select * from sample_frame",
            date_col="dt",
            dataset_source_col="dataset_source",
            sample_store_col="location_id",
            sample_fraction=0.5,
            min_samples_per_dataset=1,
        )
        connection = duckdb.connect()
        connection.register("sample_frame", frame)
        try:
            metadata = load_complete_series_sampling_metadata_from_relation(
                connection,
                spec=spec,
                available_columns=list(frame.columns),
                series_col="__tft_group_id",
            )
            sampled_frame = sample_complete_series_relation_frame(
                connection=connection,
                spec=spec,
                selected_columns=list(frame.columns),
                series_col="__tft_group_id",
            )
        finally:
            connection.close()

        datasets = cast(dict[str, dict[str, object]], metadata["datasets"])
        selected_dates_per_group = sampled_frame.groupby("__tft_group_id")[
            "dt"
        ].nunique()

        self.assertEqual(
            metadata["sample_strategy"], "complete_series_by_dataset_store"
        )
        self.assertEqual(metadata["sampled_rows"], 6)
        self.assertEqual(datasets["synthetic_foodservice_qsr"]["series_count"], 4)
        self.assertEqual(
            datasets["synthetic_foodservice_qsr"]["sampled_series_count"], 2
        )
        self.assertEqual(set(selected_dates_per_group.tolist()), {3})

    def test_parquet_loader_uses_complete_series_sampling_for_precomputed_tft_bundle(
        self,
    ) -> None:
        train_frame = self._precomputed_tft_frame()
        tuning_frame = self._precomputed_tft_frame(start_date="2024-01-04")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "optimisation_train.parquet"
            tuning_path = root / "optimisation_tuning.parquet"
            train_frame.to_parquet(train_path, index=False)
            tuning_frame.to_parquet(tuning_path, index=False)

            (
                loaded_train,
                loaded_tuning,
                _sample_store_col,
                train_metadata,
                tuning_metadata,
            ) = load_parquet_train_tuning_frames(
                train_input_path=train_path,
                tuning_input_path=tuning_path,
                logger=NoneLogger(),
                target_col="target_demand_qty_d_plus_1",
                train_sample_fraction=0.5,
                tuning_sample_fraction=0.5,
                excluded_dataset_sources=(),
            )

        self.assertEqual(
            train_metadata["sample_strategy"], "complete_series_by_dataset_store"
        )
        self.assertEqual(
            tuning_metadata["sample_strategy"], "complete_series_by_dataset_store"
        )
        self.assertEqual(len(loaded_train), 6)
        self.assertEqual(len(loaded_tuning), 6)
        self.assertEqual(
            set(loaded_train.groupby("__tft_group_id")["dt"].nunique().tolist()), {3}
        )

    def test_relation_sampling_metadata_detects_top_up_requirement(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset_source": ["fast"] * 12 + ["small"] * 4,
                "dt": pd.to_datetime(["2024-01-01"] * 16),
                "location_id": ["loc_a"] * 16,
                "product_id": [f"p_{idx}" for idx in range(16)],
                "target_demand_qty_d_plus_1": [1.0] * 16,
            }
        )
        connection = duckdb.connect()
        connection.register("sample_frame", frame)
        try:
            metadata = load_sampling_metadata_from_relation(
                connection,
                spec=RelationSamplingSpec(
                    relation_sql="select * from sample_frame",
                    date_col="dt",
                    dataset_source_col="dataset_source",
                    sample_store_col="location_id",
                    sample_fraction=0.5,
                    min_samples_per_dataset=3,
                ),
                available_columns=list(frame.columns),
            )
        finally:
            connection.close()

        self.assertTrue(relation_sampling_requires_top_up(metadata))
        datasets = cast(dict[str, dict[str, object]], metadata["datasets"])
        self.assertEqual(datasets["fast"]["sampled_rows"], 6)
        self.assertEqual(datasets["fast"]["primary_sampled_rows"], 6)
        self.assertEqual(datasets["small"]["sampled_rows"], 3)
        self.assertEqual(datasets["small"]["primary_sampled_rows"], 2)
        self.assertEqual(datasets["small"]["top_up_required"], 1)

    def test_relation_sampling_metadata_triggers_fraction_top_up_when_strata_round_down(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "dataset_source": ["fast"] * 8 + ["slow"] * 8,
                "dt": pd.to_datetime(["2024-01-01"] * 16),
                "location_id": [f"loc_{idx}" for idx in range(16)],
                "product_id": [f"p_{idx}" for idx in range(16)],
                "target_demand_qty_d_plus_1": [1.0] * 16,
            }
        )
        spec = RelationSamplingSpec(
            relation_sql="select * from sample_frame",
            date_col="dt",
            dataset_source_col="dataset_source",
            sample_store_col="location_id",
            sample_fraction=0.5,
            min_samples_per_dataset=1,
        )
        connection = duckdb.connect()
        connection.register("sample_frame", frame)
        try:
            metadata = load_sampling_metadata_from_relation(
                connection,
                spec=spec,
                available_columns=list(frame.columns),
            )
            sampled_frame = sample_relation_frame(
                connection=connection,
                spec=RelationSamplingSpec(
                    relation_sql=spec.relation_sql,
                    date_col=spec.date_col,
                    dataset_source_col=spec.dataset_source_col,
                    sample_store_col=spec.sample_store_col,
                    sample_fraction=spec.sample_fraction,
                    min_samples_per_dataset=spec.min_samples_per_dataset,
                    allow_top_up=relation_sampling_requires_top_up(metadata),
                ),
                selected_columns=list(frame.columns),
            )
        finally:
            connection.close()

        datasets = cast(dict[str, dict[str, object]], metadata["datasets"])
        self.assertTrue(relation_sampling_requires_top_up(metadata))
        self.assertEqual(datasets["fast"]["primary_sampled_rows"], 0)
        self.assertEqual(datasets["fast"]["sampled_rows"], 4)
        self.assertEqual(datasets["slow"]["primary_sampled_rows"], 0)
        self.assertEqual(datasets["slow"]["sampled_rows"], 4)
        sampled_counts = sampled_frame["dataset_source"].value_counts().to_dict()
        self.assertEqual(sampled_counts, {"fast": 4, "slow": 4})

    def _precomputed_tft_frame(self, start_date: str = "2024-01-01") -> pd.DataFrame:
        dates = pd.date_range(start_date, periods=3, freq="D")
        return pd.DataFrame(
            [
                {
                    "dataset_source": "synthetic_foodservice_qsr",
                    "dt": date,
                    "location_id": "store_a",
                    "product_id": product_id,
                    "client_id": f"store_a__{product_id}",
                    "__tft_group_id": f"store_a__{product_id}",
                    "__tft_time_idx": time_idx,
                    "target_demand_qty_d_plus_1": float(time_idx + 1),
                }
                for product_id in ("sku_1", "sku_2", "sku_3", "sku_4")
                for time_idx, date in enumerate(dates)
            ]
        )


class NoneLogger:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def log(self, *_args: object, **_kwargs: object) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
