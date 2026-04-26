from __future__ import annotations

from pathlib import Path
import sys
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

from praedixa.demand_forecast.training.sampling.models import (  # noqa: E402
    RelationSamplingQuery,
    RelationSamplingSpec,
)
from praedixa.demand_forecast.training.sampling.queries import (  # noqa: E402
    build_complete_series_sampling_query_for_relation,
    build_gold_split_sampling_query,
    build_sampling_query_for_relation,
)


class SamplingQueriesTests(unittest.TestCase):
    def test_gold_sampling_keeps_complete_series_across_dates(self) -> None:
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
                    "usable_for_training_flag": True,
                    "censor_flag": False,
                    "label_quality_score": 1.0,
                    "target_source": "observed_sales",
                }
                for product_id in ("sku_1", "sku_2", "sku_3", "sku_4")
                for date in dates
            ]
        )
        connection = duckdb.connect()
        connection.register("gold_frame", frame)
        try:
            sampled = connection.execute(
                build_gold_split_sampling_query(
                    gold_table="gold_frame",
                    split_bucket="train",
                    date_col="dt",
                    dataset_source_col="dataset_source",
                    sample_store_col="location_id",
                    sample_fraction=0.5,
                    selected_columns=list(frame.columns),
                )
            ).fetchdf()
        finally:
            connection.close()

        selected_products = sorted(sampled["product_id"].unique().tolist())
        selected_dates_per_product = sampled.groupby("product_id")["dt"].nunique()

        self.assertEqual(len(selected_products), 2)
        self.assertEqual(len(sampled), 6)
        self.assertEqual(set(selected_dates_per_product.tolist()), {3})
        self.assertEqual(sampled["dt"].nunique(), 3)
        self.assertIn("rn_sample", sampled.columns)

    def test_sampling_query_skips_top_up_ctes_when_not_needed(self) -> None:
        query = build_sampling_query_for_relation(
            query=RelationSamplingQuery(
                spec=RelationSamplingSpec(
                    relation_sql="select * from sample_frame",
                    date_col="dt",
                    dataset_source_col="dataset_source",
                    sample_store_col="location_id",
                    sample_fraction=0.1,
                    allow_top_up=False,
                ),
                selected_columns=[
                    "dataset_source",
                    "dt",
                    "location_id",
                    "product_id",
                    "client_id",
                ],
            )
        )

        self.assertNotIn("__row_id", query)
        self.assertNotIn("topup_candidates", query)
        self.assertNotIn("from sampled\norder by", query.lower())

    def test_sampling_query_keeps_top_up_ctes_when_required(self) -> None:
        query = build_sampling_query_for_relation(
            query=RelationSamplingQuery(
                spec=RelationSamplingSpec(
                    relation_sql="select * from sample_frame",
                    date_col="dt",
                    dataset_source_col="dataset_source",
                    sample_store_col="location_id",
                    sample_fraction=0.1,
                    allow_top_up=True,
                ),
                selected_columns=[
                    "dataset_source",
                    "dt",
                    "location_id",
                    "product_id",
                    "client_id",
                ],
            )
        )

        self.assertIn("__row_id", query)
        self.assertIn("topup_candidates", query)

    def test_complete_series_relation_sampling_keeps_full_groups(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        frame = pd.DataFrame(
            [
                {
                    "dataset_source": "m5_forecasting_accuracy",
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
        connection = duckdb.connect()
        connection.register("sample_frame", frame)
        try:
            sampled = connection.execute(
                build_complete_series_sampling_query_for_relation(
                    query=RelationSamplingQuery(
                        spec=RelationSamplingSpec(
                            relation_sql="select * from sample_frame",
                            date_col="dt",
                            dataset_source_col="dataset_source",
                            sample_store_col="location_id",
                            sample_fraction=0.5,
                        ),
                        selected_columns=list(frame.columns),
                    ),
                    series_col="__tft_group_id",
                )
            ).fetchdf()
        finally:
            connection.close()

        selected_groups = sorted(sampled["__tft_group_id"].unique().tolist())
        selected_dates_per_group = sampled.groupby("__tft_group_id")["dt"].nunique()

        self.assertEqual(len(selected_groups), 2)
        self.assertEqual(len(sampled), 6)
        self.assertEqual(set(selected_dates_per_group.tolist()), {3})
        self.assertNotIn("rn_sample", sampled.columns)

    def test_gold_train_sampling_requires_training_eligible_rows(self) -> None:
        query = build_gold_split_sampling_query(
            gold_table="gold.gold_daily_product_forecast_panel_d1",
            split_bucket="train",
            date_col="dt",
            dataset_source_col="dataset_source",
            sample_store_col="location_id",
            sample_fraction=1.0,
            selected_columns=[
                "dataset_source",
                "dt",
                "location_id",
                "product_id",
                "client_id",
                "usable_for_training_flag",
                "censor_flag",
                "label_quality_score",
                "target_source",
            ],
        )

        self.assertIn("coalesce(usable_for_training_flag, false)", query)
        self.assertIn("not coalesce(censor_flag, false)", query)
        self.assertIn("coalesce(label_quality_score, 0.0) >= 0.750000", query)
        self.assertIn(
            "coalesce(target_source, '') not in ('closed_or_missing_observation', 'dense_calendar_zero_fill')",
            query,
        )
        self.assertIn("dataset_source not in ('bakery')", query)

    def test_gold_val_sampling_requires_training_eligible_rows(self) -> None:
        query = build_gold_split_sampling_query(
            gold_table="gold.gold_daily_product_forecast_panel_d1",
            split_bucket="val",
            date_col="dt",
            dataset_source_col="dataset_source",
            sample_store_col="location_id",
            sample_fraction=1.0,
            selected_columns=[
                "dataset_source",
                "dt",
                "location_id",
                "product_id",
                "client_id",
                "usable_for_training_flag",
                "censor_flag",
                "label_quality_score",
                "target_source",
            ],
        )

        self.assertIn("coalesce(usable_for_training_flag, false)", query)
        self.assertNotIn("'val' <> 'train'", query)

    def test_gold_train_sampling_allows_minimal_projection_without_label_quality(
        self,
    ) -> None:
        query = build_gold_split_sampling_query(
            gold_table="gold.gold_daily_product_forecast_panel_d1",
            split_bucket="train",
            date_col="dt",
            dataset_source_col="dataset_source",
            sample_store_col="location_id",
            sample_fraction=1.0,
            selected_columns=["dataset_source", "dt", "location_id", "client_id"],
        )

        self.assertIn("and true", query)
        self.assertNotIn("coalesce(usable_for_training_flag, false)", query)


if __name__ == "__main__":
    unittest.main()
