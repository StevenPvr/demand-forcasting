from __future__ import annotations

from pathlib import Path
import sys
from typing import cast
import unittest

import duckdb
import pandas as pd


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.training.sampling_metadata import (  # noqa: E402
    load_sampling_metadata_from_relation,
    relation_sampling_requires_top_up,
)
from praedixa.demand_forecast.training.sampling_execution import (  # noqa: E402
    sample_relation_frame,
)
from praedixa.demand_forecast.training.sampling_models import (  # noqa: E402
    RelationSamplingSpec,
)


class SamplingMetadataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
