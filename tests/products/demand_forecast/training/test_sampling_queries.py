from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.training.sampling_models import (  # noqa: E402
    RelationSamplingQuery,
    RelationSamplingSpec,
)
from praedixa.demand_forecast.training.sampling_queries import (  # noqa: E402
    build_sampling_query_for_relation,
)


class SamplingQueriesTests(unittest.TestCase):
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
                selected_columns=["dataset_source", "dt", "location_id", "product_id", "series_id"],
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
                selected_columns=["dataset_source", "dt", "location_id", "product_id", "series_id"],
            )
        )

        self.assertIn("__row_id", query)
        self.assertIn("topup_candidates", query)


if __name__ == "__main__":
    unittest.main()
