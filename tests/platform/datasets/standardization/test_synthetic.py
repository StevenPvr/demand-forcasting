from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import cast
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.synthetic import (  # noqa: E402
    SyntheticColdStartConfig,
    generate_synthetic_cold_start_frame,
    write_synthetic_cold_start_dataset,
)


class SyntheticTests(unittest.TestCase):
    def test_generate_synthetic_cold_start_frame_returns_canonical_rows(self) -> None:
        frame = generate_synthetic_cold_start_frame(
            SyntheticColdStartConfig(
                start_date="2024-01-01",
                days=7,
                num_locations=2,
                products_per_location=2,
                random_seed=11,
            )
        )

        self.assertEqual(frame.height, 28)
        dataset_sources = frame.select("dataset_source").unique().to_series().to_list()
        self.assertEqual(dataset_sources, ["synthetic_v1"])
        self.assertTrue(frame["series_id"].n_unique() >= 4)
        min_demand = frame["observed_demand_qty"].min()
        self.assertIsNotNone(min_demand)
        self.assertTrue(cast(float, min_demand) >= 0.0)

    def test_write_synthetic_cold_start_dataset_persists_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            output_path = Path(tmp_dir_name) / "synthetic_v1.parquet"

            result = write_synthetic_cold_start_dataset(
                output_path=output_path,
                config=SyntheticColdStartConfig(
                    start_date="2024-01-01",
                    days=5,
                    num_locations=1,
                    products_per_location=1,
                    random_seed=3,
                ),
            )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
