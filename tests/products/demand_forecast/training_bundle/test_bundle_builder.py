from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.training_bundle.bundle_builder import (  # noqa: E402
    build_training_bundle,
)


class TrainingBundleBuilderTests(unittest.TestCase):
    def test_build_training_bundle_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train_selection_70_selected.parquet"
            tuning_path = root / "train_tuning_30_selected.parquet"
            valid_path = root / "validation_selected.parquet"
            output_dir = root / "bundle"
            train_frame = self._frame("2024-01-01", [10.0, 11.0, 12.0, 13.0], [8.0, 9.0, 10.0, 11.0], [7.0, 8.0, 9.0, 10.0], [9.0, 10.0, 11.0, 12.0])
            tuning_frame = self._frame("2024-01-05", [14.0, 15.0], [12.0, 13.0], [11.0, 12.0], [13.0, 14.0])
            valid_frame = self._frame("2024-01-07", [16.0, 17.0], [14.0, 15.0], [13.0, 14.0], [15.0, 16.0])
            self._write_inputs(train_frame, tuning_frame, valid_frame, train_path, tuning_path, valid_path)

            artifacts = build_training_bundle(
                train_input_path=train_path,
                tuning_input_path=tuning_path,
                valid_input_path=valid_path,
                output_dir=output_dir,
            )
            self._assert_artifacts_exist(artifacts)
            self._assert_feature_manifest(artifacts)
            self._assert_bundled_train(artifacts)
            self._assert_target_contract(artifacts)
            self._assert_bundle_manifest(artifacts)

    def test_build_training_bundle_materializes_directly_from_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "bundle"
            duckdb_path = root / "praedixa.duckdb"
            gold_table = "gold_daily_product_forecast_panel_d1"
            gold_frame = self._gold_frame()
            self._write_gold_table(duckdb_path, gold_table, gold_frame)

            artifacts = build_training_bundle(
                train_input_path=None,
                tuning_input_path=None,
                valid_input_path=None,
                output_dir=output_dir,
                duckdb_path=duckdb_path,
                gold_table=gold_table,
            )

            self._assert_artifacts_exist(artifacts)
            self._assert_gold_backed_bundle(artifacts)

    def _frame(
        self,
        start_date: str,
        target_values: list[float],
        rolling_values: list[float],
        lag_values: list[float],
        current_day_values: list[float],
    ) -> pd.DataFrame:
        periods = len(target_values)
        return pd.DataFrame(
            {
                "dt": pd.date_range(start_date, periods=periods, freq="D"),
                "dataset_source": ["bakery"] * periods,
                "series_id": ["store_1__sku_1"] * periods,
                "location_id": ["store_1"] * periods,
                "product_id": ["sku_1"] * periods,
                "target_demand_qty_d_plus_1": target_values,
                "rolling_mean_7": rolling_values,
                "lag_1": lag_values,
                "current_day_demand_qty": current_day_values,
            }
        )

    def _write_inputs(
        self,
        train_frame: pd.DataFrame,
        tuning_frame: pd.DataFrame,
        valid_frame: pd.DataFrame,
        train_path: Path,
        tuning_path: Path,
        valid_path: Path,
    ) -> None:
        train_frame.to_parquet(train_path, index=False)
        tuning_frame.to_parquet(tuning_path, index=False)
        valid_frame.to_parquet(valid_path, index=False)

    def _gold_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=6, freq="D"),
                "dataset_source": ["bakery"] * 6,
                "split_bucket": ["train", "train", "train", "val", "val", "test"],
                "series_id": ["store_1__sku_1"] * 6,
                "location_id": ["store_1"] * 6,
                "product_id": ["sku_1"] * 6,
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "rolling_mean_7": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                "lag_1": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "current_day_demand_qty": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )

    def _write_gold_table(
        self,
        duckdb_path: Path,
        gold_table: str,
        gold_frame: pd.DataFrame,
    ) -> None:
        connection = duckdb.connect(str(duckdb_path))
        try:
            connection.register("gold_frame", gold_frame)
            connection.execute(
                f"create table {gold_table} as select * from gold_frame"
            )
        finally:
            connection.close()

    def _assert_artifacts_exist(self, artifacts: dict[str, Path]) -> None:
        for key in ("train", "tuning", "valid", "feature_manifest", "target_contract", "bundle_manifest"):
            self.assertTrue(artifacts[key].exists())

    def _assert_feature_manifest(self, artifacts: dict[str, Path]) -> None:
        feature_manifest = json.loads(artifacts["feature_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(feature_manifest["feature_columns"], ["dataset_source", "location_id", "product_id", "rolling_mean_7"])
        self.assertEqual(feature_manifest["group_id_columns"], ["series_id"])
        self.assertEqual(feature_manifest["projection_dtypes"]["rolling_mean_7"], "float32")
        self.assertEqual(feature_manifest["projection_dtypes"]["target_demand_qty_d_plus_1"], "float32")
        self.assertNotIn("lag_1", feature_manifest["feature_columns"])
        self.assertNotIn("current_day_demand_qty", feature_manifest["feature_columns"])

    def _assert_bundled_train(self, artifacts: dict[str, Path]) -> None:
        bundled_train = pd.read_parquet(artifacts["train"])
        self.assertIn("series_id", bundled_train.columns)
        self.assertEqual(str(bundled_train["rolling_mean_7"].dtype), "float32")
        self.assertEqual(str(bundled_train["target_demand_qty_d_plus_1"].dtype), "float32")

    def _assert_target_contract(self, artifacts: dict[str, Path]) -> None:
        target_contract = json.loads(artifacts["target_contract"].read_text(encoding="utf-8"))
        self.assertEqual(target_contract["learning_target_col"], "target_demand_qty_d_plus_1")
        self.assertEqual(target_contract["absolute_target_col"], "target_demand_qty_d_plus_1")

    def _assert_bundle_manifest(self, artifacts: dict[str, Path]) -> None:
        bundle_manifest = json.loads(artifacts["bundle_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(bundle_manifest["train_rows"], 4)
        self.assertEqual(bundle_manifest["tuning_rows"], 2)
        self.assertEqual(bundle_manifest["valid_rows"], 2)
        self.assertEqual(bundle_manifest["feature_count"], 4)
        self.assertEqual(bundle_manifest["train_sha256"], self._sha256(artifacts["train"]))
        self.assertEqual(bundle_manifest["tuning_sha256"], self._sha256(artifacts["tuning"]))
        self.assertEqual(bundle_manifest["valid_sha256"], self._sha256(artifacts["valid"]))

    def _assert_gold_backed_bundle(self, artifacts: dict[str, Path]) -> None:
        bundle_manifest = json.loads(artifacts["bundle_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(bundle_manifest["train_rows"], 3)
        self.assertEqual(bundle_manifest["tuning_rows"], 2)
        self.assertEqual(bundle_manifest["valid_rows"], 1)
        self.assertIn("_cache/train.parquet", bundle_manifest["train_input_path"])
        self.assertIn("_cache/val.parquet", bundle_manifest["tuning_input_path"])
        self.assertIn("_cache/test.parquet", bundle_manifest["valid_input_path"])
        bundled_train = pd.read_parquet(artifacts["train"])
        self.assertNotIn("split_bucket", bundled_train.columns)
        self.assertNotIn("lag_1", bundled_train.columns)
        self.assertIn("rolling_mean_7", bundled_train.columns)

    def _sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
