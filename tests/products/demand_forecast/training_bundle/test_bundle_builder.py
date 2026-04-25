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
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

    def test_build_training_bundle_smoke_keeps_complete_series_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "bundle"
            duckdb_path = root / "praedixa.duckdb"
            gold_table = "gold_daily_product_forecast_panel_d1"
            self._write_gold_table(duckdb_path, gold_table, self._gold_smoke_frame())

            artifacts = build_training_bundle(
                train_input_path=None,
                tuning_input_path=None,
                valid_input_path=None,
                output_dir=output_dir,
                duckdb_path=duckdb_path,
                gold_table=gold_table,
                smoke_series_limit=1,
                smoke_dataset_source="freshretail_lt",
                smoke_min_train_rows=4,
                smoke_min_tuning_rows=2,
                smoke_min_valid_rows=1,
            )

            bundled_train = pd.read_parquet(artifacts["train"])
            bundled_tuning = pd.read_parquet(artifacts["tuning"])
            bundled_valid = pd.read_parquet(artifacts["valid"])

            self.assertEqual(sorted(bundled_train["client_id"].unique().tolist()), ["store_1__sku_1"])
            self.assertEqual(sorted(bundled_tuning["client_id"].unique().tolist()), ["store_1__sku_1"])
            self.assertEqual(sorted(bundled_valid["client_id"].unique().tolist()), ["store_1__sku_1"])
            self.assertEqual(len(bundled_train), 4)
            self.assertEqual(len(bundled_tuning), 2)
            self.assertEqual(len(bundled_valid), 1)

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
                "dataset_source": ["freshretail"] * periods,
                "series_id": ["store_1__sku_1"] * periods,
                "client_id": ["old_public_dataset_id"] * periods,
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
                "dt": pd.date_range("2024-01-01", periods=7, freq="D"),
                "dataset_source": ["freshretail"] * 6 + ["bakery"],
                "split_bucket": [
                    "train",
                    "train",
                    "train",
                    "val",
                    "val",
                    "test",
                    "test",
                ],
                "client_id": ["store_1__sku_1"] * 6 + ["bakery_store__sku_1"],
                "location_id": ["store_1"] * 6 + ["bakery_store"],
                "product_id": ["sku_1"] * 7,
                "target_demand_qty_d_plus_1": [
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                ],
                "rolling_mean_7": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                "lag_1": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                "current_day_demand_qty": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            }
        )

    def _gold_smoke_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=11, freq="D"),
                "dataset_source": ["freshretail_lt"] * 11,
                "split_bucket": [
                    "train",
                    "train",
                    "train",
                    "train",
                    "val",
                    "val",
                    "test",
                    "train",
                    "train",
                    "val",
                    "test",
                ],
                "client_id": [
                    "store_1__sku_1",
                    "store_1__sku_1",
                    "store_1__sku_1",
                    "store_1__sku_1",
                    "store_1__sku_1",
                    "store_1__sku_1",
                    "store_1__sku_1",
                    "store_2__sku_2",
                    "store_2__sku_2",
                    "store_2__sku_2",
                    "store_2__sku_2",
                ],
                "location_id": [
                    "store_1",
                    "store_1",
                    "store_1",
                    "store_1",
                    "store_1",
                    "store_1",
                    "store_1",
                    "store_2",
                    "store_2",
                    "store_2",
                    "store_2",
                ],
                "product_id": [
                    "sku_1",
                    "sku_1",
                    "sku_1",
                    "sku_1",
                    "sku_1",
                    "sku_1",
                    "sku_1",
                    "sku_2",
                    "sku_2",
                    "sku_2",
                    "sku_2",
                ],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0, 21.0, 22.0, 23.0],
                "rolling_mean_7": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 18.0, 19.0, 20.0, 21.0],
                "lag_1": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 17.0, 18.0, 19.0, 20.0],
                "current_day_demand_qty": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 19.0, 20.0, 21.0, 22.0],
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
        for key in (
            "train",
            "tuning",
            "optimisation_train",
            "optimisation_tuning",
            "valid",
            "optimisation_valid",
            "feature_manifest",
            "feature_roles",
            "split_manifest",
            "target_contract",
            "bundle_manifest",
            "optimisation_manifest",
        ):
            self.assertTrue(artifacts[key].exists())

    def _assert_feature_manifest(self, artifacts: dict[str, Path]) -> None:
        feature_manifest = json.loads(artifacts["feature_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(feature_manifest["feature_columns"], ["dataset_source", "rolling_mean_7"])
        self.assertEqual(feature_manifest["group_id_columns"], ["client_id"])
        self.assertIn("lag_1", feature_manifest["projection_columns"])
        self.assertIn("location_id", feature_manifest["projection_columns"])
        self.assertIn("product_id", feature_manifest["projection_columns"])
        self.assertEqual(feature_manifest["projection_dtypes"]["lag_1"], "float32")
        self.assertEqual(feature_manifest["projection_dtypes"]["rolling_mean_7"], "float32")
        self.assertEqual(feature_manifest["projection_dtypes"]["target_demand_qty_d_plus_1"], "float32")
        self.assertTrue(feature_manifest["feature_contract"]["rolling_mean_7"]["available_at_prediction"])
        self.assertNotIn("lag_1", feature_manifest["feature_columns"])
        self.assertNotIn("current_day_demand_qty", feature_manifest["feature_columns"])
        self.assertNotIn("series_id", feature_manifest["projection_columns"])
        self.assertNotIn("location_id", feature_manifest["feature_columns"])
        self.assertNotIn("product_id", feature_manifest["feature_columns"])

        feature_roles = json.loads(artifacts["feature_roles"].read_text(encoding="utf-8"))
        self.assertEqual(feature_roles["rolling_mean_7"]["role"], "time_varying_known_real")
        self.assertEqual(feature_roles["dataset_source"]["source_system"], "metadata")

    def _assert_bundled_train(self, artifacts: dict[str, Path]) -> None:
        bundled_train = pd.read_parquet(artifacts["train"])
        optimisation_train = pd.read_parquet(artifacts["optimisation_train"])
        self.assertIn("client_id", bundled_train.columns)
        self.assertIn("location_id", bundled_train.columns)
        self.assertIn("product_id", bundled_train.columns)
        self.assertNotIn("series_id", bundled_train.columns)
        self.assertEqual(str(bundled_train["rolling_mean_7"].dtype), "float32")
        self.assertEqual(str(bundled_train["target_demand_qty_d_plus_1"].dtype), "float32")
        self.assertIn("__tft_group_id", optimisation_train.columns)
        self.assertIn("__tft_time_idx", optimisation_train.columns)
        self.assertEqual(str(optimisation_train["__tft_time_idx"].dtype), "int32")

    def _assert_target_contract(self, artifacts: dict[str, Path]) -> None:
        target_contract = json.loads(artifacts["target_contract"].read_text(encoding="utf-8"))
        self.assertEqual(target_contract["learning_target_col"], "target_demand_qty_d_plus_1")
        self.assertEqual(target_contract["absolute_target_col"], "target_demand_qty_d_plus_1")

    def _assert_bundle_manifest(self, artifacts: dict[str, Path]) -> None:
        bundle_manifest = json.loads(artifacts["bundle_manifest"].read_text(encoding="utf-8"))
        optimisation_manifest = json.loads(artifacts["optimisation_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(bundle_manifest["bundle_version"], 2)
        self.assertEqual(bundle_manifest["train_rows"], 4)
        self.assertEqual(bundle_manifest["tuning_rows"], 2)
        self.assertEqual(bundle_manifest["valid_rows"], 2)
        self.assertEqual(bundle_manifest["feature_count"], 2)
        self.assertTrue(bundle_manifest["feature_roles_path"].endswith("feature_roles.json"))
        self.assertTrue(bundle_manifest["split_manifest_path"].endswith("split_manifest.json"))
        self.assertEqual(bundle_manifest["train_sha256"], self._sha256(artifacts["train"]))
        self.assertEqual(bundle_manifest["tuning_sha256"], self._sha256(artifacts["tuning"]))
        self.assertEqual(bundle_manifest["valid_sha256"], self._sha256(artifacts["valid"]))
        self.assertTrue(bundle_manifest["optimisation_train_path"].endswith("optimisation_train.parquet"))
        self.assertTrue(bundle_manifest["optimisation_manifest_path"].endswith("optimisation_manifest.json"))
        self.assertTrue(optimisation_manifest["precomputed_tft_support"])
        self.assertEqual(optimisation_manifest["support_columns"], ["__tft_group_id", "__tft_time_idx"])
        split_manifest = json.loads(artifacts["split_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(split_manifest["train"]["rows"], 4)
        self.assertEqual(split_manifest["tuning"]["rows"], 2)
        self.assertEqual(split_manifest["valid"]["rows"], 2)

    def _assert_gold_backed_bundle(self, artifacts: dict[str, Path]) -> None:
        bundle_manifest = json.loads(artifacts["bundle_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(bundle_manifest["train_rows"], 3)
        self.assertEqual(bundle_manifest["tuning_rows"], 2)
        self.assertEqual(bundle_manifest["valid_rows"], 2)
        self.assertIn("_cache/train.parquet", bundle_manifest["train_input_path"])
        self.assertIn("_cache/val.parquet", bundle_manifest["tuning_input_path"])
        self.assertIn("_cache/test.parquet", bundle_manifest["valid_input_path"])
        bundled_train = pd.read_parquet(artifacts["train"])
        bundled_tuning = pd.read_parquet(artifacts["tuning"])
        bundled_valid = pd.read_parquet(artifacts["valid"])
        optimisation_train = pd.read_parquet(artifacts["optimisation_train"])
        self.assertEqual(set(bundled_train["dataset_source"]), {"freshretail"})
        self.assertEqual(set(bundled_tuning["dataset_source"]), {"freshretail"})
        self.assertIn("bakery", set(bundled_valid["dataset_source"]))
        self.assertNotIn("split_bucket", bundled_train.columns)
        self.assertIn("lag_1", bundled_train.columns)
        self.assertIn("rolling_mean_7", bundled_train.columns)
        self.assertIn("__tft_group_id", optimisation_train.columns)
        self.assertIn("__tft_time_idx", optimisation_train.columns)

    def _sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
