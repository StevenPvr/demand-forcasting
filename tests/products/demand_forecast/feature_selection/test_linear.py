from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping, cast
import unittest

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

from praedixa.demand_forecast.backends.tft.feature_contract import (  # noqa: E402
    build_feature_contract,
)
from praedixa.demand_forecast.feature_selection.linear import (  # noqa: E402
    run_feature_selection,
)
from praedixa.demand_forecast.feature_selection.redundancy import (  # noqa: E402
    drop_linear_correlated_candidate_features,
    drop_nonlinear_correlated_candidate_features,
)


FEATURE_COLUMNS: list[str] = [
    "dataset_source_static_cat",
    "product_family_static_cat",
    "lag_2_unknown_real",
    "rolling_mean_7_known_real",
    "current_day_demand_qty_known_real",
    "history_available_days_known_real",
]
FEATURE_ROLES: dict[str, str] = {
    "dataset_source_static_cat": "static_categorical",
    "product_family_static_cat": "static_categorical",
    "lag_2_unknown_real": "time_varying_unknown_real",
    "rolling_mean_7_known_real": "time_varying_known_real",
    "current_day_demand_qty_known_real": "time_varying_known_real",
    "history_available_days_known_real": "time_varying_known_real",
}
TARGET_COL = "target_demand_qty_d_plus_1"


class LinearFeatureSelectionTests(unittest.TestCase):
    def test_run_feature_selection_writes_filtered_bundle_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_dir = root / "bundle"
            output_dir = root / "selected_bundle"
            bundle_dir.mkdir()
            self._write_bundle(bundle_dir)

            result = run_feature_selection(
                bundle_dir=bundle_dir,
                output_dir=output_dir,
                n_trials=2,
                n_folds=2,
                n_fold_workers=2,
            )

            self.assertEqual(result.train_rows, 8)
            feature_manifest = self._read_json(output_dir / "feature_manifest.json")
            selected_features = cast(list[str], feature_manifest["feature_columns"])
            feature_roles = cast(dict[str, object], feature_manifest["feature_roles"])
            feature_contract = cast(
                dict[str, object],
                feature_manifest["feature_contract"],
            )
            self.assertIn("dataset_source_static_cat", selected_features)
            self.assertIn("product_family_static_cat", selected_features)
            self.assertNotIn("history_available_days_known_real", selected_features)
            self.assertEqual(feature_manifest["feature_count"], len(selected_features))
            self.assertEqual(
                set(feature_roles),
                set(selected_features),
            )
            self.assertEqual(
                set(feature_contract),
                set(selected_features),
            )

            selected_train = pd.read_parquet(output_dir / "train.parquet")
            self.assertIn(TARGET_COL, selected_train.columns)
            self.assertIn("dataset_source_static_cat", selected_train.columns)
            self.assertIn("product_family_static_cat", selected_train.columns)
            self.assertNotIn(
                "history_available_days_known_real", selected_train.columns
            )

            optimisation_manifest = self._read_json(
                output_dir / "optimisation_manifest.json"
            )
            projection_columns = cast(
                list[str],
                optimisation_manifest["projection_columns"],
            )
            self.assertTrue(optimisation_manifest["precomputed_tft_support"])
            self.assertIn("__tft_time_idx", projection_columns)
            self.assertNotIn(
                "history_available_days_known_real",
                projection_columns,
            )
            self.assertIn(
                "dataset_source_static_cat",
                projection_columns,
            )

            bundle_manifest = self._read_json(output_dir / "bundle_manifest.json")
            self.assertEqual(bundle_manifest["feature_count"], len(selected_features))
            self.assertTrue(bundle_manifest["train_sha256"])
            self.assertTrue((output_dir / "feature_selection_summary.json").exists())
            self.assertTrue((output_dir / "feature_selection_importances.csv").exists())
            summary = self._read_json(output_dir / "feature_selection_summary.json")
            self.assertEqual(summary["selector_model_family"], "ElasticNet")
            self.assertEqual(summary["selection_data_scope"], "train_split_only")
            self.assertIsNone(summary["selector_tuning_mae"])
            self.assertEqual(summary["max_selected_model_features"], 50)
            self.assertEqual(summary["n_fold_workers"], 2)
            self.assertEqual(summary["categorical_candidate_features"], [])
            self.assertLessEqual(
                len(cast(list[str], summary["selected_numeric_feature_columns"])),
                50,
            )
            self.assertIn(
                "dataset_source_static_cat",
                cast(list[str], summary["passthrough_feature_columns"]),
            )
            self.assertIn("dropped_linear_correlated_features", summary)
            self.assertIn("dropped_nonlinear_correlated_features", summary)

    def test_redundancy_filters_run_linear_then_nonlinear(self) -> None:
        x_values = list(range(-10, 11))
        linear_frame = pd.DataFrame(
            {
                "linear_a_known_real": [float(value) for value in x_values],
                "linear_b_known_real": [float(value * 2) for value in x_values],
                TARGET_COL: [float(value) for value in x_values],
            }
        )
        nonlinear_frame = pd.DataFrame(
            {
                "curve_a_known_real": [float(value * value) for value in x_values],
                "curve_b_known_real": [float(value) for value in x_values],
                TARGET_COL: [float(value * value) for value in x_values],
            }
        )

        with self.assertLogs(
            "praedixa.demand_forecast.feature_selection.redundancy",
            level="INFO",
        ) as captured_logs:
            linear_kept, linear_dropped = drop_linear_correlated_candidate_features(
                linear_frame,
                ["linear_a_known_real", "linear_b_known_real"],
                TARGET_COL,
                0.95,
            )
        nonlinear_kept, nonlinear_dropped = (
            drop_nonlinear_correlated_candidate_features(
                nonlinear_frame,
                ["curve_a_known_real", "curve_b_known_real"],
                TARGET_COL,
                0.95,
            )
        )

        self.assertEqual(linear_kept, ["linear_a_known_real"])
        self.assertIn("linear_b_known_real", linear_dropped)
        self.assertIn("workers=1", "\n".join(captured_logs.output))
        self.assertIn("linear correlation progress", "\n".join(captured_logs.output))
        self.assertIn("curve_b_known_real", nonlinear_dropped)
        self.assertIn("curve_a_known_real", nonlinear_kept)

    def _write_bundle(self, bundle_dir: Path) -> None:
        train = self._frame("2024-01-01", range(8))
        tuning = self._frame("2024-01-09", range(8, 12))
        valid = self._frame("2024-01-13", range(12, 14))
        for name, frame in (
            ("train.parquet", train),
            ("tuning.parquet", tuning),
            ("valid.parquet", valid),
            ("optimisation_train.parquet", self._with_tft_support(train)),
            ("optimisation_tuning.parquet", self._with_tft_support(tuning)),
            ("optimisation_valid.parquet", self._with_tft_support(valid)),
        ):
            frame.to_parquet(bundle_dir / name, index=False)
        feature_contract = build_feature_contract(FEATURE_COLUMNS)
        self._write_json(
            bundle_dir / "feature_manifest.json",
            {
                "feature_columns": FEATURE_COLUMNS,
                "feature_count": len(FEATURE_COLUMNS),
                "group_id_columns": ["client_id"],
                "projection_columns": ["client_id", TARGET_COL, *FEATURE_COLUMNS],
                "projection_dtypes": {
                    column: "object" if column.endswith("_cat") else "float32"
                    for column in FEATURE_COLUMNS
                },
                "feature_roles": FEATURE_ROLES,
                "feature_contract": feature_contract,
            },
        )
        self._write_json(bundle_dir / "feature_roles.json", feature_contract)
        self._write_json(
            bundle_dir / "target_contract.json",
            {
                "learning_target_col": TARGET_COL,
                "absolute_target_col": TARGET_COL,
            },
        )
        self._write_json(
            bundle_dir / "optimisation_manifest.json",
            {
                "bundle_version": 1,
                "projection_columns": [
                    "client_id",
                    TARGET_COL,
                    *FEATURE_COLUMNS,
                    "__tft_group_id",
                    "__tft_time_idx",
                ],
                "group_id_columns": ["client_id"],
                "support_columns": ["__tft_group_id", "__tft_time_idx"],
                "train_group_count": 1,
                "precomputed_tft_support": True,
            },
        )
        self._write_json(
            bundle_dir / "split_manifest.json",
            {"train": {"rows": 8}, "tuning": {"rows": 4}, "valid": {"rows": 2}},
        )
        self._write_json(bundle_dir / "training_exclusion_report.json", {})

    def _frame(self, start_date: str, offsets: range) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for offset in offsets:
            rolling_signal = float(10 + offset)
            current_signal = float((offset % 3) * 2)
            rows.append(
                {
                    "dt": pd.Timestamp(start_date) + pd.Timedelta(days=offset),
                    "client_id": "client_a",
                    "dataset_source_static_cat": "synthetic_foodservice",
                    "product_family_static_cat": (
                        "high_demand" if offset % 2 == 0 else "low_demand"
                    ),
                    "lag_2_unknown_real": float(5 + offset),
                    "rolling_mean_7_known_real": rolling_signal,
                    "current_day_demand_qty_known_real": current_signal,
                    "history_available_days_known_real": 30.0,
                    TARGET_COL: (100.0 if offset % 2 == 0 else 10.0)
                    + rolling_signal * 0.1
                    + current_signal,
                }
            )
        return pd.DataFrame(rows)

    def _with_tft_support(self, frame: pd.DataFrame) -> pd.DataFrame:
        projected = frame.copy()
        projected["__tft_group_id"] = "client_a"
        projected["__tft_time_idx"] = range(len(projected))
        return projected

    def _read_json(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
