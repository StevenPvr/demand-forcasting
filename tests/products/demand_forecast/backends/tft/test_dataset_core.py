from __future__ import annotations

from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.dataset_core import (  # noqa: E402
    build_training_dataset_core,
    build_validation_dataset_core,
    clone_training_dataset_from_core,
    clone_validation_dataset_from_core,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import TFTLayout  # noqa: E402
from praedixa.demand_forecast.backends.tft.frame_utils import (  # noqa: E402
    PREDICTION_ROW_ID_COL,
    attach_group_and_time_columns,
    build_combined_frame,
    resolve_layout,
)
from praedixa.demand_forecast.backends.tft.model_common import (  # noqa: E402
    lazy_import_tft_dependencies,
)
from praedixa.demand_forecast.backends.tft.training_dataset import (  # noqa: E402
    build_categorical_encoders,
    fit_real_feature_scalers,
)


def _take_frame_rows(
    frame: pd.DataFrame,
    row_selector: slice,
) -> pd.DataFrame:
    selected = frame.iloc[row_selector]
    return selected.copy()


def _synthetic_train_valid_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    for group in ["a", "b"]:
        rows: list[dict[str, object]] = []
        for offset, dt in enumerate(pd.date_range("2024-01-01", periods=140, freq="D")):
            rows.append(
                {
                    "series_id": group,
                    "location_id": f"loc_{group}",
                    "product_id": f"prod_{group}",
                    "dataset_source": "freshretail_lt",
                    "dt": dt,
                    "rolling_mean_7": float(offset),
                    "target_demand_qty_d_plus_1": float(offset + 1),
                }
            )
        group_frame = pd.DataFrame(rows)
        train_parts.append(_take_frame_rows(group_frame, slice(None, 110)))
        valid_parts.append(_take_frame_rows(group_frame, slice(110, None)))
    train_frame = pd.concat(train_parts, ignore_index=True)
    valid_frame = pd.concat(valid_parts, ignore_index=True)
    feature_cols = ["dataset_source", "location_id", "product_id", "rolling_mean_7"]
    return train_frame, valid_frame, feature_cols


def _prepared_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str], TFTLayout, dict[str, StandardScaler]]:
    train_frame, valid_frame, feature_cols = _synthetic_train_valid_frames()
    prepared = attach_group_and_time_columns(
        build_combined_frame(
            train_frame,
            valid_frame,
            train_weights=None,
            valid_weights=None,
        ),
        feature_cols,
    )
    train_prepared = prepared.loc[prepared["__tft_split"] == "train"].copy()
    layout = resolve_layout(train_prepared, feature_cols)
    feature_scalers = fit_real_feature_scalers(train_prepared, layout=layout)
    return train_prepared, prepared, feature_cols, layout, feature_scalers


def _validation_frame(
    prepared: pd.DataFrame,
    *,
    encoder_length: int,
) -> pd.DataFrame:
    grouped_frames: list[pd.DataFrame] = []
    next_row_id = 0
    for _, group_frame in prepared.groupby("__tft_group_id", sort=False):
        history_tail = group_frame.loc[group_frame["__tft_split"] == "train"].tail(encoder_length).copy()
        valid_rows = group_frame.loc[group_frame["__tft_split"] == "valid"].copy()
        valid_rows[PREDICTION_ROW_ID_COL] = np.arange(
            next_row_id,
            next_row_id + len(valid_rows),
            dtype=np.int32,
        )
        next_row_id += len(valid_rows)
        grouped_frames.append(pd.concat([history_tail, valid_rows], ignore_index=True))
    return pd.concat(grouped_frames, ignore_index=True)


class DatasetCoreTests(unittest.TestCase):
    def test_training_dataset_core_is_picklable(self) -> None:
        train_prepared, _, _, layout, feature_scalers = _prepared_frames()
        imports = lazy_import_tft_dependencies()
        categorical_encoders = build_categorical_encoders(imports, layout=layout)

        training_core = build_training_dataset_core(
            imports,
            training_frame=train_prepared,
            target_col="target_demand_qty_d_plus_1",
            max_encoder_length=28,
            weight_col=None,
            layout=layout,
            categorical_encoders=categorical_encoders,
            real_feature_scalers=feature_scalers,
        )

        serialized = pickle.dumps(training_core.dataset)
        self.assertGreater(len(serialized), 0)

    def test_clone_training_dataset_matches_direct_build(self) -> None:
        train_prepared, _, _, layout, feature_scalers = _prepared_frames()
        imports = lazy_import_tft_dependencies()
        categorical_encoders = build_categorical_encoders(imports, layout=layout)

        direct_core = build_training_dataset_core(
            imports,
            training_frame=train_prepared,
            target_col="target_demand_qty_d_plus_1",
            max_encoder_length=14,
            weight_col=None,
            layout=layout,
            categorical_encoders=categorical_encoders,
            real_feature_scalers=feature_scalers,
        )
        master_core = build_training_dataset_core(
            imports,
            training_frame=train_prepared,
            target_col="target_demand_qty_d_plus_1",
            max_encoder_length=56,
            weight_col=None,
            layout=layout,
            categorical_encoders=categorical_encoders,
            real_feature_scalers=feature_scalers,
        )
        cloned_dataset = clone_training_dataset_from_core(
            core=master_core,
            max_encoder_length=14,
        )

        self.assertTrue(
            direct_core.dataset.index[["index_start", "index_end", "sequence_length"]]
            .reset_index(drop=True)
            .equals(
                cloned_dataset.index[["index_start", "index_end", "sequence_length"]].reset_index(drop=True)
            )
        )

    def test_clone_validation_dataset_matches_direct_build(self) -> None:
        train_prepared, prepared, _, layout, feature_scalers = _prepared_frames()
        imports = lazy_import_tft_dependencies()
        categorical_encoders = build_categorical_encoders(imports, layout=layout)

        training_core_14 = build_training_dataset_core(
            imports,
            training_frame=train_prepared,
            target_col="target_demand_qty_d_plus_1",
            max_encoder_length=14,
            weight_col=None,
            layout=layout,
            categorical_encoders=categorical_encoders,
            real_feature_scalers=feature_scalers,
        )
        training_core_56 = build_training_dataset_core(
            imports,
            training_frame=train_prepared,
            target_col="target_demand_qty_d_plus_1",
            max_encoder_length=56,
            weight_col=None,
            layout=layout,
            categorical_encoders=categorical_encoders,
            real_feature_scalers=feature_scalers,
        )
        direct_validation_core = build_validation_dataset_core(
            training_core=training_core_14,
            validation_frame=_validation_frame(prepared, encoder_length=14),
            min_prediction_idx=14,
        )
        master_validation_core = build_validation_dataset_core(
            training_core=training_core_56,
            validation_frame=_validation_frame(prepared, encoder_length=56),
            min_prediction_idx=56,
        )
        cloned_validation = clone_validation_dataset_from_core(
            core=master_validation_core,
            max_encoder_length=14,
            kept_prediction_row_ids=set(range(60)),
            valid_weight_by_row_id={row_id: 1.0 for row_id in range(60)},
        )

        self.assertTrue(
            direct_validation_core.dataset.index[["index_start", "index_end", "sequence_length"]]
            .reset_index(drop=True)
            .equals(
                cloned_validation.index[["index_start", "index_end", "sequence_length"]].reset_index(drop=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
