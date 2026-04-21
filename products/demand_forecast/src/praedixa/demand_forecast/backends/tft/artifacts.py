from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedTFTModel:
    """Bundle runtime minimal pour prediction et serialisation du backend TFT."""

    model: Any
    dataset_parameters: dict[str, Any]
    history_frame: pd.DataFrame
    group_col: str
    time_idx_col: str
    target_col: str
    prediction_row_id_col: str
    batch_size: int
    quantiles: list[float]
    best_iteration: int
    model_hyperparameters: dict[str, Any]
    feature_scalers: dict[str, StandardScaler]
    target_scaler: StandardScaler | None
    runtime_profile: str
    system_info: dict[str, Any]
    runtime_metrics: dict[str, Any]
    git_sha: str | None
    bundle_manifest: dict[str, Any] | None
    data_hashes: dict[str, str]
    normalization_strategy: dict[str, Any]
    interpretability_payload: dict[str, Any] | None
    artifact_bundle_version: int
