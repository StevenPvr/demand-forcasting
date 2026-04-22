from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.artifact_assembly import (
    build_fitted_tft_model,
    extract_interpretability_payload,
)
from praedixa.demand_forecast.backends.tft.artifacts import FittedTFTModel
from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.backends.tft.dataloader_profile import (
    build_tft_dataloader_kwargs,
)
from praedixa.demand_forecast.backends.tft.frame_utils import build_combined_frame
from praedixa.demand_forecast.backends.tft.frame_utils import (
    attach_group_and_time_columns,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    lazy_import_tft_dependencies,
    resolve_model_params,
    suppress_tft_runtime_noise,
)
from praedixa.demand_forecast.backends.tft.training_dataset import (
    TrainingDatasetArtifacts,
    build_training_dataset_artifacts,
)
from praedixa.demand_forecast.backends.tft.training_runtime import (
    fit_trainer_model,
    seed_tft_runtime,
)


def _build_model(
    imports: dict[str, Any],
    training_dataset: Any,
    resolved_params: dict[str, object],
) -> Any:
    quantile_loss = imports["QuantileLoss"](quantiles=resolved_params["quantiles"])
    logging_metrics = imports["ModuleList"]([imports["WAPEMetric"](name="wape")])
    return imports["TemporalFusionTransformer"].from_dataset(
        training_dataset,
        learning_rate=float(cast(Any, resolved_params["learning_rate"])),
        hidden_size=int(cast(Any, resolved_params["hidden_size"])),
        attention_head_size=int(cast(Any, resolved_params["attention_head_size"])),
        dropout=float(cast(Any, resolved_params["dropout"])),
        hidden_continuous_size=int(cast(Any, resolved_params["hidden_continuous_size"])),
        lstm_layers=int(cast(Any, resolved_params["lstm_layers"])),
        loss=quantile_loss,
        output_size=len(cast(list[float], resolved_params["quantiles"])),
        log_interval=-1,
        reduce_on_plateau_patience=int(cast(Any, resolved_params["patience"])),
        optimizer="adam",
        weight_decay=float(cast(Any, resolved_params["weight_decay"])),
        logging_metrics=logging_metrics,
    )


def _prepared_training_frame(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_cols: list[str],
    train_weights: np.ndarray | None,
    valid_weights: np.ndarray | None,
) -> pd.DataFrame:
    combined_frame = build_combined_frame(
        train_frame,
        valid_frame,
        train_weights=train_weights,
        valid_weights=valid_weights,
    )
    return attach_group_and_time_columns(combined_frame, feature_cols)


def _training_loaders(
    *,
    training_dataset: Any,
    validation_dataset: Any,
    batch_size: int,
    resolved_params: dict[str, object],
) -> tuple[Any, Any]:
    dataloader_kwargs = build_tft_dataloader_kwargs(
        num_workers=int(cast(Any, resolved_params["num_workers"])),
        pin_memory=bool(cast(Any, resolved_params["pin_memory"])),
        persistent_workers=bool(cast(Any, resolved_params["persistent_workers"])),
        prefetch_factor=cast(int | None, resolved_params.get("prefetch_factor")),
    )
    train_loader = training_dataset.to_dataloader(train=True, batch_size=batch_size, **dataloader_kwargs)
    valid_loader = None if validation_dataset is None else validation_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        **dataloader_kwargs,
    )
    return train_loader, valid_loader


def _resolve_fit_request(
    *,
    train_frame: pd.DataFrame | None,
    feature_cols: list[str] | None,
    model_params: dict[str, object] | None,
    default_params: dict[str, object] | None,
    default_max_iter: int,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, object]]:
    if train_frame is None or feature_cols is None:
        raise ValueError("fit_tft_model requires a train_frame and feature_cols.")
    raise_if_tft_backend_required("tft_model_utils.fit_tft_model")
    return (
        train_frame,
        feature_cols,
        lazy_import_tft_dependencies(),
        resolve_model_params(
            model_params,
            default_params=default_params,
            default_max_iter=default_max_iter,
        ),
    )


def _resolved_dataset_artifacts(
    *,
    imports: dict[str, Any],
    dataset_artifacts: TrainingDatasetArtifacts | None,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
    train_weights: np.ndarray | None,
    valid_weights: np.ndarray | None,
) -> TrainingDatasetArtifacts:
    if dataset_artifacts is not None:
        return dataset_artifacts
    prepared_frame = _prepared_training_frame(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
        train_weights=train_weights,
        valid_weights=valid_weights,
    )
    return build_training_dataset_artifacts(
        imports,
        prepared_frame=prepared_frame,
        feature_cols=feature_cols,
        target_col=target_col,
        resolved_params=resolved_params,
    )


def _fit_resolved_model(
    *,
    imports: dict[str, Any],
    dataset_artifacts: TrainingDatasetArtifacts,
    resolved_params: dict[str, object],
) -> tuple[Any, Any, int, dict[str, float | int | str | None]]:
    train_loader, valid_loader = _training_loaders(
        training_dataset=dataset_artifacts.training_dataset,
        validation_dataset=dataset_artifacts.validation_dataset,
        batch_size=int(cast(Any, resolved_params["batch_size"])),
        resolved_params=resolved_params,
    )
    model = _build_model(imports, dataset_artifacts.training_dataset, resolved_params)
    fitted_model, best_iteration, runtime_metrics = fit_trainer_model(
        imports,
        model=model,
        resolved_params=resolved_params,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    return fitted_model, valid_loader, best_iteration, runtime_metrics


def fit_tft_model(
    train_frame: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    *,
    target_col: str = "target",
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
    default_max_iter: int = 30,
    valid_frame: pd.DataFrame | None = None,
    train_weights: np.ndarray | None = None,
    valid_weights: np.ndarray | None = None,
    dataset_artifacts: TrainingDatasetArtifacts | None = None,
) -> FittedTFTModel:
    train_frame, feature_cols, imports, resolved_params = _resolve_fit_request(
        train_frame=train_frame,
        feature_cols=feature_cols,
        model_params=model_params,
        default_params=default_params,
        default_max_iter=default_max_iter,
    )
    with suppress_tft_runtime_noise():
        seed_tft_runtime(imports, resolved_params)
        resolved_dataset_artifacts = _resolved_dataset_artifacts(
            imports=imports,
            dataset_artifacts=dataset_artifacts,
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=feature_cols,
            target_col=target_col,
            resolved_params=resolved_params,
            train_weights=train_weights,
            valid_weights=valid_weights,
        )
        model, valid_loader, best_iteration, runtime_metrics = _fit_resolved_model(
            imports=imports,
            dataset_artifacts=resolved_dataset_artifacts,
            resolved_params=resolved_params,
        )
        interpretability_payload = extract_interpretability_payload(
            model=model,
            validation_loader=valid_loader,
            resolved_params=resolved_params,
            layout=resolved_dataset_artifacts.layout,
            target_col=target_col,
        )
    return build_fitted_tft_model(
        model=model,
        training_dataset=resolved_dataset_artifacts.training_dataset,
        training_slice=resolved_dataset_artifacts.training_slice,
        target_col=target_col,
        resolved_params=resolved_params,
        best_iteration=best_iteration,
        feature_scalers=resolved_dataset_artifacts.feature_scalers,
        target_scaler=None,
        normalization_strategy=resolved_dataset_artifacts.normalization_strategy,
        runtime_metrics=runtime_metrics,
        interpretability_payload=interpretability_payload,
    )
