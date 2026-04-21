from __future__ import annotations

import resource
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from praedixa.demand_forecast.backends.tft.artifacts import FittedTFTModel
from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.backends.tft.dataloader_profile import build_tft_dataloader_kwargs
from praedixa.demand_forecast.backends.tft.feature_mapping import TFTLayout
from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    PREDICTION_ROW_ID_COL,
    SPLIT_COL,
    TIME_IDX_COL,
    WEIGHT_COL,
    attach_group_and_time_columns,
    build_combined_frame,
    resolve_layout,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    DEFAULT_RUNTIME_PROFILE_NAME,
    lazy_import_tft_dependencies,
    resolve_model_params,
    suppress_tft_runtime_noise,
)
from praedixa.demand_forecast.backends.tft.system_info import collect_tft_system_info, resolve_git_sha


def _build_timeseries_dataset(
    imports: dict[str, Any],
    frame: pd.DataFrame,
    *,
    target_col: str,
    max_encoder_length: int,
    weight_col: str | None,
    layout: TFTLayout,
    categorical_encoders: dict[str, Any] | None,
    real_feature_scalers: dict[str, StandardScaler] | None,
) -> Any:
    return imports["TimeSeriesDataSet"](
        frame,
        time_idx=TIME_IDX_COL,
        target=target_col,
        group_ids=[GROUP_COL],
        weight=weight_col,
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=layout["static_categoricals"],
        static_reals=layout["static_reals"],
        time_varying_known_categoricals=layout["time_varying_known_categoricals"],
        time_varying_known_reals=layout["time_varying_known_reals"],
        time_varying_unknown_categoricals=layout["time_varying_unknown_categoricals"],
        time_varying_unknown_reals=[*layout["time_varying_unknown_reals"], target_col],
        allow_missing_timesteps=False,
        target_normalizer=None,
        categorical_encoders=categorical_encoders,
        scalers=real_feature_scalers,
    )


def _build_validation_dataset(
    imports: dict[str, Any],
    training_dataset: Any,
    prepared_frame: pd.DataFrame,
) -> Any:
    valid_frame = prepared_frame.loc[prepared_frame[SPLIT_COL] == "valid"].copy()
    if valid_frame.empty:
        return None
    return imports["TimeSeriesDataSet"].from_dataset(
        training_dataset,
        prepared_frame,
        min_prediction_idx=int(valid_frame[TIME_IDX_COL].min()),
        stop_randomization=True,
    )


def _build_model(
    imports: dict[str, Any],
    training_dataset: Any,
    resolved_params: dict[str, object],
) -> Any:
    quantile_loss = imports["QuantileLoss"](quantiles=resolved_params["quantiles"])
    return imports["TemporalFusionTransformer"].from_dataset(
        training_dataset,
        learning_rate=float(cast(Any, resolved_params["learning_rate"])),
        hidden_size=int(cast(Any, resolved_params["hidden_size"])),
        attention_head_size=int(cast(Any, resolved_params["attention_head_size"])),
        dropout=float(cast(Any, resolved_params["dropout"])),
        hidden_continuous_size=int(cast(Any, resolved_params["hidden_continuous_size"])),
        loss=quantile_loss,
        output_size=len(cast(list[float], resolved_params["quantiles"])),
        log_interval=-1,
        reduce_on_plateau_patience=int(cast(Any, resolved_params["patience"])),
        optimizer="adam",
    )


def _build_trainer(
    imports: dict[str, Any],
    resolved_params: dict[str, object],
    *,
    checkpoint_dir: str | None,
    has_validation: bool,
) -> tuple[Any, Any | None]:
    callbacks: list[Any] = []
    checkpoint_callback = None
    if has_validation and checkpoint_dir is not None:
        callbacks.append(
            imports["EarlyStopping"](
                monitor="val_loss",
                mode="min",
                patience=int(cast(Any, resolved_params["patience"])),
                strict=True,
                check_finite=True,
            )
        )
        checkpoint_callback = imports["ModelCheckpoint"](
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        callbacks.append(checkpoint_callback)
    trainer = imports["Trainer"](
        accelerator=str(cast(Any, resolved_params["accelerator"])),
        devices=int(cast(Any, resolved_params["devices"])),
        precision=str(cast(Any, resolved_params["precision"])),
        max_epochs=int(cast(Any, resolved_params["max_epochs"])),
        gradient_clip_val=float(cast(Any, resolved_params["gradient_clip_val"])),
        deterministic=True,
        benchmark=False,
        enable_checkpointing=bool(checkpoint_callback),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    return trainer, checkpoint_callback


def _seed_tft_runtime(imports: dict[str, Any], resolved_params: dict[str, object]) -> None:
    random_state = int(cast(Any, resolved_params.get("random_state", 7)))
    imports["seed_everything"](random_state, workers=True)
    imports["torch"].use_deterministic_algorithms(True)
    set_matmul_precision = getattr(imports["torch"], "set_float32_matmul_precision", None)
    if callable(set_matmul_precision):
        set_matmul_precision(str(cast(Any, resolved_params.get("matmul_precision", "highest"))))


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


def _fit_target_scaler(training_slice: pd.DataFrame, *, target_col: str) -> StandardScaler:
    numeric_target = pd.to_numeric(training_slice[target_col], errors="coerce").dropna()
    if numeric_target.empty:
        raise ValueError(f"Unable to fit TFT target scaler because `{target_col}` has no usable training values.")
    scaler = StandardScaler()
    cast(Any, scaler).fit(numeric_target.to_frame(name=target_col).astype(float))
    return scaler


def _apply_target_scaler(
    frame: pd.DataFrame,
    *,
    target_col: str,
    target_scaler: StandardScaler,
) -> pd.DataFrame:
    scaled = frame.copy()
    scaled[target_col] = pd.to_numeric(scaled[target_col], errors="coerce").astype("float32")
    valid_mask = scaled[target_col].notna()
    if valid_mask.any():
        transformed = cast(Any, target_scaler).transform(scaled.loc[valid_mask, [target_col]].astype(float))
        scaled.loc[valid_mask, target_col] = np.asarray(transformed, dtype=np.float32).reshape(-1)
    return scaled


def _real_feature_scaler_columns(layout: TFTLayout) -> list[str]:
    scaler_columns: list[str] = []
    for column in [*layout["static_reals"], *layout["time_varying_known_reals"], *layout["time_varying_unknown_reals"]]:
        if column != TIME_IDX_COL and column not in scaler_columns:
            scaler_columns.append(column)
    return scaler_columns


def _categorical_encoder_columns(layout: TFTLayout) -> list[str]:
    encoder_columns: list[str] = [GROUP_COL]
    for column in [*layout["static_categoricals"], *layout["time_varying_known_categoricals"], *layout["time_varying_unknown_categoricals"]]:
        if column not in encoder_columns:
            encoder_columns.append(column)
    return encoder_columns


def _build_categorical_encoders(
    imports: dict[str, Any],
    *,
    layout: TFTLayout,
) -> dict[str, Any]:
    return {
        column: imports["NaNLabelEncoder"](add_nan=True, warn=False)
        for column in _categorical_encoder_columns(layout)
    }


def _fit_real_feature_scalers(
    training_slice: pd.DataFrame,
    *,
    layout: TFTLayout,
) -> dict[str, StandardScaler]:
    scalers: dict[str, StandardScaler] = {}
    for column in _real_feature_scaler_columns(layout):
        numeric_values = pd.to_numeric(training_slice[column], errors="coerce").dropna()
        if numeric_values.empty:
            continue
        scaler = StandardScaler()
        cast(Any, scaler).fit(numeric_values.to_frame(name=column).astype(float))
        scalers[column] = scaler
    return scalers


def _training_dataset_components(
    imports: dict[str, Any],
    *,
    prepared_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
) -> tuple[pd.DataFrame, Any, Any, StandardScaler, dict[str, StandardScaler]]:
    layout = resolve_layout(prepared_frame, feature_cols)
    unscaled_training_slice = prepared_frame.loc[prepared_frame[SPLIT_COL] == "train"].copy()
    target_scaler = _fit_target_scaler(unscaled_training_slice, target_col=target_col)
    scaled_frame = _apply_target_scaler(prepared_frame, target_col=target_col, target_scaler=target_scaler)
    training_slice = scaled_frame.loc[scaled_frame[SPLIT_COL] == "train"].copy()
    categorical_encoders = _build_categorical_encoders(imports, layout=layout)
    feature_scalers = _fit_real_feature_scalers(training_slice, layout=layout)
    training_dataset = _build_timeseries_dataset(
        imports,
        training_slice,
        target_col=target_col,
        max_encoder_length=int(cast(Any, resolved_params["max_encoder_length"])),
        weight_col=WEIGHT_COL if WEIGHT_COL in training_slice.columns else None,
        layout=layout,
        categorical_encoders=categorical_encoders,
        real_feature_scalers=feature_scalers or None,
    )
    validation_dataset = _build_validation_dataset(imports, training_dataset, scaled_frame)
    return training_slice, training_dataset, validation_dataset, target_scaler, feature_scalers


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
    )
    train_loader = training_dataset.to_dataloader(train=True, batch_size=batch_size, **dataloader_kwargs)
    valid_loader = None if validation_dataset is None else validation_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        **dataloader_kwargs,
    )
    return train_loader, valid_loader


def _fit_trainer_model(
    imports: dict[str, Any],
    *,
    model: Any,
    resolved_params: dict[str, object],
    train_loader: Any,
    valid_loader: Any,
) -> tuple[Any, int, dict[str, float | int]]:
    torch = imports["torch"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fit_start = time.perf_counter()
    with TemporaryDirectory(prefix="tft-backend-") as checkpoint_dir:
        trainer, checkpoint_callback = _build_trainer(
            imports,
            resolved_params,
            checkpoint_dir=checkpoint_dir,
            has_validation=valid_loader is not None,
        )
        trainer.fit(model, train_loader, valid_loader)
        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            model = imports["TemporalFusionTransformer"].load_from_checkpoint(checkpoint_callback.best_model_path)
        best_iteration = max(0, int(getattr(trainer, "current_epoch", 1)) - 1)
    fit_duration_seconds = max(0.0, time.perf_counter() - fit_start)
    epochs_completed = max(1, best_iteration + 1)
    rows_seen = len(train_loader.dataset) * epochs_completed
    peak_ram_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_ram_mb = peak_ram_raw / (1024.0 * 1024.0) if sys.platform == "darwin" else peak_ram_raw / 1024.0
    peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return model, best_iteration, {
        "fit_duration_seconds": fit_duration_seconds,
        "epochs_completed": epochs_completed,
        "epoch_duration_seconds": fit_duration_seconds / float(epochs_completed),
        "rows_per_second": float(rows_seen) / fit_duration_seconds if fit_duration_seconds > 0 else 0.0,
        "loader_worker_count": int(cast(Any, resolved_params["num_workers"])),
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_bytes": peak_vram_bytes,
    }


def _history_tail_frame(frame: pd.DataFrame, *, max_encoder_length: int) -> pd.DataFrame:
    return frame.groupby(GROUP_COL, sort=False).tail(max_encoder_length).copy().reset_index(drop=True)


def _build_fitted_tft_model(
    *,
    model: Any,
    training_dataset: Any,
    training_slice: pd.DataFrame,
    target_col: str,
    resolved_params: dict[str, object],
    best_iteration: int,
    feature_scalers: dict[str, StandardScaler],
    target_scaler: StandardScaler,
    runtime_metrics: dict[str, float | int],
) -> FittedTFTModel:
    runtime_profile = str(cast(Any, resolved_params.get("runtime_profile", DEFAULT_RUNTIME_PROFILE_NAME)))
    return FittedTFTModel(
        model=model,
        dataset_parameters=cast(dict[str, Any], training_dataset.get_parameters()),
        history_frame=_history_tail_frame(
            training_slice,
            max_encoder_length=int(cast(Any, resolved_params["max_encoder_length"])),
        ),
        group_col=GROUP_COL,
        time_idx_col=TIME_IDX_COL,
        target_col=target_col,
        prediction_row_id_col=PREDICTION_ROW_ID_COL,
        batch_size=int(cast(Any, resolved_params["batch_size"])),
        quantiles=list(cast(Any, resolved_params["quantiles"])),
        best_iteration=best_iteration,
        model_hyperparameters=resolved_params,
        feature_scalers=feature_scalers,
        target_scaler=target_scaler,
        runtime_profile=runtime_profile,
        system_info=collect_tft_system_info(runtime_profile=runtime_profile),
        runtime_metrics=runtime_metrics,
        git_sha=resolve_git_sha(),
        bundle_manifest=None,
        data_hashes={},
    )


def _fit_prepared_tft_model(
    *,
    imports: dict[str, Any],
    prepared_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
) -> tuple[pd.DataFrame, Any, Any, StandardScaler, dict[str, StandardScaler], int, dict[str, float | int]]:
    training_slice, training_dataset, validation_dataset, target_scaler, feature_scalers = (
        _training_dataset_components(
            imports,
            prepared_frame=prepared_frame,
            feature_cols=feature_cols,
            target_col=target_col,
            resolved_params=resolved_params,
        )
    )
    train_loader, valid_loader = _training_loaders(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        batch_size=int(cast(Any, resolved_params["batch_size"])),
        resolved_params=resolved_params,
    )
    model = _build_model(imports, training_dataset, resolved_params)
    model, best_iteration, runtime_metrics = _fit_trainer_model(
        imports,
        model=model,
        resolved_params=resolved_params,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    return (
        training_slice,
        training_dataset,
        model,
        target_scaler,
        feature_scalers,
        best_iteration,
        runtime_metrics,
    )


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
) -> FittedTFTModel:
    train_frame, feature_cols, imports, resolved_params = _resolve_fit_request(
        train_frame=train_frame,
        feature_cols=feature_cols,
        model_params=model_params,
        default_params=default_params,
        default_max_iter=default_max_iter,
    )
    with suppress_tft_runtime_noise():
        _seed_tft_runtime(imports, resolved_params)
        prepared_frame = _prepared_training_frame(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=feature_cols,
            train_weights=train_weights,
            valid_weights=valid_weights,
        )
        training_slice, training_dataset, model, target_scaler, feature_scalers, best_iteration, runtime_metrics = _fit_prepared_tft_model(
            imports=imports,
            prepared_frame=prepared_frame,
            feature_cols=feature_cols,
            target_col=target_col,
            resolved_params=resolved_params,
        )
    return _build_fitted_tft_model(
        model=model,
        training_dataset=training_dataset,
        training_slice=training_slice,
        target_col=target_col,
        resolved_params=resolved_params,
        best_iteration=best_iteration,
        feature_scalers=feature_scalers,
        target_scaler=target_scaler,
        runtime_metrics=runtime_metrics,
    )
