from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterable, Iterator, cast
import warnings

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_mapping import (
    select_explicit_tft_feature_columns,
)
from praedixa.demand_forecast.backends.tft.runtime_profile import (
    DEFAULT_COMPILE_MODE,
    DEFAULT_DETERMINISM_MODE,
    DEFAULT_RUNTIME_PROFILE_NAME,
    resolve_runtime_profile,
)


DEFAULT_TFT_MODEL_PARAMS: dict[str, object] = {
    "runtime_profile": DEFAULT_RUNTIME_PROFILE_NAME,
    "batch_size": 64,
    "compile_mode": DEFAULT_COMPILE_MODE,
    "determinism_mode": DEFAULT_DETERMINISM_MODE,
    "dropout": 0.1,
    "gradient_clip_val": 0.1,
    "hidden_continuous_size": 8,
    "hidden_size": 16,
    "attention_head_size": 1,
    "learning_rate": 1e-2,
    "lstm_layers": 1,
    "max_encoder_length": 28,
    "max_epochs": 30,
    "patience": 3,
    "quantiles": [0.025, 0.1, 0.5, 0.9, 0.975],
    "tensorboard_logdir": None,
    "weight_decay": 1e-4,
}

_SUPPRESSED_TFT_WARNING_PATTERNS: tuple[str, ...] = (
    r"Attribute 'loss' is an instance of `nn\.Module` and is already saved during checkpointing\.",
    r"Attribute 'logging_metrics' is an instance of `nn\.Module` and is already saved during checkpointing\.",
    r"`isinstance\(treespec, LeafSpec\)` is deprecated, use `isinstance\(treespec, TreeSpec\) and treespec\.is_leaf\(\)` instead\.",
    r"The 'train_dataloader' does not have many workers which may be a bottleneck\.",
    r"The 'val_dataloader' does not have many workers which may be a bottleneck\.",
    r"The 'predict_dataloader' does not have many workers which may be a bottleneck\.",
    r"GPU available but not used\. You can set it by doing `Trainer\(accelerator='gpu'\)`\.",
)
_SUPPRESSED_TFT_LOGGERS: tuple[str, ...] = (
    "lightning.fabric.utilities.seed",
    "lightning.pytorch.utilities.rank_zero",
)


def lazy_import_tft_dependencies() -> dict[str, Any]:
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import (
        DeviceStatsMonitor,
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data.encoders import GroupNormalizer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss

    return {
        "CSVLogger": CSVLogger,
        "DeviceStatsMonitor": DeviceStatsMonitor,
        "torch": torch,
        "Trainer": Trainer,
        "seed_everything": seed_everything,
        "EarlyStopping": EarlyStopping,
        "GroupNormalizer": GroupNormalizer,
        "LearningRateMonitor": LearningRateMonitor,
        "ModelCheckpoint": ModelCheckpoint,
        "TensorBoardLogger": TensorBoardLogger,
        "TimeSeriesDataSet": TimeSeriesDataSet,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "NaNLabelEncoder": NaNLabelEncoder,
        "QuantileLoss": QuantileLoss,
    }


@contextmanager
def suppress_tft_runtime_noise() -> Iterator[None]:
    logger_states = [
        (logger_name, logging.getLogger(logger_name), logging.getLogger(logger_name).level)
        for logger_name in _SUPPRESSED_TFT_LOGGERS
    ]
    with warnings.catch_warnings():
        for pattern in _SUPPRESSED_TFT_WARNING_PATTERNS:
            warnings.filterwarnings("ignore", message=pattern)
        for _, logger, previous_level in logger_states:
            logger.setLevel(max(previous_level, logging.ERROR))
        try:
            yield
        finally:
            for _, logger, previous_level in logger_states:
                logger.setLevel(previous_level)


def select_tft_feature_columns(
    frame: pd.DataFrame,
    *,
    excluded_cols: Iterable[str],
) -> list[str]:
    return select_explicit_tft_feature_columns(frame, excluded_cols=excluded_cols)


def _base_model_params(
    default_params: dict[str, object] | None,
    model_params: dict[str, object] | None,
) -> dict[str, object]:
    return {**(default_params or DEFAULT_TFT_MODEL_PARAMS), **(model_params or {})}


def _apply_runtime_profile(
    resolved: dict[str, object],
    *,
    explicit_model_params: dict[str, object] | None,
) -> None:
    runtime_profile_name = str(resolved.get("runtime_profile", DEFAULT_RUNTIME_PROFILE_NAME))
    runtime_profile = resolve_runtime_profile(runtime_profile_name)
    explicit_keys = set((explicit_model_params or {}).keys())
    for key, value in runtime_profile.items():
        if key not in explicit_keys:
            resolved[key] = value


def _apply_derived_model_params(
    resolved: dict[str, object],
    *,
    default_max_iter: int,
) -> None:
    if "max_epochs" not in resolved:
        resolved["max_epochs"] = int(cast(Any, resolved.get("max_iter", default_max_iter)))
    if "weight_decay" not in resolved and "l2_regularization" in resolved:
        resolved["weight_decay"] = float(cast(Any, resolved["l2_regularization"]))
    if "hidden_size" not in resolved and "max_depth" in resolved:
        resolved["hidden_size"] = max(8, int(cast(Any, resolved["max_depth"])) * 4)
    if "hidden_continuous_size" not in resolved and "min_samples_leaf" in resolved:
        resolved["hidden_continuous_size"] = max(4, int(cast(Any, resolved["min_samples_leaf"])) // 2)
    if resolved.get("compile_mode") in {None, ""} and bool(cast(Any, resolved.get("torch_compile", False))):
        resolved["compile_mode"] = "reduce-overhead"


def _normalize_runtime_model_params(resolved: dict[str, object]) -> None:
    resolved["max_epochs"] = int(cast(Any, resolved["max_epochs"]))
    resolved["batch_size"] = int(cast(Any, resolved.get("batch_size", DEFAULT_TFT_MODEL_PARAMS["batch_size"])))
    resolved["compile_mode"] = str(cast(Any, resolved.get("compile_mode", DEFAULT_TFT_MODEL_PARAMS["compile_mode"])))
    resolved["determinism_mode"] = str(
        cast(Any, resolved.get("determinism_mode", DEFAULT_TFT_MODEL_PARAMS["determinism_mode"]))
    )
    resolved["torch_compile"] = resolved["compile_mode"] != "off"
    resolved["max_encoder_length"] = int(
        cast(Any, resolved.get("max_encoder_length", DEFAULT_TFT_MODEL_PARAMS["max_encoder_length"]))
    )
    resolved["lstm_layers"] = int(cast(Any, resolved.get("lstm_layers", DEFAULT_TFT_MODEL_PARAMS["lstm_layers"])))
    resolved["patience"] = int(cast(Any, resolved.get("patience", DEFAULT_TFT_MODEL_PARAMS["patience"])))
    resolved["quantiles"] = list(cast(Any, resolved.get("quantiles", DEFAULT_TFT_MODEL_PARAMS["quantiles"])))
    resolved["devices"] = int(cast(Any, resolved["devices"]))
    resolved["num_workers"] = int(cast(Any, resolved["num_workers"]))
    resolved["pin_memory"] = bool(cast(Any, resolved["pin_memory"]))
    resolved["persistent_workers"] = bool(cast(Any, resolved["persistent_workers"]))
    resolved["precision"] = str(cast(Any, resolved["precision"]))
    resolved["matmul_precision"] = str(cast(Any, resolved["matmul_precision"]))
    resolved["tensorboard_logdir"] = (
        None
        if resolved.get("tensorboard_logdir") in {None, ""}
        else str(cast(Any, resolved["tensorboard_logdir"]))
    )


def resolve_model_params(
    model_params: dict[str, object] | None,
    *,
    default_params: dict[str, object] | None,
    default_max_iter: int,
) -> dict[str, object]:
    resolved = _base_model_params(default_params, model_params)
    _apply_runtime_profile(resolved, explicit_model_params=model_params)
    _apply_derived_model_params(resolved, default_max_iter=default_max_iter)
    _normalize_runtime_model_params(resolved)
    return resolved
