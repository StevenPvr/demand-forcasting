from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from praedixa.demand_forecast.backends.tft.artifacts import FittedTFTModel
from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.backends.tft.frame_utils import (
    WEIGHT_COL,
    defragment_frame,
)

_SUPPRESSED_TFT_WARNING_PATTERNS: tuple[str, ...] = (
    r"Attribute 'loss' is an instance of `nn\.Module` and is already saved during checkpointing\.",
    r"Attribute 'logging_metrics' is an instance of `nn\.Module` and is already saved during checkpointing\.",
    r"Min encoder length and/or min_prediction_idx and/or min prediction length and/or lags are too large for .* series/groups.*",
)
_SUPPRESSED_TFT_LOGGERS: tuple[str, ...] = (
    "lightning.fabric.utilities.seed",
    "lightning.pytorch.utilities.rank_zero",
)

_METADATA_SUFFIX = ".metadata.json"
_HISTORY_SUFFIX = ".history.parquet"


def _lazy_import_tft_dependencies() -> dict[str, Any]:
    import torch
    from lightning.pytorch import seed_everything
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss

    return {
        "GroupNormalizer": GroupNormalizer,
        "NaNLabelEncoder": NaNLabelEncoder,
        "torch": torch,
        "seed_everything": seed_everything,
        "TimeSeriesDataSet": TimeSeriesDataSet,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "QuantileLoss": QuantileLoss,
    }


@contextmanager
def _suppress_tft_runtime_noise() -> Any:
    logger_states = [
        (
            logger_name,
            logging.getLogger(logger_name),
            logging.getLogger(logger_name).level,
        )
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


def _seed_tft_runtime(
    imports: dict[str, Any], model_hyperparameters: dict[str, Any]
) -> None:
    random_state = int(model_hyperparameters.get("random_state", 7))
    imports["seed_everything"](random_state, workers=True)
    warn_only = (
        str(model_hyperparameters.get("determinism_mode", "strict")) == "warn_only"
    )
    imports["torch"].use_deterministic_algorithms(True, warn_only=warn_only)
    set_matmul_precision = getattr(
        imports["torch"], "set_float32_matmul_precision", None
    )
    if callable(set_matmul_precision):
        set_matmul_precision(
            str(model_hyperparameters.get("matmul_precision", "highest"))
        )


def _build_model(
    imports: dict[str, Any],
    training_dataset: Any,
    model_hyperparameters: dict[str, Any],
) -> Any:
    quantile_loss = imports["QuantileLoss"](
        quantiles=model_hyperparameters["quantiles"]
    )
    return imports["TemporalFusionTransformer"].from_dataset(
        training_dataset,
        learning_rate=float(model_hyperparameters["learning_rate"]),
        hidden_size=int(model_hyperparameters["hidden_size"]),
        attention_head_size=int(model_hyperparameters["attention_head_size"]),
        dropout=float(model_hyperparameters["dropout"]),
        hidden_continuous_size=int(model_hyperparameters["hidden_continuous_size"]),
        lstm_layers=int(model_hyperparameters.get("lstm_layers", 1)),
        loss=quantile_loss,
        output_size=len(cast(list[float], model_hyperparameters["quantiles"])),
        log_interval=-1,
        reduce_on_plateau_patience=int(
            model_hyperparameters.get(
                "reduce_on_plateau_patience",
                model_hyperparameters["patience"],
            )
        ),
        optimizer="adam",
        weight_decay=float(model_hyperparameters.get("weight_decay", 0.0)),
    )


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + _METADATA_SUFFIX)


def _history_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + _HISTORY_SUFFIX)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        dict_value = cast(dict[Any, Any], value)
        return {str(key): _json_safe(nested) for key, nested in dict_value.items()}
    if isinstance(value, list):
        list_value = cast(list[Any], value)
        return [_json_safe(nested) for nested in list_value]
    if isinstance(value, tuple):
        tuple_value = cast(tuple[Any, ...], value)
        return [_json_safe(nested) for nested in tuple_value]
    if isinstance(value, np.ndarray):
        return [_json_safe(nested) for nested in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _scaler_attr(scaler: StandardScaler, name: str, default: Any) -> Any:
    return getattr(cast(Any, scaler), name, default)


def _serialize_standard_scaler(scaler: StandardScaler) -> dict[str, Any]:
    return {
        "with_mean": bool(_scaler_attr(scaler, "with_mean", True)),
        "with_std": bool(_scaler_attr(scaler, "with_std", True)),
        "copy": bool(_scaler_attr(scaler, "copy", True)),
        "feature_names_in": _json_safe(_scaler_attr(scaler, "feature_names_in_", [])),
        "n_features_in": int(_scaler_attr(scaler, "n_features_in_", 0)),
        "n_samples_seen": _json_safe(_scaler_attr(scaler, "n_samples_seen_", 0)),
        "mean": _json_safe(_scaler_attr(scaler, "mean_", [])),
        "var": _json_safe(_scaler_attr(scaler, "var_", [])),
        "scale": _json_safe(_scaler_attr(scaler, "scale_", [])),
    }


def _deserialize_standard_scaler(payload: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler(
        copy=bool(payload.get("copy", True)),
        with_mean=bool(payload.get("with_mean", True)),
        with_std=bool(payload.get("with_std", True)),
    )
    setattr(scaler, "n_features_in_", int(payload["n_features_in"]))
    setattr(scaler, "n_samples_seen_", np.float64(payload["n_samples_seen"]))
    setattr(scaler, "mean_", np.asarray(payload["mean"], dtype=float))
    setattr(scaler, "var_", np.asarray(payload["var"], dtype=float))
    setattr(scaler, "scale_", np.asarray(payload["scale"], dtype=float))
    feature_names = payload.get("feature_names_in", [])
    if feature_names:
        setattr(scaler, "feature_names_in_", np.asarray(feature_names, dtype=object))
    return scaler


def _serialize_nan_label_encoder(encoder: Any) -> dict[str, Any]:
    return {
        "add_nan": bool(getattr(encoder, "add_nan", False)),
        "warn": bool(getattr(encoder, "warn", True)),
        "classes": _json_safe(getattr(encoder, "classes_", {})),
        "classes_vector": _json_safe(getattr(encoder, "classes_vector_", [])),
    }


def _deserialize_nan_label_encoder(
    imports: dict[str, Any], payload: dict[str, Any]
) -> Any:
    encoder = imports["NaNLabelEncoder"](
        add_nan=bool(payload.get("add_nan", False)),
        warn=bool(payload.get("warn", True)),
    )
    encoder.classes_ = {
        str(key): int(value)
        for key, value in cast(dict[str, Any], payload.get("classes", {})).items()
    }
    encoder.classes_vector_ = np.asarray(payload.get("classes_vector", []))
    return encoder


def _serialize_group_normalizer(normalizer: Any) -> dict[str, Any]:
    norm_frame = cast(pd.DataFrame, getattr(normalizer, "norm_", pd.DataFrame()))
    return {
        "groups": _json_safe(getattr(normalizer, "groups", [])),
        "_groups": _json_safe(
            getattr(normalizer, "_groups", getattr(normalizer, "groups", []))
        ),
        "scale_by_group": bool(getattr(normalizer, "scale_by_group", False)),
        "method": str(getattr(normalizer, "method", "standard")),
        "center": bool(getattr(normalizer, "center", True)),
        "transformation": _json_safe(getattr(normalizer, "transformation", None)),
        "method_kwargs": _json_safe(getattr(normalizer, "method_kwargs", {})),
        "_method_kwargs": _json_safe(getattr(normalizer, "_method_kwargs", {})),
        "norm_index": _json_safe(norm_frame.index.tolist()),
        "norm_columns": _json_safe(norm_frame.columns.tolist()),
        "norm_values": _json_safe(norm_frame.to_numpy(dtype=float)),
        "missing": _json_safe(getattr(normalizer, "missing_", {})),
    }


def _deserialize_group_normalizer(
    imports: dict[str, Any], payload: dict[str, Any]
) -> Any:
    normalizer = imports["GroupNormalizer"](
        groups=list(cast(list[Any], payload.get("groups", []))),
        method=str(payload.get("method", "standard")),
        center=bool(payload.get("center", True)),
        scale_by_group=bool(payload.get("scale_by_group", False)),
        transformation=payload.get("transformation"),
    )
    normalizer._groups = list(
        cast(list[Any], payload.get("_groups", payload.get("groups", [])))
    )
    normalizer.method_kwargs = dict(
        cast(dict[str, Any], payload.get("method_kwargs", {}))
    )
    normalizer._method_kwargs = dict(
        cast(dict[str, Any], payload.get("_method_kwargs", {}))
    )
    normalizer.norm_ = pd.DataFrame(
        payload.get("norm_values", []),
        index=payload.get("norm_index", []),
        columns=payload.get("norm_columns", []),
    )
    normalizer.missing_ = dict(cast(dict[str, Any], payload.get("missing", {})))
    return normalizer


def _serialize_dataset_parameters(dataset_parameters: dict[str, Any]) -> dict[str, Any]:
    serialized = {
        key: _json_safe(value)
        for key, value in dataset_parameters.items()
        if key not in {"target_normalizer", "categorical_encoders", "scalers"}
    }
    serialized["target_normalizer"] = _serialize_group_normalizer(
        dataset_parameters["target_normalizer"]
    )
    serialized["categorical_encoders"] = {
        column: _serialize_nan_label_encoder(encoder)
        for column, encoder in cast(
            dict[str, Any], dataset_parameters.get("categorical_encoders", {})
        ).items()
    }
    serialized["scalers"] = {
        column: _serialize_standard_scaler(scaler)
        for column, scaler in cast(
            dict[str, Any], dataset_parameters.get("scalers", {})
        ).items()
        if isinstance(scaler, StandardScaler)
    }
    return serialized


def _deserialize_dataset_parameters(
    imports: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    parameters = {
        key: value
        for key, value in payload.items()
        if key not in {"target_normalizer", "categorical_encoders", "scalers"}
    }
    parameters["target_normalizer"] = _deserialize_group_normalizer(
        imports,
        cast(dict[str, Any], payload["target_normalizer"]),
    )
    parameters["categorical_encoders"] = {
        column: _deserialize_nan_label_encoder(imports, cast(dict[str, Any], encoder))
        for column, encoder in cast(
            dict[str, Any], payload.get("categorical_encoders", {})
        ).items()
    }
    parameters["scalers"] = {
        column: _deserialize_standard_scaler(cast(dict[str, Any], scaler))
        for column, scaler in cast(dict[str, Any], payload.get("scalers", {})).items()
    }
    return parameters


def save_tft_checkpoint_bundle(
    fitted_model: FittedTFTModel,
    output_path: str | Path,
) -> Path:
    raise_if_tft_backend_required("tft_checkpoint_io.save_tft_checkpoint_bundle")

    imports = _lazy_import_tft_dependencies()
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_model.history_frame.to_parquet(
        _history_path(resolved_output_path), index=False
    )
    imports["torch"].save(fitted_model.model.state_dict(), resolved_output_path)
    metadata = {
        "payload_version": int(fitted_model.artifact_bundle_version),
        "dataset_parameters": _serialize_dataset_parameters(
            fitted_model.dataset_parameters
        ),
        "history_frame_path": _history_path(resolved_output_path).name,
        "group_col": fitted_model.group_col,
        "time_idx_col": fitted_model.time_idx_col,
        "target_col": fitted_model.target_col,
        "prediction_row_id_col": fitted_model.prediction_row_id_col,
        "batch_size": fitted_model.batch_size,
        "quantiles": fitted_model.quantiles,
        "best_iteration": fitted_model.best_iteration,
        "model_hyperparameters": fitted_model.model_hyperparameters,
        "feature_scalers": _serialize_dataset_parameters(
            fitted_model.dataset_parameters
        ).get("scalers", {}),
        "target_scaler": _serialize_standard_scaler(fitted_model.target_scaler)
        if fitted_model.target_scaler is not None
        else None,
        "runtime_profile": fitted_model.runtime_profile,
        "system_info": fitted_model.system_info,
        "runtime_metrics": fitted_model.runtime_metrics,
        "git_sha": fitted_model.git_sha,
        "bundle_manifest": fitted_model.bundle_manifest,
        "data_hashes": fitted_model.data_hashes,
        "normalization_strategy": fitted_model.normalization_strategy,
        "interpretability_payload": fitted_model.interpretability_payload,
        "artifact_bundle_version": fitted_model.artifact_bundle_version,
    }
    _metadata_path(resolved_output_path).write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_output_path


def _normalization_strategy_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    strategy = payload.get("normalization_strategy")
    if isinstance(strategy, dict):
        return dict(cast(dict[str, Any], strategy))
    raise RuntimeError("TFT checkpoint metadata is missing `normalization_strategy`.")


def _rebuild_model_from_payload(payload: dict[str, Any]) -> Any:
    imports = _lazy_import_tft_dependencies()
    dataset_parameters = cast(dict[str, Any], payload["dataset_parameters"])
    history_frame = cast(pd.DataFrame, payload["history_frame"])
    resolved_params = cast(dict[str, Any], payload["model_hyperparameters"])
    if "scalers" not in dataset_parameters and "feature_scalers" in payload:
        dataset_parameters = {
            **dataset_parameters,
            "scalers": payload["feature_scalers"],
        }
    reconstruction_frame = history_frame.copy()
    future_rows = (
        reconstruction_frame.groupby(str(payload["group_col"]), sort=False)
        .tail(1)
        .copy()
    )
    future_rows[str(payload["time_idx_col"])] = (
        future_rows[str(payload["time_idx_col"])].astype(int) + 1
    )
    future_rows[str(payload["target_col"])] = 0.0
    if (
        dataset_parameters.get("weight") is not None
        and WEIGHT_COL not in future_rows.columns
    ):
        future_rows[WEIGHT_COL] = 1.0
    reconstruction_frame = defragment_frame(
        pd.concat([reconstruction_frame, future_rows], ignore_index=True)
    )
    with _suppress_tft_runtime_noise():
        _seed_tft_runtime(imports, resolved_params)
        training_dataset = imports["TimeSeriesDataSet"].from_parameters(
            dataset_parameters,
            reconstruction_frame,
            stop_randomization=True,
        )
        model = _build_model(imports, training_dataset, resolved_params)
        model.load_state_dict(payload["state_dict"])
        model.eval()
    payload["dataset_parameters"] = dataset_parameters
    return model


def load_tft_checkpoint_bundle(input_path: str | Path) -> FittedTFTModel:
    raise_if_tft_backend_required("tft_checkpoint_io.load_tft_checkpoint_bundle")

    imports = _lazy_import_tft_dependencies()
    resolved_input_path = Path(input_path)
    metadata_path = _metadata_path(resolved_input_path)
    if not metadata_path.exists():
        raise RuntimeError(
            "TFT checkpoint metadata sidecar is missing; refusing unsafe pickle load. "
            "Re-save the model with the v2 checkpoint writer."
        )
    metadata = cast(
        dict[str, Any],
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )
    payload: dict[str, Any] = {
        **metadata,
        "state_dict": imports["torch"].load(
            resolved_input_path,
            map_location="cpu",
            weights_only=True,
        ),
        "dataset_parameters": _deserialize_dataset_parameters(
            imports,
            cast(dict[str, Any], metadata["dataset_parameters"]),
        ),
        "target_scaler": _deserialize_standard_scaler(
            cast(dict[str, Any], metadata["target_scaler"])
        )
        if metadata.get("target_scaler") is not None
        else None,
        "history_frame": pd.read_parquet(
            resolved_input_path.parent / str(metadata["history_frame_path"])
        ),
    }
    model = _rebuild_model_from_payload(payload)
    dataset_parameters = cast(dict[str, Any], payload["dataset_parameters"])
    history_frame = cast(pd.DataFrame, payload["history_frame"])
    feature_scalers = cast(dict[str, Any], dataset_parameters.get("scalers", {}))
    return FittedTFTModel(
        model=model,
        dataset_parameters=dataset_parameters,
        history_frame=history_frame,
        group_col=str(payload["group_col"]),
        time_idx_col=str(payload["time_idx_col"]),
        target_col=str(payload["target_col"]),
        prediction_row_id_col=str(payload["prediction_row_id_col"]),
        batch_size=int(payload["batch_size"]),
        quantiles=list(cast(list[float], payload["quantiles"])),
        best_iteration=int(payload["best_iteration"]),
        model_hyperparameters=cast(dict[str, Any], payload["model_hyperparameters"]),
        feature_scalers=feature_scalers,
        target_scaler=payload.get("target_scaler"),
        runtime_profile=str(payload.get("runtime_profile", "unknown")),
        system_info=cast(dict[str, Any], payload.get("system_info", {})),
        runtime_metrics=cast(dict[str, Any], payload.get("runtime_metrics", {})),
        git_sha=cast(str | None, payload.get("git_sha")),
        bundle_manifest=cast(dict[str, Any] | None, payload.get("bundle_manifest")),
        data_hashes=cast(dict[str, str], payload.get("data_hashes", {})),
        normalization_strategy=_normalization_strategy_from_payload(payload),
        interpretability_payload=cast(
            dict[str, Any] | None, payload.get("interpretability_payload")
        ),
        artifact_bundle_version=int(
            payload.get("artifact_bundle_version", payload.get("payload_version", 1))
        ),
    )
