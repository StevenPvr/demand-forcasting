from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any, cast
import warnings

import pandas as pd

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


def _lazy_import_tft_dependencies() -> dict[str, Any]:
    import torch
    from lightning.pytorch import seed_everything
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    return {
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


def save_tft_checkpoint_bundle(
    fitted_model: FittedTFTModel,
    output_path: str | Path,
) -> Path:
    raise_if_tft_backend_required("tft_checkpoint_io.save_tft_checkpoint_bundle")

    imports = _lazy_import_tft_dependencies()
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "payload_version": int(fitted_model.artifact_bundle_version),
        "state_dict": fitted_model.model.state_dict(),
        "dataset_parameters": fitted_model.dataset_parameters,
        "history_frame": fitted_model.history_frame,
        "group_col": fitted_model.group_col,
        "time_idx_col": fitted_model.time_idx_col,
        "target_col": fitted_model.target_col,
        "prediction_row_id_col": fitted_model.prediction_row_id_col,
        "batch_size": fitted_model.batch_size,
        "quantiles": fitted_model.quantiles,
        "best_iteration": fitted_model.best_iteration,
        "model_hyperparameters": fitted_model.model_hyperparameters,
        "feature_scalers": fitted_model.feature_scalers,
        "target_scaler": fitted_model.target_scaler,
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
    imports["torch"].save(payload, resolved_output_path)
    return resolved_output_path


def _normalization_strategy_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    strategy = payload.get("normalization_strategy")
    if isinstance(strategy, dict):
        return dict(cast(dict[str, Any], strategy))
    return {
        "kind": "legacy_external_standard_scaler"
        if payload.get("target_scaler") is not None
        else "dataset_native",
        "legacy_external_scaler": payload.get("target_scaler") is not None,
    }


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
    payload = cast(
        dict[str, Any],
        imports["torch"].load(Path(input_path), map_location="cpu", weights_only=False),
    )
    model = _rebuild_model_from_payload(payload)
    return FittedTFTModel(
        model=model,
        dataset_parameters=cast(dict[str, Any], payload["dataset_parameters"]),
        history_frame=cast(pd.DataFrame, payload["history_frame"]),
        group_col=str(payload["group_col"]),
        time_idx_col=str(payload["time_idx_col"]),
        target_col=str(payload["target_col"]),
        prediction_row_id_col=str(payload["prediction_row_id_col"]),
        batch_size=int(payload["batch_size"]),
        quantiles=list(cast(list[float], payload["quantiles"])),
        best_iteration=int(payload["best_iteration"]),
        model_hyperparameters=cast(dict[str, Any], payload["model_hyperparameters"]),
        feature_scalers=cast(dict[str, Any], payload.get("feature_scalers", {})),
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
