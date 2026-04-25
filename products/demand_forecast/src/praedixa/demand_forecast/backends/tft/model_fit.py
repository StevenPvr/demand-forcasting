from __future__ import annotations

import logging
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
from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    PREDICTION_ROW_ID_COL,
    TIME_IDX_COL,
    attach_group_and_time_columns,
    build_combined_frame,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    lazy_import_tft_dependencies,
    resolve_model_params,
    suppress_tft_runtime_noise,
)
from praedixa.demand_forecast.backends.tft.model_predict import predict_with_tft_model
from praedixa.demand_forecast.contracts.targets import (
    reconstruct_absolute_predictions,
    resolve_target_contract,
)
from praedixa.demand_forecast.backends.tft.training_dataset import (
    TrainingDatasetArtifacts,
    build_training_dataset_artifacts,
)
from praedixa.demand_forecast.backends.tft.training_runtime import (
    find_learning_rate,
    fit_trainer_model,
    seed_tft_runtime,
)

LOGGER = logging.getLogger(__name__)
_FLOAT_COMPARISON_EPSILON = 1e-8


def _compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.abs(y_true).sum())
    if abs(denominator) <= _FLOAT_COMPARISON_EPSILON:
        return float("inf")
    return float(np.abs(y_true - y_pred).sum() / denominator)


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
        hidden_continuous_size=int(
            cast(Any, resolved_params["hidden_continuous_size"])
        ),
        lstm_layers=int(cast(Any, resolved_params["lstm_layers"])),
        loss=quantile_loss,
        output_size=len(cast(list[float], resolved_params["quantiles"])),
        log_interval=-1,
        reduce_on_plateau_patience=int(
            cast(Any, resolved_params["reduce_on_plateau_patience"])
        ),
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
    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, **dataloader_kwargs
    )
    valid_loader = (
        None
        if validation_dataset is None
        else validation_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            **dataloader_kwargs,
        )
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
    extra_callbacks: list[Any] | None = None,
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
        extra_callbacks=extra_callbacks,
    )
    return fitted_model, valid_loader, best_iteration, runtime_metrics


def _temporary_fitted_model(
    *,
    model: Any,
    dataset_artifacts: TrainingDatasetArtifacts,
    target_col: str,
    resolved_params: dict[str, object],
) -> FittedTFTModel:
    lightweight_inference_params = dict(resolved_params)
    lightweight_inference_params.update(
        {
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "pin_memory": False,
        }
    )
    return FittedTFTModel(
        model=model,
        dataset_parameters=cast(
            dict[str, Any], dataset_artifacts.training_dataset.get_parameters()
        ),
        history_frame=dataset_artifacts.training_slice,
        group_col=GROUP_COL,
        time_idx_col=TIME_IDX_COL,
        target_col=target_col,
        prediction_row_id_col=PREDICTION_ROW_ID_COL,
        batch_size=int(cast(Any, resolved_params["batch_size"])),
        quantiles=list(cast(Any, resolved_params["quantiles"])),
        best_iteration=0,
        model_hyperparameters=lightweight_inference_params,
        feature_scalers=dataset_artifacts.feature_scalers,
        target_scaler=None,
        runtime_profile=str(
            cast(Any, resolved_params.get("runtime_profile", "local_cpu"))
        ),
        system_info={},
        runtime_metrics={},
        git_sha=None,
        bundle_manifest=None,
        data_hashes={},
        normalization_strategy=dataset_artifacts.normalization_strategy,
        interpretability_payload=None,
        artifact_bundle_version=2,
    )


def _business_validation_metrics_payload(
    *,
    valid_frame: pd.DataFrame,
    raw_predictions: np.ndarray,
    target_col: str,
    train_frame: pd.DataFrame,
) -> dict[str, float]:
    target_contract = resolve_target_contract(
        train_frame,
        valid_frame,
        requested_target_col=target_col,
    )
    absolute_predictions = reconstruct_absolute_predictions(
        raw_predictions,
        valid_frame,
        target_contract,
    )
    absolute_target = valid_frame[target_contract.absolute_target_col].to_numpy(
        dtype=float,
        copy=False,
    )
    valid_mask = np.isfinite(absolute_target) & np.isfinite(absolute_predictions)
    if not valid_mask.any():
        return {}
    filtered_target = absolute_target[valid_mask]
    filtered_predictions = absolute_predictions[valid_mask]
    return {
        "business_val_wape": _compute_wape(filtered_target, filtered_predictions),
        "business_val_abs_bias": float(
            np.mean(np.abs(filtered_predictions - filtered_target))
        ),
    }


def _business_validation_logging_callback(
    imports: dict[str, Any],
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_cols: list[str],
    target_col: str,
    dataset_artifacts: TrainingDatasetArtifacts,
    resolved_params: dict[str, object],
) -> Any | None:
    if valid_frame is None or valid_frame.empty:
        return None
    callback_base: type[Any] = cast(type[Any], imports["Callback"])

    def _update_trainer_callback_metrics(
        pl_module: Any,
        trainer: Any, metrics_payload: dict[str, float]
    ) -> None:
        callback_metrics = getattr(trainer, "callback_metrics", None)
        if callback_metrics is None:
            return
        try:
            torch_module = imports.get("torch")
            module_device = getattr(pl_module, "device", None)
            for metric_name, metric_value in metrics_payload.items():
                tensor_factory = None
                if torch_module is not None:
                    tensor_factory = getattr(
                        torch_module,
                        "as_tensor",
                        getattr(torch_module, "tensor", None),
                    )
                if callable(tensor_factory):
                    callback_metrics[metric_name] = tensor_factory(
                        float(metric_value),
                        device=module_device,
                    )
                else:
                    callback_metrics[metric_name] = float(metric_value)
        except (TypeError, KeyError, AttributeError):
            return

    class _BusinessValidationLoggingCallback(callback_base):
        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            transient_model = _temporary_fitted_model(
                model=pl_module,
                dataset_artifacts=dataset_artifacts,
                target_col=target_col,
                resolved_params=resolved_params,
            )
            raw_predictions = predict_with_tft_model(
                transient_model,
                valid_frame,
                feature_cols,
                fallback_policy="raise",
            )
            business_metrics = _business_validation_metrics_payload(
                valid_frame=valid_frame,
                raw_predictions=raw_predictions,
                target_col=target_col,
                train_frame=train_frame,
            )
            if not business_metrics:
                return
            _update_trainer_callback_metrics(pl_module, trainer, business_metrics)
            LOGGER.info(
                "TFT business validation metrics: epoch=%s business_val_wape=%.6f business_val_abs_bias=%.6f",
                int(getattr(trainer, "current_epoch", 0)),
                float(business_metrics["business_val_wape"]),
                float(business_metrics["business_val_abs_bias"]),
            )

    return _BusinessValidationLoggingCallback()


def calibrate_tft_learning_rate(
    *,
    dataset_artifacts: TrainingDatasetArtifacts,
    resolved_params: dict[str, object],
) -> float:
    imports = lazy_import_tft_dependencies()
    with suppress_tft_runtime_noise():
        seed_tft_runtime(imports, resolved_params)
        train_loader, valid_loader = _training_loaders(
            training_dataset=dataset_artifacts.training_dataset,
            validation_dataset=dataset_artifacts.validation_dataset,
            batch_size=int(cast(Any, resolved_params["batch_size"])),
            resolved_params=resolved_params,
        )
        model = _build_model(
            imports,
            dataset_artifacts.training_dataset,
            resolved_params,
        )
        return find_learning_rate(
            imports,
            model=model,
            resolved_params=resolved_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
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
        business_metrics_callback = None
        if bool(
            cast(
                Any,
                resolved_params.get("enable_business_validation_metrics", True),
            )
        ):
            business_metrics_callback = _business_validation_logging_callback(
                imports,
                train_frame=train_frame,
                valid_frame=valid_frame,
                feature_cols=feature_cols,
                target_col=target_col,
                dataset_artifacts=resolved_dataset_artifacts,
                resolved_params=resolved_params,
            )
        model, valid_loader, best_iteration, runtime_metrics = _fit_resolved_model(
            imports=imports,
            dataset_artifacts=resolved_dataset_artifacts,
            resolved_params=resolved_params,
            extra_callbacks=(
                [] if business_metrics_callback is None else [business_metrics_callback]
            ),
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
