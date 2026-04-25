from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.artifacts import FittedTFTModel
from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.backends.tft.dataloader_profile import (
    build_tft_dataloader_kwargs,
)
from praedixa.demand_forecast.backends.tft.frame_utils import (
    attach_group_and_time_columns,
    attach_prediction_row_ids,
    defragment_frame,
    prepare_split_frame,
    WEIGHT_COL,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    lazy_import_tft_dependencies,
    suppress_tft_runtime_noise,
)


_BASELINE_POINT_CANDIDATES: tuple[str, ...] = (
    "target_same_dow_mean_4w",
    "same_dow_mean_4w",
    "seasonal_naive_d7",
    "target_seasonal_naive_d7",
    "target_lag_7",
    "sale_amount_lag_7",
    "lag_7",
    "current_day_demand_qty",
    "observed_demand_qty",
)

FallbackPolicy = Literal["raise", "baseline"]


def _record_prediction_diagnostics(
    fitted_model: FittedTFTModel,
    *,
    used_fallback: bool,
    fallback_reason: str | None,
    fallback_rows: int,
) -> None:
    fitted_model.runtime_metrics["last_prediction_diagnostics"] = {
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "fallback_rows": fallback_rows,
    }


def _raise_or_fallback_point_predictions(
    frame: pd.DataFrame,
    *,
    fitted_model: FittedTFTModel,
    fallback_policy: FallbackPolicy,
    reason: str,
) -> np.ndarray:
    if fallback_policy == "raise":
        raise RuntimeError(reason)
    _record_prediction_diagnostics(
        fitted_model,
        used_fallback=True,
        fallback_reason=reason,
        fallback_rows=len(frame),
    )
    return _fallback_point_predictions(frame)


def _raise_or_fallback_quantile_predictions(
    frame: pd.DataFrame,
    *,
    fitted_model: FittedTFTModel,
    quantile_columns: list[str],
    fallback_policy: FallbackPolicy,
    reason: str,
) -> pd.DataFrame:
    if fallback_policy == "raise":
        raise RuntimeError(reason)
    _record_prediction_diagnostics(
        fitted_model,
        used_fallback=True,
        fallback_reason=reason,
        fallback_rows=len(frame),
    )
    return _fallback_quantile_predictions(frame, quantile_columns=quantile_columns)


def _prepare_prediction_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    fitted_model: FittedTFTModel,
) -> pd.DataFrame:
    prediction_frame = attach_prediction_row_ids(frame)
    prediction_frame[fitted_model.target_col] = 0.0
    prediction_frame = prepare_split_frame(
        prediction_frame, split_name="predict", weights=None
    )
    prepared = attach_group_and_time_columns(prediction_frame, feature_cols)
    history_max_time = fitted_model.history_frame.groupby(
        fitted_model.group_col,
        sort=False,
    )[fitted_model.time_idx_col].max()
    prepared[fitted_model.time_idx_col] = (
        prepared.groupby(fitted_model.group_col, sort=False).cumcount()
        + prepared[fitted_model.group_col]
        .map(history_max_time)
        .fillna(-1)
        .astype(np.int32)
        + 1
    )
    if (
        fitted_model.dataset_parameters.get("weight") is not None
        and WEIGHT_COL not in prepared.columns
    ):
        prepared[WEIGHT_COL] = 1.0
    return prepared.reset_index(drop=True)


def _supported_prediction_group_ids(
    fitted_model: FittedTFTModel,
) -> set[str]:
    if (
        fitted_model.history_frame.empty
        or fitted_model.group_col not in fitted_model.history_frame.columns
    ):
        return set()
    max_encoder_length = int(
        fitted_model.model_hyperparameters.get("max_encoder_length", 1)
    )
    group_sizes = fitted_model.history_frame.groupby(
        fitted_model.group_col,
        sort=False,
    ).size()
    return {
        str(group_id)
        for group_id, size in group_sizes.items()
        if int(size) >= max_encoder_length
    }


def _predictable_future_frame(
    prepared_future: pd.DataFrame,
    *,
    fitted_model: FittedTFTModel,
) -> pd.DataFrame:
    supported_group_ids = _supported_prediction_group_ids(fitted_model)
    if not supported_group_ids:
        return prepared_future.iloc[0:0].copy()
    return prepared_future.loc[
        prepared_future[fitted_model.group_col]
        .astype("string")
        .isin(supported_group_ids)
    ].copy()


def _prediction_output_to_numpy(prediction: Any) -> np.ndarray:
    output_tensor = prediction.output.detach()
    float_dtype = getattr(output_tensor, "float", None)
    if callable(float_dtype):
        output_tensor = float_dtype()
    output_tensor_any = cast(Any, output_tensor)
    resolved = np.asarray(output_tensor_any.cpu().numpy(), dtype=float)
    if resolved.ndim == 3 and resolved.shape[1] == 1:
        return resolved[:, 0, :]
    return resolved


def _prediction_trainer_kwargs(fitted_model: FittedTFTModel) -> dict[str, object]:
    return {
        "accelerator": str(fitted_model.model_hyperparameters["accelerator"]),
        "devices": int(fitted_model.model_hyperparameters["devices"]),
        "precision": str(fitted_model.model_hyperparameters["precision"]),
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }


def _inverse_target_scaler(
    values: np.ndarray, *, fitted_model: FittedTFTModel
) -> np.ndarray:
    _ = fitted_model
    return np.asarray(values, dtype=float)


def _inverse_target_scaler_quantiles(
    values: np.ndarray, *, fitted_model: FittedTFTModel
) -> np.ndarray:
    _ = fitted_model
    return np.asarray(values, dtype=float)


def _prediction_dataset(
    imports: dict[str, Any],
    *,
    fitted_model: FittedTFTModel,
    prepared_future: pd.DataFrame,
) -> Any:
    combined_frame = defragment_frame(
        pd.concat([fitted_model.history_frame, prepared_future], ignore_index=True)
    )
    return imports["TimeSeriesDataSet"].from_parameters(
        fitted_model.dataset_parameters,
        combined_frame,
        min_prediction_idx=int(prepared_future[fitted_model.time_idx_col].min()),
        stop_randomization=True,
    )


def _prediction_dataloader(
    prediction_dataset: Any,
    *,
    fitted_model: FittedTFTModel,
) -> Any:
    dataloader_kwargs = build_tft_dataloader_kwargs(
        num_workers=int(fitted_model.model_hyperparameters["num_workers"]),
        pin_memory=bool(fitted_model.model_hyperparameters["pin_memory"]),
        persistent_workers=bool(
            fitted_model.model_hyperparameters["persistent_workers"]
        ),
        prefetch_factor=cast(
            int | None, fitted_model.model_hyperparameters.get("prefetch_factor")
        ),
    )
    return prediction_dataset.to_dataloader(
        train=False, batch_size=fitted_model.batch_size, **dataloader_kwargs
    )


def _prediction_index_frame(prediction: Any) -> pd.DataFrame:
    return cast(pd.DataFrame, prediction.index.copy()).reset_index(drop=True)


def _aligned_prediction_frame(
    *,
    prepared_future: pd.DataFrame,
    prediction_payload: pd.DataFrame,
    fitted_model: FittedTFTModel,
) -> pd.DataFrame:
    return prepared_future.loc[
        :,
        [
            fitted_model.prediction_row_id_col,
            fitted_model.group_col,
            fitted_model.time_idx_col,
        ],
    ].merge(
        prediction_payload,
        on=[fitted_model.group_col, fitted_model.time_idx_col],
        how="left",
        validate="one_to_one",
    )


def _fill_missing_point_predictions(
    aligned: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    fitted_model: FittedTFTModel,
    fallback_policy: FallbackPolicy,
) -> np.ndarray:
    ordered = aligned.sort_values(fitted_model.prediction_row_id_col).reset_index(
        drop=True
    )
    missing_count = int(ordered["prediction"].isna().sum())
    if missing_count and fallback_policy == "raise":
        raise RuntimeError(
            f"TFT prediction alignment missed {missing_count} row(s); fallback is disabled."
        )
    fallback_predictions = _fallback_point_predictions(frame)
    fallback_by_row_id = {
        row_id: float(fallback_predictions[row_id])
        for row_id in range(len(fallback_predictions))
    }
    filled_predictions = (
        ordered["prediction"]
        .astype(float)
        .where(
            ordered["prediction"].notna(),
            ordered[fitted_model.prediction_row_id_col]
            .map(fallback_by_row_id)
            .astype(float),
        )
    )
    return filled_predictions.to_numpy(dtype=float)


def _fill_missing_quantile_predictions(
    aligned: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    fitted_model: FittedTFTModel,
    quantile_columns: list[str],
    fallback_policy: FallbackPolicy,
) -> pd.DataFrame:
    ordered = aligned.sort_values(fitted_model.prediction_row_id_col).reset_index(
        drop=True
    )
    missing_count = int(ordered[quantile_columns].isna().any(axis=1).sum())
    if missing_count and fallback_policy == "raise":
        raise RuntimeError(
            f"TFT quantile prediction alignment missed {missing_count} row(s); fallback is disabled."
        )
    fallback_quantiles = _fallback_quantile_predictions(
        frame, quantile_columns=quantile_columns
    )
    fallback_quantiles[fitted_model.prediction_row_id_col] = np.arange(
        len(fallback_quantiles), dtype=np.int32
    )
    fallback_by_row_id = fallback_quantiles.set_index(
        fitted_model.prediction_row_id_col
    )
    for column in quantile_columns:
        ordered[column] = (
            ordered[column]
            .astype(float)
            .where(
                ordered[column].notna(),
                ordered[fitted_model.prediction_row_id_col]
                .map(fallback_by_row_id[column])
                .astype(float),
            )
        )
    return ordered.loc[:, quantile_columns].reset_index(drop=True)


def _quantile_column_name(quantile: float) -> str:
    percentage = format(float(quantile) * 100.0, "g").replace(".", "_")
    return f"prediction_p{percentage}"


def _baseline_prediction_column(frame: pd.DataFrame) -> str | None:
    for column in _BASELINE_POINT_CANDIDATES:
        if column in frame.columns:
            return column
    return None


def _fallback_point_predictions(frame: pd.DataFrame) -> np.ndarray:
    baseline_column = _baseline_prediction_column(frame)
    if baseline_column is None:
        return np.zeros(len(frame), dtype=float)
    return np.clip(
        pd.to_numeric(frame[baseline_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        a_min=0.0,
        a_max=None,
    )


def _fallback_quantile_predictions(
    frame: pd.DataFrame,
    *,
    quantile_columns: list[str],
) -> pd.DataFrame:
    point_predictions = _fallback_point_predictions(frame)
    return pd.DataFrame(
        {column: point_predictions.copy() for column in quantile_columns}
    )


def _predict_payload(
    *,
    imports: dict[str, Any],
    fitted_model: FittedTFTModel,
    prediction_loader: Any,
    mode: str,
) -> Any:
    fitted_model.model.eval()
    with imports["torch"].inference_mode():
        return fitted_model.model.predict(
            prediction_loader,
            mode=mode,
            return_index=True,
            trainer_kwargs=_prediction_trainer_kwargs(fitted_model),
        )


def predict_with_tft_model(
    fitted_model: FittedTFTModel,
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    fallback_policy: FallbackPolicy = "raise",
) -> np.ndarray:
    raise_if_tft_backend_required("tft_model_utils.predict_with_tft_model")
    if frame.empty:
        return np.asarray([], dtype=float)
    imports = lazy_import_tft_dependencies()
    try:
        prepared_future = _prepare_prediction_frame(
            frame, feature_cols=feature_cols, fitted_model=fitted_model
        )
        predictable_future = _predictable_future_frame(
            prepared_future, fitted_model=fitted_model
        )
        if predictable_future.empty:
            return _raise_or_fallback_point_predictions(
                frame,
                fitted_model=fitted_model,
                fallback_policy=fallback_policy,
                reason="No future rows have enough TFT encoder history.",
            )
        with suppress_tft_runtime_noise():
            prediction_dataset = _prediction_dataset(
                imports,
                fitted_model=fitted_model,
                prepared_future=predictable_future,
            )
            prediction_loader = _prediction_dataloader(
                prediction_dataset, fitted_model=fitted_model
            )
            prediction = _predict_payload(
                imports=imports,
                fitted_model=fitted_model,
                prediction_loader=prediction_loader,
                mode="prediction",
            )
        prediction_index = _prediction_index_frame(prediction)
        prediction_index["prediction"] = np.asarray(
            _prediction_output_to_numpy(prediction), dtype=float
        ).reshape(-1)
        aligned = _aligned_prediction_frame(
            prepared_future=prepared_future,
            prediction_payload=prediction_index.loc[
                :,
                [fitted_model.group_col, fitted_model.time_idx_col, "prediction"],
            ],
            fitted_model=fitted_model,
        )
        ordered = _fill_missing_point_predictions(
            aligned,
            frame=frame,
            fitted_model=fitted_model,
            fallback_policy=fallback_policy,
        )
        missing_count = int(aligned["prediction"].isna().sum())
        _record_prediction_diagnostics(
            fitted_model,
            used_fallback=missing_count > 0,
            fallback_reason="missing_aligned_point_predictions"
            if missing_count
            else None,
            fallback_rows=missing_count,
        )
        return _inverse_target_scaler(ordered, fitted_model=fitted_model)
    except (RuntimeError, ValueError, KeyError) as exc:
        return _raise_or_fallback_point_predictions(
            frame,
            fitted_model=fitted_model,
            fallback_policy=fallback_policy,
            reason=str(exc),
        )


def predict_quantiles_with_tft_model(
    fitted_model: FittedTFTModel,
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    fallback_policy: FallbackPolicy = "raise",
) -> pd.DataFrame:
    raise_if_tft_backend_required("tft_model_utils.predict_quantiles_with_tft_model")
    quantile_columns = [
        _quantile_column_name(quantile) for quantile in fitted_model.quantiles
    ]
    if frame.empty:
        return pd.DataFrame(columns=quantile_columns)
    imports = lazy_import_tft_dependencies()
    try:
        prepared_future = _prepare_prediction_frame(
            frame, feature_cols=feature_cols, fitted_model=fitted_model
        )
        predictable_future = _predictable_future_frame(
            prepared_future, fitted_model=fitted_model
        )
        if predictable_future.empty:
            return _raise_or_fallback_quantile_predictions(
                frame,
                fitted_model=fitted_model,
                quantile_columns=quantile_columns,
                fallback_policy=fallback_policy,
                reason="No future rows have enough TFT encoder history.",
            )
        with suppress_tft_runtime_noise():
            prediction_dataset = _prediction_dataset(
                imports,
                fitted_model=fitted_model,
                prepared_future=predictable_future,
            )
            prediction_loader = _prediction_dataloader(
                prediction_dataset, fitted_model=fitted_model
            )
            prediction = _predict_payload(
                imports=imports,
                fitted_model=fitted_model,
                prediction_loader=prediction_loader,
                mode="quantiles",
            )
        predicted_values = _inverse_target_scaler_quantiles(
            _prediction_output_to_numpy(prediction),
            fitted_model=fitted_model,
        )
        if predicted_values.ndim != 2 or predicted_values.shape[1] != len(
            quantile_columns
        ):
            raise RuntimeError(
                "Unexpected TFT quantile prediction shape during alignment."
            )
        prediction_index = _prediction_index_frame(prediction)
        prediction_payload = pd.concat(
            [
                prediction_index.loc[
                    :, [fitted_model.group_col, fitted_model.time_idx_col]
                ],
                pd.DataFrame(predicted_values, columns=quantile_columns),
            ],
            axis=1,
        )
        aligned = _aligned_prediction_frame(
            prepared_future=prepared_future,
            prediction_payload=prediction_payload,
            fitted_model=fitted_model,
        )
        return _fill_missing_quantile_predictions(
            aligned,
            frame=frame,
            fitted_model=fitted_model,
            quantile_columns=quantile_columns,
            fallback_policy=fallback_policy,
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        return _raise_or_fallback_quantile_predictions(
            frame,
            fitted_model=fitted_model,
            quantile_columns=quantile_columns,
            fallback_policy=fallback_policy,
            reason=str(exc),
        )


def save_tft_model(
    fitted_model: FittedTFTModel,
    output_path: str | Path,
) -> Path:
    from praedixa.demand_forecast.backends.tft.checkpoint_io import (
        save_tft_checkpoint_bundle,
    )

    return save_tft_checkpoint_bundle(fitted_model, output_path)


def load_tft_model(input_path: str | Path) -> FittedTFTModel:
    from praedixa.demand_forecast.backends.tft.checkpoint_io import (
        load_tft_checkpoint_bundle,
    )

    return load_tft_checkpoint_bundle(input_path)
