from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.artifacts import FittedTFTModel
from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.backends.tft.dataloader_profile import build_tft_dataloader_kwargs
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


def _prepare_prediction_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    fitted_model: FittedTFTModel,
) -> pd.DataFrame:
    prediction_frame = attach_prediction_row_ids(frame)
    prediction_frame[fitted_model.target_col] = 0.0
    prediction_frame = prepare_split_frame(prediction_frame, split_name="predict", weights=None)
    prepared = attach_group_and_time_columns(prediction_frame, feature_cols)
    history_max_time = fitted_model.history_frame.groupby(
        fitted_model.group_col,
        sort=False,
    )[fitted_model.time_idx_col].max()
    prepared[fitted_model.time_idx_col] = (
        prepared.groupby(fitted_model.group_col, sort=False).cumcount()
        + prepared[fitted_model.group_col].map(history_max_time).fillna(-1).astype(np.int32)
        + 1
    )
    if fitted_model.dataset_parameters.get("weight") is not None and WEIGHT_COL not in prepared.columns:
        prepared[WEIGHT_COL] = 1.0
    return prepared.reset_index(drop=True)


def _prediction_output_to_numpy(prediction: Any) -> np.ndarray:
    resolved = np.asarray(prediction.output.detach().cpu().numpy(), dtype=float)
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


def _inverse_target_scaler(values: np.ndarray, *, fitted_model: FittedTFTModel) -> np.ndarray:
    if fitted_model.target_scaler is None or values.size == 0:
        return np.asarray(values, dtype=float)
    inverse = cast(Any, fitted_model.target_scaler).inverse_transform(values.reshape(-1, 1))
    return np.asarray(inverse, dtype=float).reshape(-1)


def _inverse_target_scaler_quantiles(values: np.ndarray, *, fitted_model: FittedTFTModel) -> np.ndarray:
    if fitted_model.target_scaler is None or values.size == 0:
        return np.asarray(values, dtype=float)
    inverse = cast(Any, fitted_model.target_scaler).inverse_transform(values.reshape(-1, 1))
    return np.asarray(inverse, dtype=float).reshape(values.shape)


def _prediction_dataset(
    imports: dict[str, Any],
    *,
    fitted_model: FittedTFTModel,
    prepared_future: pd.DataFrame,
) -> Any:
    combined_frame = defragment_frame(pd.concat([fitted_model.history_frame, prepared_future], ignore_index=True))
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
        persistent_workers=bool(fitted_model.model_hyperparameters["persistent_workers"]),
    )
    return prediction_dataset.to_dataloader(train=False, batch_size=fitted_model.batch_size, **dataloader_kwargs)


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


def _quantile_column_name(quantile: float) -> str:
    percentage = format(float(quantile) * 100.0, "g").replace(".", "_")
    return f"prediction_p{percentage}"


def predict_with_tft_model(
    fitted_model: FittedTFTModel,
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    raise_if_tft_backend_required("tft_model_utils.predict_with_tft_model")
    if frame.empty:
        return np.asarray([], dtype=float)
    imports = lazy_import_tft_dependencies()
    prepared_future = _prepare_prediction_frame(frame, feature_cols=feature_cols, fitted_model=fitted_model)
    with suppress_tft_runtime_noise():
        prediction_dataset = _prediction_dataset(imports, fitted_model=fitted_model, prepared_future=prepared_future)
        prediction_loader = _prediction_dataloader(prediction_dataset, fitted_model=fitted_model)
        prediction = fitted_model.model.predict(
            prediction_loader,
            mode="prediction",
            return_index=True,
            trainer_kwargs=_prediction_trainer_kwargs(fitted_model),
        )
    prediction_index = _prediction_index_frame(prediction)
    prediction_index["prediction"] = np.asarray(_prediction_output_to_numpy(prediction), dtype=float).reshape(-1)
    aligned = _aligned_prediction_frame(
        prepared_future=prepared_future,
        prediction_payload=prediction_index.loc[
            :,
            [fitted_model.group_col, fitted_model.time_idx_col, "prediction"],
        ],
        fitted_model=fitted_model,
    )
    if aligned["prediction"].isna().any():
        raise RuntimeError("Prediction alignment failed for one or more TFT forecast rows.")
    ordered = aligned.sort_values(fitted_model.prediction_row_id_col)["prediction"].to_numpy(dtype=float)
    return _inverse_target_scaler(ordered, fitted_model=fitted_model)


def predict_quantiles_with_tft_model(
    fitted_model: FittedTFTModel,
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    raise_if_tft_backend_required("tft_model_utils.predict_quantiles_with_tft_model")
    quantile_columns = [_quantile_column_name(quantile) for quantile in fitted_model.quantiles]
    if frame.empty:
        return pd.DataFrame(columns=quantile_columns)
    imports = lazy_import_tft_dependencies()
    prepared_future = _prepare_prediction_frame(frame, feature_cols=feature_cols, fitted_model=fitted_model)
    with suppress_tft_runtime_noise():
        prediction_dataset = _prediction_dataset(imports, fitted_model=fitted_model, prepared_future=prepared_future)
        prediction_loader = _prediction_dataloader(prediction_dataset, fitted_model=fitted_model)
        prediction = fitted_model.model.predict(
            prediction_loader,
            mode="quantiles",
            return_index=True,
            trainer_kwargs=_prediction_trainer_kwargs(fitted_model),
        )
    predicted_values = _inverse_target_scaler_quantiles(
        _prediction_output_to_numpy(prediction),
        fitted_model=fitted_model,
    )
    if predicted_values.ndim != 2 or predicted_values.shape[1] != len(quantile_columns):
        raise RuntimeError("Unexpected TFT quantile prediction shape during alignment.")
    prediction_index = _prediction_index_frame(prediction)
    prediction_payload = pd.concat(
        [
            prediction_index.loc[:, [fitted_model.group_col, fitted_model.time_idx_col]],
            pd.DataFrame(predicted_values, columns=quantile_columns),
        ],
        axis=1,
    )
    aligned = _aligned_prediction_frame(
        prepared_future=prepared_future,
        prediction_payload=prediction_payload,
        fitted_model=fitted_model,
    )
    if aligned[quantile_columns].isna().any().any():
        raise RuntimeError("Quantile prediction alignment failed for one or more TFT forecast rows.")
    return aligned.sort_values(fitted_model.prediction_row_id_col).loc[:, quantile_columns].reset_index(drop=True)


def save_tft_model(
    fitted_model: FittedTFTModel,
    output_path: str | Path,
) -> Path:
    from praedixa.demand_forecast.backends.tft.checkpoint_io import save_tft_checkpoint_bundle

    return save_tft_checkpoint_bundle(fitted_model, output_path)


def load_tft_model(input_path: str | Path) -> FittedTFTModel:
    from praedixa.demand_forecast.backends.tft.checkpoint_io import load_tft_checkpoint_bundle

    return load_tft_checkpoint_bundle(input_path)
