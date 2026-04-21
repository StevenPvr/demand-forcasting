from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from praedixa.demand_forecast.backends.tft.artifacts import FittedTFTModel
from praedixa.demand_forecast.backends.tft.feature_mapping import TFTLayout
from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    PREDICTION_ROW_ID_COL,
    TIME_IDX_COL,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    DEFAULT_RUNTIME_PROFILE_NAME,
)
from praedixa.demand_forecast.backends.tft.system_info import (
    collect_tft_system_info,
    resolve_git_sha,
)


def _history_tail_frame(frame: pd.DataFrame, *, max_encoder_length: int) -> pd.DataFrame:
    return frame.groupby(GROUP_COL, sort=False).tail(max_encoder_length).copy().reset_index(drop=True)


def _tensor_values(value: Any) -> list[float]:
    if not hasattr(value, "detach"):
        return np.asarray(value, dtype=float).reshape(-1).tolist()
    detached = value.detach()
    float_tensor = getattr(detached, "float", None)
    if callable(float_tensor):
        detached = float_tensor()
    detached_any = cast(Any, detached)
    return np.asarray(detached_any.cpu().numpy(), dtype=float).reshape(-1).tolist()


def _named_variable_weights(columns: list[str], raw_values: Any) -> dict[str, float]:
    values = _tensor_values(raw_values)
    named = {
        column: float(values[index])
        for index, column in enumerate(columns[: len(values)])
    }
    return dict(sorted(named.items(), key=lambda item: item[1], reverse=True))


def _interpretability_columns(layout: TFTLayout, target_col: str) -> tuple[list[str], list[str], list[str]]:
    static_columns = [*layout["static_categoricals"], *layout["static_reals"]]
    encoder_columns = [
        *layout["time_varying_known_categoricals"],
        *layout["time_varying_known_reals"],
        *layout["time_varying_unknown_categoricals"],
        *layout["time_varying_unknown_reals"],
        target_col,
    ]
    decoder_columns = [
        *layout["time_varying_known_categoricals"],
        *layout["time_varying_known_reals"],
        *layout["time_varying_unknown_categoricals"],
        *layout["time_varying_unknown_reals"],
    ]
    return static_columns, encoder_columns, decoder_columns


def extract_interpretability_payload(
    *,
    model: Any,
    validation_loader: Any,
    resolved_params: dict[str, object],
    layout: TFTLayout,
    target_col: str,
) -> dict[str, Any] | None:
    if validation_loader is None:
        return None
    prediction = model.predict(
        validation_loader,
        mode="raw",
        return_x=True,
        trainer_kwargs={
            "accelerator": str(cast(Any, resolved_params["accelerator"])),
            "devices": int(cast(Any, resolved_params["devices"])),
            "precision": str(cast(Any, resolved_params["precision"])),
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    interpreted = model.interpret_output(prediction.output, reduction="mean")
    static_columns, encoder_columns, decoder_columns = _interpretability_columns(layout, target_col)
    return {
        "reduction": "mean",
        "attention_prediction_horizon": 0,
        "attention": _tensor_values(interpreted["attention"]),
        "encoder_length_histogram": _tensor_values(interpreted["encoder_length_histogram"]),
        "decoder_length_histogram": _tensor_values(interpreted["decoder_length_histogram"]),
        "variable_selection": {
            "static": _named_variable_weights(static_columns, interpreted["static_variables"]),
            "encoder": _named_variable_weights(encoder_columns, interpreted["encoder_variables"]),
            "decoder": _named_variable_weights(decoder_columns, interpreted["decoder_variables"]),
        },
        "top_features": {
            "static": list(_named_variable_weights(static_columns, interpreted["static_variables"]).items())[:5],
            "encoder": list(_named_variable_weights(encoder_columns, interpreted["encoder_variables"]).items())[:10],
            "decoder": list(_named_variable_weights(decoder_columns, interpreted["decoder_variables"]).items())[:10],
        },
    }


def build_fitted_tft_model(
    *,
    model: Any,
    training_dataset: Any,
    training_slice: pd.DataFrame,
    target_col: str,
    resolved_params: dict[str, object],
    best_iteration: int,
    feature_scalers: dict[str, StandardScaler],
    target_scaler: StandardScaler | None,
    normalization_strategy: dict[str, Any],
    runtime_metrics: dict[str, float | int | str | None],
    interpretability_payload: dict[str, Any] | None,
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
        system_info=collect_tft_system_info(
            runtime_profile=runtime_profile,
            determinism_mode=str(cast(Any, resolved_params.get("determinism_mode", "strict"))),
            compile_mode=str(cast(Any, resolved_params.get("compile_mode", "off"))),
        ),
        runtime_metrics=runtime_metrics,
        git_sha=resolve_git_sha(),
        bundle_manifest=None,
        data_hashes={},
        normalization_strategy=normalization_strategy,
        interpretability_payload=interpretability_payload,
        artifact_bundle_version=2,
    )
