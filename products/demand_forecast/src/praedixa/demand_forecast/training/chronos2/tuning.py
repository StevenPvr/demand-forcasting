from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd

from praedixa.demand_forecast.backends.chronos2.model import load_chronos2_pipeline
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_CHRONOS2_FINETUNE_BATCH_SIZE_CHOICES,
    DEFAULT_CHRONOS2_FINETUNE_CONTEXT_LENGTH_CHOICES,
    DEFAULT_CHRONOS2_FINETUNE_DEVICE_MAP,
    DEFAULT_CHRONOS2_FINETUNE_LEARNING_RATE_RANGE,
    DEFAULT_CHRONOS2_FINETUNE_LORA_ALPHA_CHOICES,
    DEFAULT_CHRONOS2_FINETUNE_LORA_DROPOUT_RANGE,
    DEFAULT_CHRONOS2_FINETUNE_LORA_R_CHOICES,
    DEFAULT_CHRONOS2_FINETUNE_MODE,
    DEFAULT_CHRONOS2_FINETUNE_MODEL_PATH,
    DEFAULT_CHRONOS2_FINETUNE_NUM_STEPS_CHOICES,
    DEFAULT_TARGET_TRANSFORM,
    DEFAULT_TUNING_RANDOM_SEED,
    DEFAULT_TUNING_TRIALS,
)

_DATE_COL = "dt"
_SERIES_COL = "client_id"
_LOCATION_COL = "location_id"
_PRODUCT_COL = "product_id"
_CHRONOS_ID_COL = "series_id"
_CHRONOS_TIMESTAMP_COL = "timestamp"
_CHRONOS_TARGET_COL = "target"
_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


def optimize_chronos2_finetune_params(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    baseline_wape: float,
    target_contract: TargetContract,
    *,
    baseline_dataset_wape: dict[str, float] | None = None,
    logger: logging.Logger,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_TUNING_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    _ = baseline_dataset_wape, target_transform
    resolved_params = _resolved_model_params(model_params)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        study_name="praedixa_chronos2_finetune",
    )
    objective = _objective(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=_numeric_feature_cols(train_frame, feature_cols),
        target_col=target_contract.learning_target_col,
        model_params=resolved_params,
        logger=logger,
    )
    study.optimize(objective, n_trials=tuning_trials, show_progress_bar=False)
    best_params = {**resolved_params, **study.best_trial.params}
    return (
        best_params,
        _trials_report(study=study, baseline_wape=baseline_wape),
        _runtime_metadata(
            model_params=resolved_params,
            feature_cols=feature_cols,
            numeric_feature_cols=_numeric_feature_cols(train_frame, feature_cols),
            trial_count=tuning_trials,
            random_seed=random_seed,
        ),
    )


def _objective(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
    logger: logging.Logger,
) -> Any:
    def _score(trial: optuna.trial.Trial) -> float:
        trial_params = _sample_trial_params(trial, model_params=model_params)
        fold_scores: list[float] = []
        for fold in folds:
            train_idx = cast(np.ndarray, fold["train_idx"])
            valid_idx = cast(np.ndarray, fold["valid_idx"])
            fit_frame = pd.concat(
                [train_frame, tuning_frame.iloc[train_idx]],
                ignore_index=True,
            )
            valid_frame = tuning_frame.iloc[valid_idx].copy()
            fold_scores.append(
                _fit_score_fold(
                    fit_frame=fit_frame,
                    valid_frame=valid_frame,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    trial_params=trial_params,
                    logger=logger,
                )
            )
        score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_wape", fold_scores)
        return score

    return _score


def _fit_score_fold(
    *,
    fit_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    trial_params: dict[str, object],
    logger: logging.Logger,
) -> float:
    _configure_torch_single_job()
    with tempfile.TemporaryDirectory(prefix="praedixa_chronos2_finetune_") as tmp_dir:
        pipeline = load_chronos2_pipeline(trial_params)
        finetuned = pipeline.fit(
            inputs=_chronos_tasks(
                fit_frame,
                feature_cols=feature_cols,
                target_col=target_col,
            ),
            prediction_length=_prediction_length(valid_frame),
            finetune_mode=str(trial_params["finetune_mode"]),
            lora_config=_lora_config(trial_params),
            context_length=_int_param(trial_params, "context_length"),
            learning_rate=_float_param(trial_params, "learning_rate"),
            num_steps=_int_param(trial_params, "num_steps"),
            batch_size=_int_param(trial_params, "batch_size"),
            output_dir=Path(tmp_dir),
            remove_printer_callback=True,
            disable_data_parallel=True,
            dataloader_num_workers=0,
            save_strategy="no",
            eval_strategy="no",
            logging_strategy="no",
        )
        prediction = _predict_fold(
            pipeline=finetuned,
            context_frame=fit_frame,
            future_frame=valid_frame,
            feature_cols=feature_cols,
            target_col=target_col,
            trial_params=trial_params,
        )
    score = _wape(valid_frame[target_col].to_numpy(dtype=float), prediction)
    logger.info("Chronos-2 fine-tune fold scored: wape=%.6f", score)
    return score


def _predict_fold(
    *,
    pipeline: Any,
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    trial_params: dict[str, object],
) -> np.ndarray:
    context_frame = build_chronos2_scoring_context(context_frame, future_frame)
    prediction_length = _prediction_length(future_frame)
    forecast = pipeline.predict_df(
        _chronos_df(context_frame, feature_cols, target_col=target_col),
        future_df=build_padded_future_chronos_df(
            future_frame,
            feature_cols,
            prediction_length=prediction_length,
        ),
        prediction_length=prediction_length,
        quantile_levels=list(_QUANTILE_LEVELS),
        id_column=_CHRONOS_ID_COL,
        timestamp_column=_CHRONOS_TIMESTAMP_COL,
        target=_CHRONOS_TARGET_COL,
        batch_size=_int_param(trial_params, "batch_size"),
        context_length=_int_param(trial_params, "context_length"),
        cross_learning=True,
        validate_inputs=False,
    )
    return _align_predictions(future_frame, forecast)


def build_chronos2_scoring_context(
    context_frame: pd.DataFrame, future_frame: pd.DataFrame
) -> pd.DataFrame:
    future_series = set(_series_key(future_frame).astype(str).tolist())
    context_series = _series_key(context_frame).astype(str)
    scoped = context_frame.loc[context_series.isin(future_series)].copy()
    if scoped.empty:
        raise ValueError("Chronos-2 scoring context has no history for future series.")
    return scoped


def _chronos_tasks(
    frame: pd.DataFrame, *, feature_cols: list[str], target_col: str
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    ordered = _ordered_frame(frame)
    for _, group in ordered.groupby("_chronos_series_key", sort=False):
        task: dict[str, object] = {
            "target": group[target_col].to_numpy(dtype=np.float32, copy=True)
        }
        covariates = {
            column: group[column].to_numpy(dtype=np.float32, copy=True)
            for column in feature_cols
            if column in group.columns
        }
        if covariates:
            task["past_covariates"] = covariates
        tasks.append(task)
    return tasks


def _chronos_df(
    frame: pd.DataFrame, feature_cols: list[str], *, target_col: str | None
) -> pd.DataFrame:
    columns: dict[str, object] = {
        _CHRONOS_ID_COL: _series_values(frame),
        _CHRONOS_TIMESTAMP_COL: pd.to_datetime(frame[_DATE_COL]),
    }
    if target_col is not None:
        columns[_CHRONOS_TARGET_COL] = frame[target_col].to_numpy(
            dtype=float,
            copy=True,
        )
    for column in feature_cols:
        if column in frame.columns:
            columns[column] = frame[column].to_numpy(dtype=float, copy=True)
    output = pd.DataFrame(columns, copy=False)
    return output.sort_values([_CHRONOS_ID_COL, _CHRONOS_TIMESTAMP_COL]).reset_index(
        drop=True
    )


def build_padded_future_chronos_df(
    frame: pd.DataFrame, feature_cols: list[str], *, prediction_length: int
) -> pd.DataFrame:
    chronos_frame = _chronos_df(frame, feature_cols, target_col=None)
    padded_groups = [
        _pad_future_group(group, feature_cols, prediction_length=prediction_length)
        for _, group in chronos_frame.groupby(_CHRONOS_ID_COL, sort=False)
    ]
    return pd.concat(padded_groups, ignore_index=True).sort_values(
        [_CHRONOS_ID_COL, _CHRONOS_TIMESTAMP_COL]
    )


def _pad_future_group(
    group: pd.DataFrame, feature_cols: list[str], *, prediction_length: int
) -> pd.DataFrame:
    ordered = group.sort_values(_CHRONOS_TIMESTAMP_COL).reset_index(drop=True)
    missing_rows = prediction_length - len(ordered)
    if missing_rows <= 0:
        return ordered.head(prediction_length)
    last_row = {str(key): value for key, value in ordered.iloc[-1].to_dict().items()}
    last_timestamp = pd.Timestamp(last_row[_CHRONOS_TIMESTAMP_COL])
    padded_rows: list[dict[str, object]] = []
    for step in range(1, missing_rows + 1):
        row = dict(last_row)
        row[_CHRONOS_TIMESTAMP_COL] = last_timestamp + pd.Timedelta(days=step)
        for column in feature_cols:
            if column in row and pd.isna(row[column]):
                row[column] = 0.0
        padded_rows.append(row)
    return pd.concat([ordered, pd.DataFrame(padded_rows)], ignore_index=True)


def _align_predictions(
    source_frame: pd.DataFrame, forecast: pd.DataFrame
) -> np.ndarray:
    expected = pd.DataFrame(
        {
            _CHRONOS_ID_COL: _series_values(source_frame),
            _CHRONOS_TIMESTAMP_COL: pd.to_datetime(source_frame[_DATE_COL]),
            "_row_order": np.arange(len(source_frame), dtype=np.int64),
        }
    )
    aligned = expected.merge(
        forecast.assign(
            **{_CHRONOS_TIMESTAMP_COL: pd.to_datetime(forecast[_CHRONOS_TIMESTAMP_COL])}
        ),
        on=[_CHRONOS_ID_COL, _CHRONOS_TIMESTAMP_COL],
        how="left",
    ).sort_values("_row_order")
    return aligned["0.5"].fillna(0.0).to_numpy(dtype=float)


def _sample_trial_params(
    trial: optuna.trial.Trial, *, model_params: dict[str, object]
) -> dict[str, object]:
    lr_low, lr_high = DEFAULT_CHRONOS2_FINETUNE_LEARNING_RATE_RANGE
    dropout_low, dropout_high = DEFAULT_CHRONOS2_FINETUNE_LORA_DROPOUT_RANGE
    return {
        **model_params,
        "context_length": trial.suggest_categorical(
            "context_length", list(DEFAULT_CHRONOS2_FINETUNE_CONTEXT_LENGTH_CHOICES)
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", list(DEFAULT_CHRONOS2_FINETUNE_BATCH_SIZE_CHOICES)
        ),
        "num_steps": trial.suggest_categorical(
            "num_steps", list(DEFAULT_CHRONOS2_FINETUNE_NUM_STEPS_CHOICES)
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", lr_low, lr_high, log=True
        ),
        "lora_r": trial.suggest_categorical(
            "lora_r", list(DEFAULT_CHRONOS2_FINETUNE_LORA_R_CHOICES)
        ),
        "lora_alpha": trial.suggest_categorical(
            "lora_alpha", list(DEFAULT_CHRONOS2_FINETUNE_LORA_ALPHA_CHOICES)
        ),
        "lora_dropout": trial.suggest_float("lora_dropout", dropout_low, dropout_high),
    }


def _resolved_model_params(model_params: dict[str, object] | None) -> dict[str, object]:
    return {
        "model_path": DEFAULT_CHRONOS2_FINETUNE_MODEL_PATH,
        "device_map": DEFAULT_CHRONOS2_FINETUNE_DEVICE_MAP,
        "finetune_mode": DEFAULT_CHRONOS2_FINETUNE_MODE,
        **(model_params or {}),
    }


def _configure_torch_single_job() -> None:
    try:
        import torch
    except ImportError:
        return
    set_num_threads = getattr(torch, "set_num_threads", None)
    if callable(set_num_threads):
        set_num_threads(1)
    set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
    try:
        if callable(set_num_interop_threads):
            set_num_interop_threads(1)
    except RuntimeError:
        return


def _int_param(params: dict[str, object], key: str) -> int:
    value = params[key]
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Chronos-2 parameter `{key}` must be numeric.")
    return int(value)


def _float_param(params: dict[str, object], key: str) -> float:
    value = params[key]
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Chronos-2 parameter `{key}` must be numeric.")
    return float(value)


def _lora_config(trial_params: dict[str, object]) -> dict[str, object] | None:
    if str(trial_params["finetune_mode"]) != "lora":
        return None
    return {
        "r": _int_param(trial_params, "lora_r"),
        "lora_alpha": _int_param(trial_params, "lora_alpha"),
        "lora_dropout": _float_param(trial_params, "lora_dropout"),
        "target_modules": [
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    }


def _numeric_feature_cols(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [
        column
        for column in feature_cols
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _series_key(frame: pd.DataFrame) -> pd.Series:
    if _SERIES_COL in frame.columns:
        return frame[_SERIES_COL].astype(str)
    return frame[_LOCATION_COL].astype(str) + "__" + frame[_PRODUCT_COL].astype(str)


def _series_values(frame: pd.DataFrame) -> pd.Series:
    return _series_key(frame)


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered[_DATE_COL] = pd.to_datetime(ordered[_DATE_COL])
    ordered["_chronos_series_key"] = _series_key(ordered)
    return ordered.sort_values(["_chronos_series_key", _DATE_COL])


def _prediction_length(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 1
    return int(frame.groupby(_series_key(frame), sort=False)[_DATE_COL].nunique().max())


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.abs(y_true).sum())
    if denominator <= 1e-8:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.abs(y_true - y_pred).sum() / denominator)


def _trials_report(*, study: optuna.Study, baseline_wape: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        value = float(trial.value) if trial.value is not None else float("nan")
        rows.append(
            {
                "trial": int(trial.number),
                "mean_wape": value,
                "baseline_wape_improvement_pct": (
                    (baseline_wape - value) / baseline_wape if baseline_wape else 0.0
                ),
                **trial.params,
            }
        )
    return pd.DataFrame(rows)


def _runtime_metadata(
    *,
    model_params: dict[str, object],
    feature_cols: list[str],
    numeric_feature_cols: list[str],
    trial_count: int,
    random_seed: int,
) -> dict[str, object]:
    return {
        "backend": "chronos2_finetune",
        "protocol": "chronos2_lora_finetune_bakery",
        "trial_count": trial_count,
        "random_seed": random_seed,
        "model_params": model_params,
        "feature_count": len(feature_cols),
        "numeric_covariate_count": len(numeric_feature_cols),
    }


__all__ = [
    "build_chronos2_scoring_context",
    "build_padded_future_chronos_df",
    "optimize_chronos2_finetune_params",
]
