from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praedixa.demand_forecast.evaluation.constants import (
    DEFAULT_BEST_PARAMS_PATH,
    DEFAULT_EVALUATION_MODEL_BACKEND,
    DEFAULT_EVALUATION_TRAIN_SAMPLE_FRACTION,
    DEFAULT_EVALUATION_TUNING_SAMPLE_FRACTION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REQUESTED_TARGET_COL,
    DEFAULT_TRAIN_SELECTION_INPUT_PATH,
    DEFAULT_TRAIN_TUNING_INPUT_PATH,
    DEFAULT_VAL_INPUT_PATH,
)
from praedixa.demand_forecast.evaluation.modeling import fit_final_model
from praedixa.demand_forecast.evaluation.modeling import (
    fit_final_xgboost_model,
)
from praedixa.demand_forecast.evaluation.orchestrator_context import (
    ensure_evaluation_target_dir,
)
from praedixa.demand_forecast.evaluation.orchestrator_runtime import (
    run_evaluation_pipeline,
)
from praedixa.demand_forecast.evaluation.refit import (
    evaluate_daily_refit_predictions,
)
from praedixa.demand_forecast.evaluation.xgboost_runtime import (
    evaluate_xgboost_daily_refit_predictions,
    save_xgboost_model,
)
from praedixa.demand_forecast.evaluation.reporting import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residuals_acf_pacf,
    plot_residuals_qq,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DUCKDB_PATH,
    DEFAULT_GOLD_TABLE,
)
from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.backends.xgboost.backend import (
    raise_if_xgboost_backend_required,
)


def save_tft_model(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.backends.tft.model_utils import save_tft_model as impl

    return impl(*args, **kwargs)


def raise_if_chronos2_backend_required(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.backends.chronos2.backend import (
        raise_if_chronos2_backend_required as impl,
    )

    return impl(*args, **kwargs)


def evaluate_chronos2_daily_refit_predictions(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.chronos2_runtime import (
        evaluate_chronos2_daily_refit_predictions as impl,
    )

    return impl(*args, **kwargs)


def fit_final_chronos2_model(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.chronos2_runtime import (
        fit_final_chronos2_model as impl,
    )

    return impl(*args, **kwargs)


def save_chronos2_model(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.chronos2_runtime import (
        save_chronos2_model as impl,
    )

    return impl(*args, **kwargs)


def raise_if_moirai_backend_required(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.backends.moirai.backend import (
        raise_if_moirai_backend_required as impl,
    )

    return impl(*args, **kwargs)


def evaluate_moirai_daily_refit_predictions(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.moirai_runtime import (
        evaluate_moirai_daily_refit_predictions as impl,
    )

    return impl(*args, **kwargs)


def fit_final_moirai_model(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.moirai_runtime import (
        fit_final_moirai_model as impl,
    )

    return impl(*args, **kwargs)


def save_moirai_model(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.moirai_runtime import (
        save_moirai_model as impl,
    )

    return impl(*args, **kwargs)


def raise_if_timesfm_backend_required(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.backends.timesfm.backend import (
        raise_if_timesfm_backend_required as impl,
    )

    return impl(*args, **kwargs)


def evaluate_timesfm_daily_refit_predictions(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.timesfm_runtime import (
        evaluate_timesfm_daily_refit_predictions as impl,
    )

    return impl(*args, **kwargs)


def fit_final_timesfm_model(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.timesfm_runtime import (
        fit_final_timesfm_model as impl,
    )

    return impl(*args, **kwargs)


def save_timesfm_model(*args: Any, **kwargs: Any) -> Any:
    from praedixa.demand_forecast.evaluation.timesfm_runtime import (
        save_timesfm_model as impl,
    )

    return impl(*args, **kwargs)


@dataclass(frozen=True)
class EvaluationBuildRequest:
    train_selection_input_path: str | Path | None = DEFAULT_TRAIN_SELECTION_INPUT_PATH
    train_tuning_input_path: str | Path | None = DEFAULT_TRAIN_TUNING_INPUT_PATH
    val_input_path: str | Path | None = DEFAULT_VAL_INPUT_PATH
    best_params_path: str | Path = DEFAULT_BEST_PARAMS_PATH
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
    target_col: str = DEFAULT_REQUESTED_TARGET_COL
    duckdb_path: str | Path = DEFAULT_DUCKDB_PATH
    gold_table: str = DEFAULT_GOLD_TABLE
    train_sample_fraction: float = DEFAULT_EVALUATION_TRAIN_SAMPLE_FRACTION
    tuning_sample_fraction: float = DEFAULT_EVALUATION_TUNING_SAMPLE_FRACTION
    model_backend: str = DEFAULT_EVALUATION_MODEL_BACKEND
    evaluation_dataset_source: str | None = None
    daily_refit_bakery_weight_multipliers: tuple[float, ...] | None = None
    daily_refit_parallel_workers: int = 1


def build_evaluation_outputs(
    request: EvaluationBuildRequest | None = None,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    resolved_request = request or EvaluationBuildRequest()
    resolved_logger = logger or logging.getLogger(__name__)
    target_dir = ensure_evaluation_target_dir(resolved_request.output_dir)
    backend = resolved_request.model_backend.strip().lower()
    if backend == "xgboost":
        return run_evaluation_pipeline(
            request=resolved_request,
            target_dir=target_dir,
            logger=resolved_logger,
            evaluate_daily_refit_fn=evaluate_xgboost_daily_refit_predictions,
            fit_final_model_fn=fit_final_xgboost_model,
            save_model_fn=save_xgboost_model,
            require_backend_available_fn=raise_if_xgboost_backend_required,
            plot_actual_vs_predicted_fn=plot_actual_vs_predicted,
            plot_residuals_fn=plot_residuals,
            plot_residuals_qq_fn=plot_residuals_qq,
            plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf,
        )
    if backend == "chronos2":
        return run_evaluation_pipeline(
            request=resolved_request,
            target_dir=target_dir,
            logger=resolved_logger,
            evaluate_daily_refit_fn=evaluate_chronos2_daily_refit_predictions,
            fit_final_model_fn=fit_final_chronos2_model,
            save_model_fn=save_chronos2_model,
            require_backend_available_fn=raise_if_chronos2_backend_required,
            plot_actual_vs_predicted_fn=plot_actual_vs_predicted,
            plot_residuals_fn=plot_residuals,
            plot_residuals_qq_fn=plot_residuals_qq,
            plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf,
        )
    if backend == "moirai":
        return run_evaluation_pipeline(
            request=resolved_request,
            target_dir=target_dir,
            logger=resolved_logger,
            evaluate_daily_refit_fn=evaluate_moirai_daily_refit_predictions,
            fit_final_model_fn=fit_final_moirai_model,
            save_model_fn=save_moirai_model,
            require_backend_available_fn=raise_if_moirai_backend_required,
            plot_actual_vs_predicted_fn=plot_actual_vs_predicted,
            plot_residuals_fn=plot_residuals,
            plot_residuals_qq_fn=plot_residuals_qq,
            plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf,
        )
    if backend == "timesfm":
        return run_evaluation_pipeline(
            request=resolved_request,
            target_dir=target_dir,
            logger=resolved_logger,
            evaluate_daily_refit_fn=evaluate_timesfm_daily_refit_predictions,
            fit_final_model_fn=fit_final_timesfm_model,
            save_model_fn=save_timesfm_model,
            require_backend_available_fn=raise_if_timesfm_backend_required,
            plot_actual_vs_predicted_fn=plot_actual_vs_predicted,
            plot_residuals_fn=plot_residuals,
            plot_residuals_qq_fn=plot_residuals_qq,
            plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf,
        )
    if backend != "tft":
        raise ValueError(
            f"Unsupported evaluation model backend `{resolved_request.model_backend}`."
        )
    return run_evaluation_pipeline(
        request=resolved_request,
        target_dir=target_dir,
        logger=resolved_logger,
        evaluate_daily_refit_fn=evaluate_daily_refit_predictions,
        fit_final_model_fn=fit_final_model,
        save_model_fn=save_tft_model,
        require_backend_available_fn=raise_if_tft_backend_required,
        plot_actual_vs_predicted_fn=plot_actual_vs_predicted,
        plot_residuals_fn=plot_residuals,
        plot_residuals_qq_fn=plot_residuals_qq,
        plot_residuals_acf_pacf_fn=plot_residuals_acf_pacf,
    )
