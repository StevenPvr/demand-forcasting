from __future__ import annotations

import logging
import os
import platform
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.feature_screening.addon_selection_execution import (
    run_parallel_addon_selection,
    run_sequential_addon_selection,
)
from praedixa.demand_forecast.feature_screening.addon_selection_models import AddonSelectionContext
from praedixa.demand_forecast.feature_screening.constants import (
    DEFAULT_PROGRESS_LOG_EVERY,
    DEFAULT_TARGET_COL,
)
from praedixa.demand_forecast.feature_screening.metrics import compute_wape_improvement_pct
from praedixa.demand_forecast.feature_screening.tuning import fit_and_score_tft_model


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def resolve_parallelism(
    requested_workers: int | None,
    candidate_count: int,
    *,
    logger: logging.Logger,
    total_threads: int | None = None,
) -> tuple[int, int]:
    available_threads = max(1, total_threads or (os.cpu_count() or 1))
    if _is_macos():
        logger.info(
            "macOS detected: forcing outer parallelism to 1 worker for la stabilite du backend TFT, using all cores inside each fit."
        )
        return 1, available_threads
    max_workers = max(1, requested_workers or available_threads)
    resolved_workers = max(1, min(max_workers, max(1, candidate_count)))
    threads_per_worker = max(1, available_threads // resolved_workers)
    return resolved_workers, threads_per_worker


def score_candidate_task(
    *,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate: str,
    target_col: str,
    model_params: dict[str, object] | None,
    threads_per_worker: int,
    base_mean_wape: float,
    base_scores: list[float],
    logger: logging.Logger,
) -> dict[str, object]:
    candidate_scores = fit_and_score_tft_model(
        frame=frame,
        folds=folds,
        feature_cols=[*base_feature_cols, candidate],
        logger=logger,
        target_col=target_col,
        model_params=model_params,
        num_threads_override=threads_per_worker,
    )
    return candidate_result_payload(candidate, candidate_scores, base_mean_wape, base_scores)


def score_candidate_task_memmap(
    *,
    base_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate: str,
    candidate_values: np.ndarray,
    target_col: str,
    model_params: dict[str, object] | None,
    threads_per_worker: int,
    base_mean_wape: float,
    base_scores: list[float],
    logger: logging.Logger,
) -> dict[str, object]:
    candidate_frame = base_frame.copy(deep=False)
    candidate_frame[candidate] = candidate_values
    candidate_scores = fit_and_score_tft_model(
        frame=candidate_frame,
        folds=folds,
        feature_cols=[*base_feature_cols, candidate],
        logger=logger,
        target_col=target_col,
        model_params=model_params,
        num_threads_override=threads_per_worker,
    )
    return candidate_result_payload(candidate, candidate_scores, base_mean_wape, base_scores)


def candidate_result_payload(
    candidate: str,
    candidate_scores: list[float],
    base_mean_wape: float,
    base_scores: list[float],
) -> dict[str, object]:
    candidate_mean_wape = float(np.mean(candidate_scores))
    wape_improvement_pct = compute_wape_improvement_pct(base_mean_wape, candidate_mean_wape)
    return {
        "feature": candidate,
        "base_mean_wape": base_mean_wape,
        "candidate_mean_wape": candidate_mean_wape,
        "gain_vs_base_model": base_mean_wape - candidate_mean_wape,
        "wape_improvement_pct": wape_improvement_pct,
        "keep": wape_improvement_pct > 0.0,
        "base_fold_wape_scores": [float(score) for score in base_scores],
        "candidate_fold_wape_scores": [float(score) for score in candidate_scores],
    }


def _score_base_model(
    *,
    base_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object] | None,
    total_threads: int,
    logger: logging.Logger,
) -> tuple[list[float], float]:
    base_scores = fit_and_score_tft_model(
        frame=base_frame,
        folds=folds,
        feature_cols=base_feature_cols,
        logger=logger,
        target_col=target_col,
        model_params=model_params,
        num_threads_override=total_threads,
    )
    return base_scores, float(np.mean(base_scores))


def _log_base_model_summary(
    *,
    base_feature_cols: list[str],
    candidate_cols: list[str],
    resolved_workers: int,
    threads_per_worker: int,
    base_mean_wape: float,
    logger: logging.Logger,
) -> None:
    logger.info(
        "Base TFT model scored: feature_count=%s mean_wape=%.6f candidate_count=%s workers=%s threads_per_worker=%s",
        len(base_feature_cols),
        base_mean_wape,
        len(candidate_cols),
        resolved_workers,
        threads_per_worker,
    )


def _run_addon_selection(
    *,
    context: AddonSelectionContext,
    resolved_workers: int,
) -> tuple[list[str], list[dict[str, object]]]:
    if resolved_workers == 1:
        return run_sequential_addon_selection(context)
    return run_parallel_addon_selection(context=context, resolved_workers=resolved_workers)


def run_single_addon_lag_selection_memmap(
    *,
    base_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate_cols: list[str],
    candidate_matrix: np.memmap,
    candidate_to_index: dict[str, int],
    logger: logging.Logger,
    target_col: str = DEFAULT_TARGET_COL,
    model_params: dict[str, object] | None = None,
    num_workers: int | None = None,
    checkpoint_every: int = DEFAULT_PROGRESS_LOG_EVERY,
    checkpoint_callback: Callable[[list[dict[str, object]], list[str]], None] | None = None,
) -> tuple[list[str], pd.DataFrame, dict[str, object]]:
    total_threads, resolved_workers, context = prepare_addon_selection(
        base_frame=base_frame,
        folds=folds,
        base_feature_cols=base_feature_cols,
        candidate_cols=candidate_cols,
        candidate_matrix=candidate_matrix,
        candidate_to_index=candidate_to_index,
        target_col=target_col,
        model_params=model_params,
        num_workers=num_workers,
        checkpoint_every=checkpoint_every,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
    )
    selected_features, report_rows = _run_addon_selection(context=context, resolved_workers=resolved_workers)
    report = addon_selection_report(report_rows)
    base_model_report = base_model_summary(base_feature_cols, context, resolved_workers, total_threads)
    logger.info("Single add-on selection complete: selected=%s/%s", len(selected_features), len(candidate_cols))
    return selected_features, report, base_model_report


def addon_runtime_settings(
    *,
    model_params: dict[str, object] | None,
    num_workers: int | None,
    candidate_count: int,
    logger: logging.Logger,
) -> tuple[int, int, int]:
    total_threads = int(cast(Any, (model_params or {}).get("n_jobs", os.cpu_count() or 1)))
    resolved_workers, threads_per_worker = resolve_parallelism(
        requested_workers=num_workers,
        candidate_count=candidate_count,
        logger=logger,
        total_threads=total_threads,
    )
    return total_threads, resolved_workers, threads_per_worker


def prepare_addon_selection(
    *,
    base_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    candidate_cols: list[str],
    candidate_matrix: np.memmap,
    candidate_to_index: dict[str, int],
    target_col: str,
    model_params: dict[str, object] | None,
    num_workers: int | None,
    checkpoint_every: int,
    checkpoint_callback: Callable[[list[dict[str, object]], list[str]], None] | None,
    logger: logging.Logger,
) -> tuple[int, int, AddonSelectionContext]:
    total_threads, resolved_workers, threads_per_worker = addon_runtime_settings(model_params=model_params, num_workers=num_workers, candidate_count=len(candidate_cols), logger=logger)
    base_scores, base_mean_wape = scored_addon_state(
        base_frame=base_frame,
        folds=folds,
        base_feature_cols=base_feature_cols,
        target_col=target_col,
        model_params=model_params,
        total_threads=total_threads,
        logger=logger,
    )
    _log_base_model_summary(
        base_feature_cols=base_feature_cols,
        candidate_cols=candidate_cols,
        resolved_workers=resolved_workers,
        threads_per_worker=threads_per_worker,
        base_mean_wape=base_mean_wape,
        logger=logger,
    )
    return total_threads, resolved_workers, AddonSelectionContext(
        base_frame=base_frame,
        folds=folds,
        base_feature_cols=base_feature_cols,
        candidate_cols=candidate_cols,
        candidate_matrix=candidate_matrix,
        candidate_to_index=candidate_to_index,
        target_col=target_col,
        model_params=model_params,
        threads_per_worker=threads_per_worker,
        base_mean_wape=base_mean_wape,
        base_scores=base_scores,
        checkpoint_every=checkpoint_every,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
    )


def scored_addon_state(
    *,
    base_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    base_feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object] | None,
    total_threads: int,
    logger: logging.Logger,
) -> tuple[list[float], float]:
    return _score_base_model(
        base_frame=base_frame,
        folds=folds,
        base_feature_cols=base_feature_cols,
        target_col=target_col,
        model_params=model_params,
        total_threads=total_threads,
        logger=logger,
    )


def addon_selection_report(report_rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(report_rows).sort_values(by="wape_improvement_pct", ascending=False).reset_index(drop=True)


def base_model_summary(
    base_feature_cols: list[str],
    context: AddonSelectionContext,
    resolved_workers: int,
    total_threads: int,
) -> dict[str, object]:
    return {
        "base_feature_count": len(base_feature_cols),
        "base_mean_wape": context.base_mean_wape,
        "base_fold_wape_scores": [float(score) for score in context.base_scores],
        "num_workers": resolved_workers,
        "threads_per_worker": context.threads_per_worker,
        "total_threads": total_threads,
    }
