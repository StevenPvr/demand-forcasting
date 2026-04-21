from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Any, cast

from praedixa.demand_forecast.feature_screening.addon_selection_models import AddonSelectionContext


def run_sequential_addon_selection(context: AddonSelectionContext) -> tuple[list[str], list[dict[str, object]]]:
    selected_features: list[str] = []
    report_rows: list[dict[str, object]] = []
    total = len(context.candidate_cols)
    for index, candidate in enumerate(context.candidate_cols, start=1):
        result = score_memmap_candidate(context, candidate)
        append_candidate_result(result=result, candidate=candidate, selected_features=selected_features, report_rows=report_rows)
        maybe_checkpoint(context=context, processed=index, total=total, report_rows=report_rows, selected_features=selected_features)
        maybe_log_progress(
            processed=index,
            total=total,
            selected_count=len(selected_features),
            candidate=candidate,
            improvement_pct=float(cast(Any, result["wape_improvement_pct"])),
            logger=context.logger,
        )
    return selected_features, report_rows


def run_parallel_addon_selection(
    *,
    context: AddonSelectionContext,
    resolved_workers: int,
) -> tuple[list[str], list[dict[str, object]]]:
    completed = 0
    results_by_candidate: dict[str, dict[str, object]] = {}
    total = len(context.candidate_cols)
    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        future_to_candidate = {executor.submit(score_memmap_candidate, context, candidate): candidate for candidate in context.candidate_cols}
        for future in as_completed(future_to_candidate):
            candidate = future_to_candidate[future]
            results_by_candidate[candidate] = future.result()
            completed += 1
            ordered_report = ordered_parallel_report(context.candidate_cols, results_by_candidate)
            selected_features = ordered_parallel_selected_features(context.candidate_cols, results_by_candidate)
            maybe_checkpoint(context=context, processed=completed, total=total, report_rows=ordered_report, selected_features=selected_features)
            maybe_log_progress(
                processed=completed,
                total=total,
                selected_count=len(selected_features),
                candidate=candidate,
                improvement_pct=float(cast(Any, results_by_candidate[candidate]["wape_improvement_pct"])),
                logger=context.logger,
            )
    return ordered_parallel_selected_features(context.candidate_cols, results_by_candidate), ordered_parallel_report(
        context.candidate_cols,
        results_by_candidate,
    )


def ordered_parallel_report(
    candidate_cols: list[str],
    results_by_candidate: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return [results_by_candidate[candidate] for candidate in candidate_cols if candidate in results_by_candidate]


def ordered_parallel_selected_features(
    candidate_cols: list[str],
    results_by_candidate: dict[str, dict[str, object]],
) -> list[str]:
    return [candidate for candidate in candidate_cols if candidate in results_by_candidate and bool(results_by_candidate[candidate]["keep"])]


def maybe_checkpoint(
    *,
    context: AddonSelectionContext,
    processed: int,
    total: int,
    report_rows: list[dict[str, object]],
    selected_features: list[str],
) -> None:
    should_checkpoint = processed == 1 or processed % context.checkpoint_every == 0 or processed == total
    if context.checkpoint_callback and should_checkpoint:
        context.checkpoint_callback(report_rows, selected_features)


def maybe_log_progress(
    *,
    processed: int,
    total: int,
    selected_count: int,
    candidate: str,
    improvement_pct: float,
    logger: logging.Logger,
) -> None:
    if processed == 1 or processed % 25 == 0 or processed == total:
        logger.info(
            "Add-on progress: tested=%s/%s kept=%s last_feature=%s wape_improvement_pct=%.6f",
            processed,
            total,
            selected_count,
            candidate,
            improvement_pct,
        )


def append_candidate_result(
    *,
    result: dict[str, object],
    candidate: str,
    selected_features: list[str],
    report_rows: list[dict[str, object]],
) -> None:
    if bool(result["keep"]):
        selected_features.append(candidate)
    report_rows.append(result)


def score_memmap_candidate(context: AddonSelectionContext, candidate: str) -> dict[str, object]:
    from praedixa.demand_forecast.feature_screening.addon_selection import score_candidate_task_memmap

    return score_candidate_task_memmap(
        base_frame=context.base_frame,
        folds=context.folds,
        base_feature_cols=context.base_feature_cols,
        candidate=candidate,
        candidate_values=context.candidate_matrix[:, context.candidate_to_index[candidate]],
        target_col=context.target_col,
        model_params=context.model_params,
        threads_per_worker=context.threads_per_worker,
        base_mean_wape=context.base_mean_wape,
        base_scores=context.base_scores,
        logger=context.logger,
    )
