from __future__ import annotations

from dataclasses import dataclass
import os

from research_praedixa.distributed.config import DistributedConfig


@dataclass(frozen=True)
class RuntimeBudget:
    total_logical_cores: int
    pro_task_slots: int
    air_task_slots: int
    cluster_task_slots: int
    xgboost_threads_per_task: int
    feature_selection_candidate_batch_size: int
    evaluation_batch_dates_per_task: int
    inference_batch_rows: int
    optuna_parallel_tasks: int


def build_runtime_budget(config: DistributedConfig) -> RuntimeBudget:
    total_cores = int(os.cpu_count() or 1)
    pro_slots = int(config.pro_workers.worker_count * config.pro_workers.dask_threads_per_worker)
    air_slots = int(config.air_workers.worker_count * config.air_workers.dask_threads_per_worker)
    cluster_slots = max(1, pro_slots + air_slots)
    xgboost_threads = max(
        config.pro_workers.xgboost_threads_per_task,
        config.air_workers.xgboost_threads_per_task,
    )
    optuna_parallel_tasks = min(cluster_slots, int(config.pipelines.optimisation.optuna_parallel_tasks))
    return RuntimeBudget(
        total_logical_cores=total_cores,
        pro_task_slots=pro_slots,
        air_task_slots=air_slots,
        cluster_task_slots=cluster_slots,
        xgboost_threads_per_task=xgboost_threads,
        feature_selection_candidate_batch_size=int(config.pipelines.feature_selection.candidate_batch_size),
        evaluation_batch_dates_per_task=int(config.pipelines.evaluation.batch_dates_per_task),
        inference_batch_rows=int(config.pipelines.inference.batch_rows),
        optuna_parallel_tasks=max(1, optuna_parallel_tasks),
    )
