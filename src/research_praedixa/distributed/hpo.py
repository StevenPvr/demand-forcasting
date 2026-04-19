from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from distributed import as_completed
import optuna
from optuna.trial import TrialState
import pandas as pd

from research_praedixa.distributed.budgets import RuntimeBudget, build_runtime_budget
from research_praedixa.distributed.cluster import connect_client, wait_for_workers
from research_praedixa.distributed.config import DistributedConfig, load_distributed_config
from research_praedixa.distributed.logging import execution_identity
from research_praedixa.distributed.prepare import DEFAULT_DISTRIBUTED_RUNTIME_DIR, load_runtime_manifest
from research_praedixa.distributed.storage import ensure_study
from research_praedixa.memory_utils import read_parquet_projected
from research_praedixa.optimisation.pipeline import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    _drop_constant_feature_columns,
    _fit_and_score_xgboost_on_tuning,
    _sample_optuna_params,
    build_grouped_tuning_walk_forward_folds_by_dataset,
    build_tuning_walk_forward_folds_by_dataset,
    compute_wape_improvement_pct,
    evaluate_statistical_baselines_macro,
)
from research_praedixa.target_utils import build_target_contract_metadata, ensure_learning_target_column, resolve_target_contract
from research_praedixa.xgboost_utils import select_numeric_feature_columns


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimisationRuntimeBundle:
    train_frame: pd.DataFrame
    tuning_frame: pd.DataFrame
    feature_cols: list[str]
    target_contract: Any
    training_folds: list[dict[str, object]]
    baseline_folds: list[dict[str, object]]
    absolute_target_col: str


_OPTIMISATION_RUNTIME_CACHE: dict[tuple[str, str, str, int], OptimisationRuntimeBundle] = {}


def _json_dump(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_optimisation_runtime(
    *,
    train_path: str,
    tuning_path: str,
    requested_target_col: str,
    n_folds: int,
) -> OptimisationRuntimeBundle:
    cache_key = (train_path, tuning_path, requested_target_col, n_folds)
    cached = _OPTIMISATION_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    train_frame = read_parquet_projected(train_path)
    tuning_frame = read_parquet_projected(tuning_path)
    train_frame[DEFAULT_DATE_COL] = pd.to_datetime(train_frame[DEFAULT_DATE_COL])
    tuning_frame[DEFAULT_DATE_COL] = pd.to_datetime(tuning_frame[DEFAULT_DATE_COL])
    if DEFAULT_DATASET_SOURCE_COL not in train_frame.columns:
        train_frame[DEFAULT_DATASET_SOURCE_COL] = "legacy"
    if DEFAULT_DATASET_SOURCE_COL not in tuning_frame.columns:
        tuning_frame[DEFAULT_DATASET_SOURCE_COL] = "legacy"
    train_frame = train_frame.sort_values(DEFAULT_DATE_COL).reset_index(drop=True)
    tuning_frame = tuning_frame.sort_values(DEFAULT_DATE_COL).reset_index(drop=True)
    target_contract = resolve_target_contract(
        train_frame,
        tuning_frame,
        requested_target_col=requested_target_col,
    )
    train_frame = ensure_learning_target_column(train_frame, target_contract)
    tuning_frame = ensure_learning_target_column(tuning_frame, target_contract)
    train_frame = train_frame[train_frame[target_contract.learning_target_col].notna()].copy().reset_index(drop=True)
    tuning_frame = tuning_frame[tuning_frame[target_contract.learning_target_col].notna()].copy().reset_index(drop=True)
    numeric_feature_cols = select_numeric_feature_columns(
        train_frame,
        excluded_cols={
            DEFAULT_DATE_COL,
            target_contract.learning_target_col,
            target_contract.absolute_target_col,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
        },
    )
    identifier_feature_cols = [
        column
        for column in DEFAULT_IDENTIFIER_FEATURE_COLS
        if column in train_frame.columns and column not in numeric_feature_cols
    ]
    feature_cols, _ = _drop_constant_feature_columns(train_frame, [*numeric_feature_cols, *identifier_feature_cols])
    training_folds = build_grouped_tuning_walk_forward_folds_by_dataset(
        tuning_frame,
        date_col=DEFAULT_DATE_COL,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        n_folds=n_folds,
    )
    baseline_folds = build_tuning_walk_forward_folds_by_dataset(
        tuning_frame,
        date_col=DEFAULT_DATE_COL,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        n_folds=n_folds,
    )
    bundle = OptimisationRuntimeBundle(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        training_folds=training_folds,
        baseline_folds=baseline_folds,
        absolute_target_col=target_contract.absolute_target_col,
    )
    _OPTIMISATION_RUNTIME_CACHE[cache_key] = bundle
    return bundle


def _run_single_optuna_trial(
    *,
    config_path: str,
    study_name: str,
    runtime_dir: str,
    requested_target_col: str,
    n_folds: int,
    random_seed: int,
    base_model_params: dict[str, object],
    baseline_wape: float,
) -> dict[str, object]:
    config = load_distributed_config(config_path)
    study = ensure_study(study_name=study_name, config=config)
    trial = study.ask()
    bundle = _load_optimisation_runtime(
        train_path=str(Path(runtime_dir) / "optimisation/train.parquet"),
        tuning_path=str(Path(runtime_dir) / "optimisation/tuning.parquet"),
        requested_target_col=requested_target_col,
        n_folds=n_folds,
    )
    try:
        trial_params = {**base_model_params, **_sample_optuna_params(trial=trial, random_seed=random_seed)}
        trial_params["n_jobs"] = build_runtime_budget(config).xgboost_threads_per_task
        trial_params["max_parallel_fold_workers"] = int(config.pipelines.optimisation.max_parallel_fold_workers)
        tuning_result = _fit_and_score_xgboost_on_tuning(
            train_frame=bundle.train_frame,
            tuning_frame=bundle.tuning_frame,
            folds=bundle.training_folds,
            feature_cols=bundle.feature_cols,
            target_contract=bundle.target_contract,
            model_params=trial_params,
            total_threads=int(trial_params["n_jobs"]),
            target_transform=bundle.target_contract.target_mode,
        )
        mean_wape = float(tuning_result["macro_mean_wape"])
        improvement_pct = compute_wape_improvement_pct(baseline_wape, mean_wape)
        trial.set_user_attr("mean_wape", mean_wape)
        trial.set_user_attr("dataset_mean_wape", tuning_result["dataset_mean_wape"])
        trial.set_user_attr("fold_wape_scores", tuning_result["fold_results"])
        trial.set_user_attr("execution_identity", execution_identity())
        study.tell(trial, improvement_pct, state=TrialState.COMPLETE)
        return {
            "trial": trial.number,
            "mean_wape": mean_wape,
            "improvement_pct": improvement_pct,
            **execution_identity(),
        }
    except Exception:
        study.tell(trial, state=TrialState.FAIL)
        raise


def run_distributed_optuna_search(
    *,
    config: DistributedConfig,
    study_name: str = "praedixa-foundation-xgboost",
    runtime_dir: str | Path = DEFAULT_DISTRIBUTED_RUNTIME_DIR,
    requested_target_col: str = "target_delta_log_wow_d_plus_1",
    random_seed: int = 42,
    base_model_params: dict[str, object] | None = None,
    tuning_trials: int | None = None,
    output_dir: str | Path = "data/optimisation_distributed",
) -> dict[str, Path]:
    runtime_root = (config.repo_path / runtime_dir).resolve()
    output_root = (config.repo_path / output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_runtime_manifest(runtime_root)
    budget = build_runtime_budget(config)
    bundle = _load_optimisation_runtime(
        train_path=str(manifest["optimisation"]["train_path"]),
        tuning_path=str(manifest["optimisation"]["tuning_path"]),
        requested_target_col=requested_target_col,
        n_folds=int(config.pipelines.optimisation.n_folds),
    )
    baseline_rows, best_baseline = evaluate_statistical_baselines_macro(
        bundle.tuning_frame,
        bundle.baseline_folds,
        absolute_target_col=bundle.absolute_target_col,
        target_contract=bundle.target_contract,
    )
    logger.info(
        "Distributed HPO baseline ready: baseline=%s mean_wape=%.6f target_contract=%s",
        best_baseline["baseline_name"],
        best_baseline["mean_wape"],
        build_target_contract_metadata(bundle.target_contract),
    )

    total_trials = int(tuning_trials or config.pipelines.optimisation.tuning_trials)
    study = ensure_study(study_name=study_name, config=config)
    completed_trials = len([trial for trial in study.trials if trial.state == TrialState.COMPLETE])
    remaining_trials = max(0, total_trials - completed_trials)
    client = connect_client(config)
    wait_for_workers(client, minimum_workers=budget.cluster_task_slots)
    pending_futures = []
    completed_metadata: list[dict[str, object]] = []
    submitted = 0
    parallel_tasks = min(budget.optuna_parallel_tasks, remaining_trials or budget.optuna_parallel_tasks)
    future_iterator = as_completed()

    try:
        while submitted < remaining_trials or pending_futures:
            while submitted < remaining_trials and len(pending_futures) < parallel_tasks:
                future = client.submit(
                    _run_single_optuna_trial,
                    config_path=str(config.config_path),
                    study_name=study_name,
                    runtime_dir=str(runtime_root),
                    requested_target_col=requested_target_col,
                    n_folds=int(config.pipelines.optimisation.n_folds),
                    random_seed=random_seed,
                    base_model_params=base_model_params or {},
                    baseline_wape=float(best_baseline["mean_wape"]),
                    pure=False,
                )
                pending_futures.append(future)
                future_iterator.add(future)
                submitted += 1

            for future in future_iterator:
                completed_metadata.append(future.result())
                pending_futures.remove(future)
                break
    finally:
        client.close()

    final_study = ensure_study(study_name=study_name, config=config)
    tuning_rows: list[dict[str, object]] = []
    for trial in final_study.trials:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            continue
        tuning_rows.append(
            {
                "trial": int(trial.number),
                "mean_wape": float(trial.user_attrs["mean_wape"]),
                "baseline_wape_improvement_pct": float(trial.value),
                "fold_wape_scores": trial.user_attrs["fold_wape_scores"],
                "dataset_mean_wape": trial.user_attrs["dataset_mean_wape"],
                "execution_identity": trial.user_attrs.get("execution_identity"),
                **trial.params,
            }
        )
    tuning_report = pd.DataFrame(tuning_rows).sort_values(
        by="baseline_wape_improvement_pct",
        ascending=False,
    ).reset_index(drop=True)
    best_params = {**(base_model_params or {}), **final_study.best_params}
    best_params["random_state"] = random_seed + final_study.best_trial.number + 1
    best_params["verbosity"] = 0
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"
    best_params["tree_method"] = "hist"
    best_params["n_jobs"] = budget.xgboost_threads_per_task
    best_params.pop("max_parallel_fold_workers", None)

    best_params_path = output_root / "best_optuna_params.json"
    trials_report_path = output_root / "optuna_trials_report.csv"
    baseline_report_path = output_root / "baseline_report.json"
    study_summary_path = output_root / "study_summary.json"
    _json_dump(best_params_path, best_params)
    tuning_report.to_csv(trials_report_path, index=False)
    _json_dump(
        baseline_report_path,
        {"best_baseline": best_baseline, "all_baselines": baseline_rows},
    )
    _json_dump(
        study_summary_path,
        {
            "study_name": study_name,
            "storage_url": config.postgres.sqlalchemy_url,
            "completed_trials": len(tuning_rows),
            "requested_trials": total_trials,
            "parallel_tasks": parallel_tasks,
            "worker_execution": completed_metadata,
            "runtime_budget": budget.__dict__,
        },
    )
    logger.info(
        "Distributed Optuna search complete: study=%s completed_trials=%s best_improvement_pct=%.6f",
        study_name,
        len(tuning_rows),
        float(final_study.best_trial.value),
    )
    return {
        "best_params": best_params_path,
        "trials_report": trials_report_path,
        "baseline_report": baseline_report_path,
        "study_summary": study_summary_path,
    }
