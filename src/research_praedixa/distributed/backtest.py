from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from distributed import as_completed
import pandas as pd

from research_praedixa.distributed.budgets import build_runtime_budget
from research_praedixa.distributed.cluster import connect_client, wait_for_workers
from research_praedixa.distributed.config import DistributedConfig
from research_praedixa.distributed.logging import execution_identity
from research_praedixa.distributed.prepare import DEFAULT_DISTRIBUTED_RUNTIME_DIR, load_runtime_manifest
from research_praedixa.evaluation.bakery_style import (
    DEFAULT_UNIT_COST_EUR,
    REFERENCE_DATE_COL,
    REFERENCE_PRODUCT_COL,
    build_predictions_frame,
    build_simple_economic_gain_payload,
    build_statistical_baselines_payload,
    compute_metrics_payload,
    enrich_predictions_with_best_baseline,
)
from research_praedixa.evaluation.pipeline import (
    _build_diagnostics_payload,
    _fit_evaluation_model,
    _fit_final_model,
    _load_local_mode_frames,
    _plot_actual_vs_predicted,
    _plot_residuals,
    _predict_absolute,
    _prepare_scored_test_frame,
    _prepare_training_target_frame,
    _select_feature_columns,
    build_daily_walk_forward_folds,
    load_best_params,
)
from research_praedixa.memory_utils import read_parquet_projected
from research_praedixa.target_utils import build_target_contract_metadata, resolve_target_contract


logger = logging.getLogger(__name__)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _run_evaluation_fold_batch(
    *,
    train_path: str,
    valid_path: str,
    test_path: str,
    feature_cols: list[str],
    target_contract_metadata: dict[str, str | None],
    model_params: dict[str, object],
    fold_batch: list[dict[str, object]],
) -> dict[str, object]:
    train_frame = read_parquet_projected(train_path)
    valid_frame = read_parquet_projected(valid_path)
    test_frame = read_parquet_projected(test_path)
    target_contract = resolve_target_contract(
        pd.concat([train_frame, valid_frame], ignore_index=True),
        test_frame,
        requested_target_col=str(target_contract_metadata["learning_target_col"]),
    )
    prediction_parts: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    best_iterations: list[int] = []
    base_train_rows = int(len(train_frame))
    base_valid_rows = int(len(valid_frame))

    for fold in fold_batch:
        eval_date = str(fold["eval_date"])
        bakery_history_frame = test_frame.iloc[fold["history_idx"]].copy()
        refit_train_frame = pd.concat([train_frame, bakery_history_frame], ignore_index=True)
        known_history_rows = int(base_train_rows + base_valid_rows + len(bakery_history_frame))
        model, best_iteration = _fit_evaluation_model(
            train_frame=refit_train_frame,
            valid_frame=valid_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
            model_params=model_params,
        )
        best_iterations.append(best_iteration)
        fold_test_frame = test_frame.iloc[fold["valid_idx"]].copy()
        fold_reference = fold_test_frame.loc[:, [REFERENCE_DATE_COL, REFERENCE_PRODUCT_COL]].copy()
        absolute_predictions = _predict_absolute(
            model,
            fold_test_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
        )
        fold_predictions = build_predictions_frame(
            fold_reference,
            actual=fold_test_frame[target_contract.absolute_target_col],
            prediction_raw=absolute_predictions,
            train_rows_used=known_history_rows,
        )
        prediction_parts.append(fold_predictions)
        absolute_target = fold_test_frame[target_contract.absolute_target_col].astype(float)
        prediction_series = fold_predictions["prediction_raw"].astype(float)
        reports.append(
            {
                "fold": int(fold["fold"]),
                "eval_date": eval_date,
                "history_rows": known_history_rows,
                "bakery_history_rows": int(len(bakery_history_frame)),
                "valid_rows": int(len(fold_predictions)),
                "best_iteration": int(best_iteration),
                "mae": float((absolute_target - prediction_series).abs().mean()),
                "rmse": float((((absolute_target - prediction_series) ** 2).mean()) ** 0.5),
            }
        )
    predictions_df = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    return {
        "predictions": predictions_df.to_dict(orient="records"),
        "reports": reports,
        "best_iterations": best_iterations,
        "execution_identity": execution_identity(),
    }


def run_distributed_backtest(
    *,
    config: DistributedConfig,
    runtime_dir: str | Path = DEFAULT_DISTRIBUTED_RUNTIME_DIR,
    best_params_path: str | Path = "data/optimisation_distributed/best_optuna_params.json",
    output_dir: str | Path = "data/evaluation_distributed",
) -> dict[str, Path]:
    runtime_root = (config.repo_path / runtime_dir).resolve()
    output_root = (config.repo_path / output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_runtime_manifest(runtime_root)
    best_params = load_best_params(best_params_path)

    train_path = str(manifest["evaluation"]["train_path"])
    valid_path = str(manifest["evaluation"]["valid_path"])
    test_path = str(manifest["evaluation"]["test_path"])
    history_reference_path = str(manifest["evaluation"]["history_reference_path"])
    scored_reference_test_path = str(manifest["evaluation"]["scored_reference_test_path"])

    train_frame = read_parquet_projected(train_path)
    valid_frame = read_parquet_projected(valid_path)
    test_frame = read_parquet_projected(test_path)
    history_reference = read_parquet_projected(history_reference_path)
    scored_reference_test = read_parquet_projected(scored_reference_test_path)
    requested_target_col = str(manifest["evaluation"]["requested_target_col"])

    combined_pretest_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    target_contract = resolve_target_contract(combined_pretest_frame, test_frame, requested_target_col=requested_target_col)
    train_frame = _prepare_training_target_frame(train_frame, target_contract=target_contract)
    valid_frame = _prepare_training_target_frame(valid_frame, target_contract=target_contract)
    test_frame, scored_reference_test, dropped_test_rows = _prepare_scored_test_frame(
        test_frame,
        scored_reference_test,
        target_contract=target_contract,
    )
    feature_cols, constant_feature_cols, identifier_feature_cols = _select_feature_columns(
        pd.concat([train_frame, valid_frame], ignore_index=True),
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
    )
    folds = build_daily_walk_forward_folds(test_frame, date_col=REFERENCE_DATE_COL)
    batch_size = max(1, int(config.pipelines.evaluation.batch_dates_per_task))
    fold_batches = [folds[index:index + batch_size] for index in range(0, len(folds), batch_size)]

    client = connect_client(config)
    wait_for_workers(client, minimum_workers=build_runtime_budget(config).cluster_task_slots)
    prediction_parts: list[pd.DataFrame] = []
    daily_report_rows: list[dict[str, object]] = []
    best_iterations: list[int] = []
    worker_execution: list[dict[str, object]] = []
    try:
        futures = [
            client.submit(
                _run_evaluation_fold_batch,
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
                feature_cols=feature_cols,
                target_contract_metadata=build_target_contract_metadata(target_contract),
                model_params=best_params,
                fold_batch=batch,
                pure=False,
            )
            for batch in fold_batches
        ]
        for future in as_completed(futures):
            result = future.result()
            if result["predictions"]:
                prediction_parts.append(pd.DataFrame(result["predictions"]))
            daily_report_rows.extend(result["reports"])
            best_iterations.extend(int(value) for value in result["best_iterations"])
            worker_execution.append(result["execution_identity"])
    finally:
        client.close()

    predictions_df = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["product", "target_date"]
    ).reset_index(drop=True)
    daily_report_df = pd.DataFrame(daily_report_rows).sort_values("fold").reset_index(drop=True)
    best_iteration = int(round(float(pd.Series(best_iterations, dtype=float).mean()))) if best_iterations else 2000
    baselines_payload = build_statistical_baselines_payload(history_reference, scored_reference_test)
    predictions_df, baseline_savings_payload = enrich_predictions_with_best_baseline(
        predictions_df,
        baselines_payload,
        unit_cost_eur=DEFAULT_UNIT_COST_EUR,
    )
    metrics_payload = compute_metrics_payload(history_reference, predictions_df)
    metrics_payload["model_family"] = "FOUNDATION_XGBOOST"
    if baseline_savings_payload is not None:
        metrics_payload["business_impact"] = baseline_savings_payload
    diagnostics_payload = _build_diagnostics_payload(
        predictions_df,
        evaluation_mode="distributed_bakery_reference_overlap",
        feature_count=len(feature_cols),
        best_iteration=best_iteration,
        overlap_metadata={
            **manifest["evaluation"]["overlap_metadata"],
            "dropped_test_rows_without_learning_target": int(dropped_test_rows),
        },
        daily_refit=True,
    )
    economic_gain_payload = build_simple_economic_gain_payload(predictions_df, metrics_payload)

    final_model = _fit_final_model(
        fit_frame=pd.concat([train_frame, valid_frame, test_frame], ignore_index=True),
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=best_params,
        num_boost_round=best_iteration,
    )
    metrics_output_path = output_root / "foundation_xgboost_test_metrics.json"
    baselines_output_path = output_root / "foundation_xgboost_statistical_baselines.json"
    predictions_output_path = output_root / "foundation_xgboost_test_predictions.csv"
    diagnostics_output_path = output_root / "foundation_xgboost_test_diagnostics.json"
    model_card_output_path = output_root / "foundation_xgboost_model_card.json"
    final_model_output_path = output_root / "foundation_xgboost_final_model.json"
    metadata_output_path = output_root / "foundation_xgboost_evaluation_metadata.json"
    economic_gain_output_path = output_root / "foundation_xgboost_economic_gain_vs_best_baseline.json"
    daily_report_output_path = output_root / "foundation_xgboost_daily_refit_metrics.csv"
    predictions_plot_path = output_root / "foundation_xgboost_actual_vs_predicted.png"
    residuals_plot_path = output_root / "foundation_xgboost_residuals.png"

    _json_dump(metrics_output_path, metrics_payload)
    _json_dump(baselines_output_path, baselines_payload)
    predictions_df.to_csv(predictions_output_path, index=False)
    _json_dump(diagnostics_output_path, diagnostics_payload)
    _json_dump(economic_gain_output_path, economic_gain_payload)
    daily_report_df.to_csv(daily_report_output_path, index=False)
    final_model.save_model(str(final_model_output_path))
    _json_dump(
        metadata_output_path,
        {
            "feature_cols": feature_cols,
            "constant_feature_cols": constant_feature_cols,
            "identifier_feature_cols": identifier_feature_cols,
            "best_iteration": best_iteration,
            "worker_execution": worker_execution,
            **build_target_contract_metadata(target_contract),
        },
    )
    _json_dump(
        model_card_output_path,
        {
            "model_family": "FOUNDATION_XGBOOST",
            "target_contract": build_target_contract_metadata(target_contract),
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
        },
    )
    _plot_actual_vs_predicted(predictions_df, predictions_plot_path)
    _plot_residuals(predictions_df, residuals_plot_path)
    logger.info(
        "Distributed bakery backtest complete: predictions=%s daily_refits=%s best_iteration=%s",
        len(predictions_df),
        len(daily_report_df),
        best_iteration,
    )
    return {
        "metrics_json": metrics_output_path,
        "predictions_csv": predictions_output_path,
        "diagnostics_json": diagnostics_output_path,
        "model_card_json": model_card_output_path,
        "final_model_json": final_model_output_path,
        "metadata_json": metadata_output_path,
        "economic_gain_json": economic_gain_output_path,
        "daily_refit_metrics_csv": daily_report_output_path,
        "actual_vs_predicted_plot": predictions_plot_path,
        "residuals_plot": residuals_plot_path,
    }
