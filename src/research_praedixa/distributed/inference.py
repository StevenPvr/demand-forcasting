from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from distributed import as_completed
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from research_praedixa.distributed.budgets import build_runtime_budget
from research_praedixa.distributed.cluster import connect_client, wait_for_workers
from research_praedixa.distributed.config import DistributedConfig
from research_praedixa.distributed.logging import execution_identity
from research_praedixa.distributed.sync import sync_project_tree
from research_praedixa.memory_utils import read_parquet_projected
from research_praedixa.target_utils import TargetContract, reconstruct_absolute_predictions
from research_praedixa.xgboost_utils import predict_with_xgboost_booster


logger = logging.getLogger(__name__)


def _json_dump(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_target_contract_from_model_card(model_card_path: str | Path) -> tuple[list[str], TargetContract]:
    payload = json.loads(Path(model_card_path).read_text(encoding="utf-8"))
    target_contract_payload = payload["target_contract"]
    return payload["feature_cols"], TargetContract(
        learning_target_col=str(target_contract_payload["learning_target_col"]),
        absolute_target_col=str(target_contract_payload["absolute_target_col"]),
        target_mode=str(target_contract_payload["target_transform"]),
        reconstruction_anchor_col=target_contract_payload["reconstruction_anchor_col"],
    )


def _materialize_inference_chunks(
    *,
    input_path: Path,
    chunks_dir: Path,
    batch_rows: int,
) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(str(input_path))
    chunk_paths: list[Path] = []
    for batch_index, batch in enumerate(parquet_file.iter_batches(batch_size=batch_rows), start=1):
        chunk_path = chunks_dir / f"chunk_{batch_index:05d}.parquet"
        pq.write_table(pa.Table.from_batches([batch]), chunk_path)
        chunk_paths.append(chunk_path)
    if not chunk_paths:
        raise ValueError(f"No inference chunks were materialized from `{input_path}`.")
    return chunk_paths


def _score_inference_chunk(
    *,
    chunk_path: str,
    model_path: str,
    feature_cols: list[str],
    target_contract_payload: dict[str, str | None],
) -> dict[str, object]:
    target_contract = TargetContract(
        learning_target_col=str(target_contract_payload["learning_target_col"]),
        absolute_target_col=str(target_contract_payload["absolute_target_col"]),
        target_mode=str(target_contract_payload["target_transform"]),
        reconstruction_anchor_col=target_contract_payload["reconstruction_anchor_col"],
    )
    frame = read_parquet_projected(chunk_path)
    model = xgb.Booster()
    model.load_model(model_path)
    predictions = predict_with_xgboost_booster(model, frame, feature_cols)
    absolute_predictions = reconstruct_absolute_predictions(predictions, frame, target_contract)
    scored = frame.copy()
    scored["prediction_raw"] = absolute_predictions
    return {
        "rows": scored.to_dict(orient="records"),
        "execution_identity": execution_identity(),
    }


def run_distributed_batch_inference(
    *,
    config: DistributedConfig,
    input_path: str | Path,
    model_path: str | Path = "data/evaluation_distributed/foundation_xgboost_final_model.json",
    model_card_path: str | Path = "data/evaluation_distributed/foundation_xgboost_model_card.json",
    output_dir: str | Path = "data/predictions_distributed",
) -> dict[str, Path]:
    input_file = Path(input_path).resolve()
    output_root = (config.repo_path / output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    feature_cols, target_contract = _load_target_contract_from_model_card(model_card_path)
    batch_rows = int(config.pipelines.inference.batch_rows)
    chunk_paths = _materialize_inference_chunks(
        input_path=input_file,
        chunks_dir=output_root / "_chunks",
        batch_rows=batch_rows,
    )
    sync_project_tree(config)
    client = connect_client(config)
    wait_for_workers(client, minimum_workers=build_runtime_budget(config).cluster_task_slots)
    scored_parts: list[pd.DataFrame] = []
    worker_execution: list[dict[str, object]] = []
    try:
        futures = [
            client.submit(
                _score_inference_chunk,
                chunk_path=str(chunk_path),
                model_path=str(model_path),
                feature_cols=feature_cols,
                target_contract_payload={
                    "learning_target_col": target_contract.learning_target_col,
                    "absolute_target_col": target_contract.absolute_target_col,
                    "target_transform": target_contract.target_mode,
                    "reconstruction_anchor_col": target_contract.reconstruction_anchor_col,
                },
                pure=False,
            )
            for chunk_path in chunk_paths
        ]
        for future in as_completed(futures):
            result = future.result()
            scored_parts.append(pd.DataFrame(result["rows"]))
            worker_execution.append(result["execution_identity"])
    finally:
        client.close()
    predictions_df = pd.concat(scored_parts, ignore_index=True)
    predictions_output_path = output_root / "batch_predictions.parquet"
    metadata_output_path = output_root / "batch_predictions_metadata.json"
    predictions_df.to_parquet(predictions_output_path, index=False)
    _json_dump(
        metadata_output_path,
        {
            "input_path": str(input_file),
            "model_path": str(model_path),
            "row_count": int(len(predictions_df)),
            "worker_execution": worker_execution,
        },
    )
    logger.info("Distributed batch inference complete: rows=%s output=%s", len(predictions_df), predictions_output_path)
    return {
        "predictions_parquet": predictions_output_path,
        "metadata_json": metadata_output_path,
    }
