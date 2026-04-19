from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from research_praedixa.distributed.config import DistributedConfig
from research_praedixa.distributed.sync import sync_project_tree
from research_praedixa.evaluation.pipeline import (
    DEFAULT_BAKERY_REFERENCE_TEST_CSV,
    DEFAULT_BAKERY_REFERENCE_TRAIN_CSV,
    DEFAULT_BAKERY_REFERENCE_VAL_CSV,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_GOLD_TABLE,
    DEFAULT_REQUESTED_TARGET_COL,
    _load_gold_reference_mode_frames,
)
from research_praedixa.global_dataset.pipeline import build_global_daily_standardization
from research_praedixa.memory_utils import downcast_pandas_frame, read_parquet_projected
from research_praedixa.optimisation.pipeline import (
    DEFAULT_TRAIN_SAMPLE_FRACTION,
    DEFAULT_TUNING_SAMPLE_FRACTION,
    load_gold_train_tuning_frames,
)


DEFAULT_DISTRIBUTED_RUNTIME_DIR = Path("data/distributed_runtime")


@dataclass(frozen=True)
class PreparedDistributedArtifacts:
    runtime_dir: Path
    optimisation_train: Path
    optimisation_tuning: Path
    evaluation_train: Path
    evaluation_valid: Path
    evaluation_test: Path
    evaluation_history_reference: Path
    evaluation_scored_reference: Path
    feature_selection_input: Path | None
    manifest_path: Path


def _json_dump(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def prepare_distributed_runtime(
    config: DistributedConfig,
    *,
    runtime_dir: str | Path = DEFAULT_DISTRIBUTED_RUNTIME_DIR,
    run_local_silver_pipeline: bool = False,
    run_local_gold_pipeline: bool = False,
    rebuild_global_dataset: bool = False,
    feature_selection_input_path: str | Path | None = None,
    sync_to_air: bool = False,
) -> PreparedDistributedArtifacts:
    target_dir = (config.repo_path / runtime_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if run_local_silver_pipeline:
        from scripts.run_local_silver import LocalSilverRunConfig, run_local_silver

        run_local_silver(LocalSilverRunConfig(
            data_dir=config.repo_path / "data",
            dbt_select="+tag:silver",
            run_bronze_load=True,
            run_dbt_tests=True,
        ))
    if run_local_gold_pipeline:
        from scripts.run_local_gold import LocalGoldRunConfig, run_local_gold

        run_local_gold(LocalGoldRunConfig(
            dbt_select="tag:gold",
            refresh_open_exogenous=True,
            run_dbt_tests=True,
        ))
    if rebuild_global_dataset:
        build_global_daily_standardization(output_dir=config.repo_path / "data/global_dataset")

    optimisation_dir = target_dir / "optimisation"
    optimisation_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = target_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    train_frame, tuning_frame, sample_store_col, train_sampling_metadata, tuning_sampling_metadata = load_gold_train_tuning_frames(
        duckdb_path=DEFAULT_DUCKDB_PATH,
        gold_table=DEFAULT_GOLD_TABLE,
        train_sample_fraction=config.pipelines.optimisation.train_sample_fraction,
        tuning_sample_fraction=config.pipelines.optimisation.tuning_sample_fraction,
    )
    optimisation_train = optimisation_dir / "train.parquet"
    optimisation_tuning = optimisation_dir / "tuning.parquet"
    downcast_pandas_frame(train_frame).to_parquet(optimisation_train, index=False)
    downcast_pandas_frame(tuning_frame).to_parquet(optimisation_tuning, index=False)

    (
        evaluation_train_frame,
        evaluation_valid_frame,
        evaluation_test_frame,
        history_reference_frame,
        scored_reference_test_frame,
        overlap_metadata,
    ) = _load_gold_reference_mode_frames(
        duckdb_path=DEFAULT_DUCKDB_PATH,
        gold_table=DEFAULT_GOLD_TABLE,
        train_sample_fraction=config.pipelines.optimisation.train_sample_fraction,
        tuning_sample_fraction=config.pipelines.optimisation.tuning_sample_fraction,
        bakery_reference_train_csv=DEFAULT_BAKERY_REFERENCE_TRAIN_CSV,
        bakery_reference_val_csv=DEFAULT_BAKERY_REFERENCE_VAL_CSV,
        bakery_reference_test_csv=DEFAULT_BAKERY_REFERENCE_TEST_CSV,
        train_frame=train_frame,
        valid_frame=tuning_frame,
    )
    evaluation_train = evaluation_dir / "train.parquet"
    evaluation_valid = evaluation_dir / "valid.parquet"
    evaluation_test = evaluation_dir / "test.parquet"
    evaluation_history_reference = evaluation_dir / "history_reference.parquet"
    evaluation_scored_reference = evaluation_dir / "scored_reference_test.parquet"
    downcast_pandas_frame(evaluation_train_frame).to_parquet(evaluation_train, index=False)
    downcast_pandas_frame(evaluation_valid_frame).to_parquet(evaluation_valid, index=False)
    downcast_pandas_frame(evaluation_test_frame).to_parquet(evaluation_test, index=False)
    downcast_pandas_frame(history_reference_frame).to_parquet(evaluation_history_reference, index=False)
    downcast_pandas_frame(scored_reference_test_frame).to_parquet(evaluation_scored_reference, index=False)

    feature_selection_input_dst: Path | None = None
    if feature_selection_input_path is not None:
        feature_selection_dir = target_dir / "feature_selection"
        feature_selection_dir.mkdir(parents=True, exist_ok=True)
        feature_selection_input_src = (config.repo_path / feature_selection_input_path).resolve()
        feature_selection_input_dst = feature_selection_dir / feature_selection_input_src.name
        if feature_selection_input_src != feature_selection_input_dst:
            shutil.copy2(feature_selection_input_src, feature_selection_input_dst)

    manifest_path = target_dir / "manifest.json"
    _json_dump(
        manifest_path,
        {
            "optimisation": {
                "train_path": str(optimisation_train),
                "tuning_path": str(optimisation_tuning),
                "sample_store_col": sample_store_col,
                "train_sampling_metadata": train_sampling_metadata,
                "tuning_sampling_metadata": tuning_sampling_metadata,
            },
            "evaluation": {
                "train_path": str(evaluation_train),
                "valid_path": str(evaluation_valid),
                "test_path": str(evaluation_test),
                "history_reference_path": str(evaluation_history_reference),
                "scored_reference_test_path": str(evaluation_scored_reference),
                "overlap_metadata": overlap_metadata,
                "requested_target_col": DEFAULT_REQUESTED_TARGET_COL,
            },
            "feature_selection": {
                "input_path": str(feature_selection_input_dst) if feature_selection_input_dst is not None else None,
            },
        },
    )

    if sync_to_air:
        sync_project_tree(config)

    return PreparedDistributedArtifacts(
        runtime_dir=target_dir,
        optimisation_train=optimisation_train,
        optimisation_tuning=optimisation_tuning,
        evaluation_train=evaluation_train,
        evaluation_valid=evaluation_valid,
        evaluation_test=evaluation_test,
        evaluation_history_reference=evaluation_history_reference,
        evaluation_scored_reference=evaluation_scored_reference,
        feature_selection_input=feature_selection_input_dst,
        manifest_path=manifest_path,
    )


def load_runtime_manifest(
    runtime_dir: str | Path = DEFAULT_DISTRIBUTED_RUNTIME_DIR,
) -> dict[str, object]:
    runtime_root = Path(runtime_dir)
    manifest_path = runtime_root / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))
