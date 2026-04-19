from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import yaml


DEFAULT_DISTRIBUTED_CONFIG_PATH = Path("config/distributed.yaml")


@dataclass(frozen=True)
class ProjectConfig:
    repo_path: Path
    python_bin: str
    dask_bin: str

    def resolve_python_bin(self) -> Path:
        return self.repo_path / self.python_bin

    def resolve_dask_bin(self) -> Path:
        return self.repo_path / self.dask_bin


@dataclass(frozen=True)
class SchedulerConfig:
    host: str
    bind_host: str
    port: int
    dashboard_port: int

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    bind_host: str
    port: int
    database: str
    user: str
    password: str
    data_dir: Path
    log_path: Path
    air_allow_cidr: str

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def psycopg_url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class WorkerPoolConfig:
    worker_count: int
    dask_threads_per_worker: int
    xgboost_threads_per_task: int
    memory_limit: str
    local_directory: Path


@dataclass(frozen=True)
class RemoteWorkerPoolConfig(WorkerPoolConfig):
    ssh_host: str
    fallback_host: str


@dataclass(frozen=True)
class RuntimeConfig:
    logs_dir: Path
    pid_dir: Path
    sync_mode: str
    sync_excludes: tuple[str, ...]
    sync_includes: tuple[str, ...]


@dataclass(frozen=True)
class OptimisationDistributedConfig:
    train_sample_fraction: float
    tuning_sample_fraction: float
    n_folds: int
    tuning_trials: int
    optuna_parallel_tasks: int
    max_parallel_fold_workers: int


@dataclass(frozen=True)
class FeatureSelectionDistributedConfig:
    candidate_batch_size: int
    n_folds: int
    tuning_trials: int


@dataclass(frozen=True)
class EvaluationDistributedConfig:
    batch_dates_per_task: int


@dataclass(frozen=True)
class InferenceDistributedConfig:
    batch_rows: int


@dataclass(frozen=True)
class PipelinesConfig:
    optimisation: OptimisationDistributedConfig
    feature_selection: FeatureSelectionDistributedConfig
    evaluation: EvaluationDistributedConfig
    inference: InferenceDistributedConfig


@dataclass(frozen=True)
class DistributedConfig:
    project: ProjectConfig
    scheduler: SchedulerConfig
    postgres: PostgresConfig
    pro_workers: WorkerPoolConfig
    air_workers: RemoteWorkerPoolConfig
    runtime: RuntimeConfig
    pipelines: PipelinesConfig
    config_path: Path

    @property
    def repo_path(self) -> Path:
        return self.project.repo_path

    @property
    def scheduler_address(self) -> str:
        return self.scheduler.address

    def resolve_logs_dir(self) -> Path:
        return self.repo_path / self.runtime.logs_dir

    def resolve_pid_dir(self) -> Path:
        return self.repo_path / self.runtime.pid_dir


def _resolve_path(project_root: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    return path if path.is_absolute() else (project_root / path)


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Distributed config key `{key}` must be a mapping.")
    return value


def load_distributed_config(config_path: str | Path | None = None) -> DistributedConfig:
    path = Path(config_path or os.environ.get("PRAEDIXA_DISTRIBUTED_CONFIG", DEFAULT_DISTRIBUTED_CONFIG_PATH))
    if not path.is_absolute():
        path = Path.cwd() / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Distributed config at `{path}` must be a mapping.")

    project_raw = _require_mapping(raw, "project")
    project_root = _resolve_path(path.parent.parent if path.parent.name == "config" else Path.cwd(), project_raw["repo_path"])
    scheduler_raw = _require_mapping(raw, "scheduler")
    postgres_raw = _require_mapping(raw, "postgres")
    workers_raw = _require_mapping(raw, "workers")
    runtime_raw = _require_mapping(raw, "runtime")
    pipelines_raw = _require_mapping(raw, "pipelines")

    project = ProjectConfig(
        repo_path=project_root,
        python_bin=str(project_raw["python_bin"]),
        dask_bin=str(project_raw["dask_bin"]),
    )
    scheduler = SchedulerConfig(
        host=str(scheduler_raw["host"]),
        bind_host=str(scheduler_raw["bind_host"]),
        port=int(scheduler_raw["port"]),
        dashboard_port=int(scheduler_raw["dashboard_port"]),
    )
    postgres = PostgresConfig(
        host=str(postgres_raw["host"]),
        bind_host=str(postgres_raw["bind_host"]),
        port=int(postgres_raw["port"]),
        database=str(postgres_raw["database"]),
        user=str(postgres_raw["user"]),
        password=str(os.environ.get("PRAEDIXA_OPTUNA_DB_PASSWORD", postgres_raw["password"])),
        data_dir=_resolve_path(project_root, str(postgres_raw["data_dir"])),
        log_path=_resolve_path(project_root, str(postgres_raw["log_path"])),
        air_allow_cidr=str(postgres_raw["air_allow_cidr"]),
    )
    pro_raw = _require_mapping(workers_raw, "pro")
    air_raw = _require_mapping(workers_raw, "air")
    pro_workers = WorkerPoolConfig(
        worker_count=int(pro_raw["worker_count"]),
        dask_threads_per_worker=int(pro_raw["dask_threads_per_worker"]),
        xgboost_threads_per_task=int(pro_raw["xgboost_threads_per_task"]),
        memory_limit=str(pro_raw["memory_limit"]),
        local_directory=_resolve_path(project_root, str(pro_raw["local_directory"])),
    )
    air_workers = RemoteWorkerPoolConfig(
        ssh_host=str(air_raw["ssh_host"]),
        fallback_host=str(air_raw["fallback_host"]),
        worker_count=int(air_raw["worker_count"]),
        dask_threads_per_worker=int(air_raw["dask_threads_per_worker"]),
        xgboost_threads_per_task=int(air_raw["xgboost_threads_per_task"]),
        memory_limit=str(air_raw["memory_limit"]),
        local_directory=_resolve_path(project_root, str(air_raw["local_directory"])),
    )
    runtime = RuntimeConfig(
        logs_dir=_resolve_path(project_root, str(runtime_raw["logs_dir"])),
        pid_dir=_resolve_path(project_root, str(runtime_raw["pid_dir"])),
        sync_mode=str(runtime_raw["sync_mode"]),
        sync_excludes=tuple(str(item) for item in runtime_raw.get("sync_excludes", [])),
        sync_includes=tuple(str(item) for item in runtime_raw.get("sync_includes", [])),
    )
    pipelines = PipelinesConfig(
        optimisation=OptimisationDistributedConfig(**_require_mapping(pipelines_raw, "optimisation")),
        feature_selection=FeatureSelectionDistributedConfig(**_require_mapping(pipelines_raw, "feature_selection")),
        evaluation=EvaluationDistributedConfig(**_require_mapping(pipelines_raw, "evaluation")),
        inference=InferenceDistributedConfig(**_require_mapping(pipelines_raw, "inference")),
    )
    return DistributedConfig(
        project=project,
        scheduler=scheduler,
        postgres=postgres,
        pro_workers=pro_workers,
        air_workers=air_workers,
        runtime=runtime,
        pipelines=pipelines,
        config_path=path,
    )
