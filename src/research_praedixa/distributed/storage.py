from __future__ import annotations

import optuna

from research_praedixa.distributed.config import DistributedConfig


def build_optuna_storage(config: DistributedConfig) -> optuna.storages.RDBStorage:
    return optuna.storages.RDBStorage(
        url=config.postgres.sqlalchemy_url,
        heartbeat_interval=60,
        grace_period=300,
        engine_kwargs={"pool_pre_ping": True},
    )


def ensure_study(
    *,
    study_name: str,
    config: DistributedConfig,
    direction: str = "maximize",
) -> optuna.study.Study:
    storage = build_optuna_storage(config)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )
