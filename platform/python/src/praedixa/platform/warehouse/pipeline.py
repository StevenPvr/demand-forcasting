from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging

from praedixa.platform.warehouse.local_gold import (
    LocalGoldRunConfig,
    build_local_gold_env,
    refresh_open_exogenous_inputs,
)
from praedixa.platform.warehouse.local_silver import (
    LocalSilverRunConfig,
    load_core_bronze_sources,
)
from praedixa.platform.warehouse.dbt_runner import (
    DbtStageRunConfig,
    DbtStageRunner,
    resolve_dbt_test_exclude,
    resolve_dbt_test_selector,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MedallionRunConfig:
    """Explicit local medallion execution plan."""

    load_core_bronze: bool
    run_silver: bool
    refresh_open_exogenous: bool
    run_gold: bool
    silver: LocalSilverRunConfig
    gold: LocalGoldRunConfig


@dataclass(frozen=True)
class MedallionStep:
    """One named step in the local medallion run."""

    name: str
    enabled: bool


def build_medallion_steps(config: MedallionRunConfig) -> list[MedallionStep]:
    """Return the concrete local medallion steps in execution order."""

    return [
        MedallionStep("load_core_bronze", config.load_core_bronze),
        MedallionStep("run_silver", config.run_silver),
        MedallionStep("refresh_open_exogenous", config.refresh_open_exogenous),
        MedallionStep("run_gold", config.run_gold),
    ]


def validate_medallion_plan(config: MedallionRunConfig) -> None:
    """Reject incoherent local medallion plans before side effects."""

    if not any(step.enabled for step in build_medallion_steps(config)):
        raise ValueError("At least one medallion step must be enabled.")
    if config.refresh_open_exogenous and not config.run_silver:
        logger.info(
            "Open exogenous refresh assumes silver already exists because run_silver is disabled."
        )


def run_local_medallion_pipeline(
    config: MedallionRunConfig,
    *,
    dbt_runner_factory: Callable[[], DbtStageRunner] = DbtStageRunner,
) -> dict[str, object]:
    """Run the local medallion pipeline with explicit bronze/silver/open/gold steps."""

    validate_medallion_plan(config)
    results: dict[str, object] = {}
    env = build_local_gold_env()
    env["PRAEDIXA_BRONZE_DATA_DIR"] = str(config.silver.data_dir)

    if config.load_core_bronze:
        logger.info("Starting medallion step: load_core_bronze.")
        results["load_core_bronze"] = load_core_bronze_sources(config.silver, env)

    if config.run_silver:
        logger.info("Starting medallion step: run_silver.")
        runner = dbt_runner_factory()
        results["run_silver"] = runner.run_stage(
            config=DbtStageRunConfig(
                selector=config.silver.dbt_select,
                test_selector=resolve_dbt_test_selector(config.silver.dbt_select),
                test_exclude=resolve_dbt_test_exclude(),
                run_tests=config.silver.run_dbt_tests,
            ),
            env=env,
        ).as_dict()

    if config.refresh_open_exogenous:
        logger.info("Starting medallion step: refresh_open_exogenous.")
        results["refresh_open_exogenous"] = refresh_open_exogenous_inputs(env)

    if config.run_gold:
        logger.info("Starting medallion step: run_gold.")
        runner = dbt_runner_factory()
        results["run_gold"] = runner.run_stage(
            config=DbtStageRunConfig(
                selector=config.gold.dbt_select,
                test_selector=config.gold.dbt_select.strip().lstrip("+"),
                run_tests=config.gold.run_dbt_tests,
            ),
            env=env,
        ).as_dict()

    return results
