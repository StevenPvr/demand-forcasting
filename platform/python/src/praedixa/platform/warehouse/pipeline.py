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


class MedallionPipelineRunner:
    """Execute the local medallion workflow without duplicating dbt stage wiring."""

    def __init__(
        self,
        dbt_runner_factory: Callable[[], DbtStageRunner] = DbtStageRunner,
    ) -> None:
        self._dbt_runner_factory = dbt_runner_factory

    def run(self, config: MedallionRunConfig) -> dict[str, object]:
        validate_medallion_plan(config)
        env = self._build_env(config)
        results: dict[str, object] = {}

        if config.load_core_bronze:
            results["load_core_bronze"] = self._load_core_bronze(config, env)
        if config.run_silver:
            results["run_silver"] = self._run_silver(config, env)
        if config.refresh_open_exogenous:
            results["refresh_open_exogenous"] = self._refresh_open_exogenous(env)
        if config.run_gold:
            results["run_gold"] = self._run_gold(config, env)

        return results

    def _build_env(self, config: MedallionRunConfig) -> dict[str, str]:
        env = build_local_gold_env()
        env["PRAEDIXA_BRONZE_DATA_DIR"] = str(config.silver.data_dir)
        return env

    def _load_core_bronze(
        self,
        config: MedallionRunConfig,
        env: dict[str, str],
    ) -> dict[str, object] | None:
        logger.info("Starting medallion step: load_core_bronze.")
        return load_core_bronze_sources(config.silver, env)

    def _run_silver(
        self,
        config: MedallionRunConfig,
        env: dict[str, str],
    ) -> dict[str, object]:
        logger.info("Starting medallion step: run_silver.")
        return self._run_dbt_stage(
            stage_config=DbtStageRunConfig(
                selector=config.silver.dbt_select,
                test_selector=resolve_dbt_test_selector(config.silver.dbt_select),
                test_exclude=resolve_dbt_test_exclude(),
                run_tests=config.silver.run_dbt_tests,
            ),
            env=env,
        )

    def _refresh_open_exogenous(self, env: dict[str, str]) -> dict[str, object]:
        logger.info("Starting medallion step: refresh_open_exogenous.")
        return refresh_open_exogenous_inputs(env)

    def _run_gold(
        self,
        config: MedallionRunConfig,
        env: dict[str, str],
    ) -> dict[str, object]:
        logger.info("Starting medallion step: run_gold.")
        return self._run_dbt_stage(
            stage_config=DbtStageRunConfig(
                selector=config.gold.dbt_select,
                test_selector=config.gold.dbt_select.strip().lstrip("+"),
                run_tests=config.gold.run_dbt_tests,
            ),
            env=env,
        )

    def _run_dbt_stage(
        self,
        *,
        stage_config: DbtStageRunConfig,
        env: dict[str, str],
    ) -> dict[str, object]:
        return (
            self._dbt_runner_factory()
            .run_stage(
                config=stage_config,
                env=env,
            )
            .as_dict()
        )


def run_local_medallion_pipeline(
    config: MedallionRunConfig,
    *,
    dbt_runner_factory: Callable[[], DbtStageRunner] = DbtStageRunner,
) -> dict[str, object]:
    """Run the local medallion pipeline with explicit bronze/silver/open/gold steps."""

    return MedallionPipelineRunner(dbt_runner_factory=dbt_runner_factory).run(config)
