from __future__ import annotations

from dataclasses import dataclass
import logging

from praedixa.platform.warehouse.local_gold import (
    LocalGoldRunConfig,
    build_default_local_gold_run_config,
    run_local_gold,
)
from praedixa.platform.warehouse.local_silver import (
    LocalSilverRunConfig,
    build_default_local_silver_run_config,
    run_local_silver,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalMedallionRunConfig:
    """Runtime configuration for the local Praedixa medallion workflow."""

    run_silver: bool
    run_gold: bool
    silver: LocalSilverRunConfig
    gold: LocalGoldRunConfig


def build_default_local_medallion_run_config() -> LocalMedallionRunConfig:
    """Return the default local medallion workflow configuration."""

    return LocalMedallionRunConfig(
        run_silver=True,
        run_gold=True,
        silver=build_default_local_silver_run_config(),
        gold=build_default_local_gold_run_config(),
    )


def run_local_medallion(config: LocalMedallionRunConfig) -> dict[str, object]:
    """Run the local Praedixa medallion workflow from silver through gold."""

    if not config.run_silver and not config.run_gold:
        raise ValueError("At least one medallion step must be enabled.")

    results: dict[str, object] = {}
    if config.run_silver:
        logger.info("Starting medallion silver step.")
        results["silver"] = run_local_silver(config.silver)
        logger.info("Finished medallion silver step.")
    if config.run_gold:
        logger.info("Starting medallion gold step.")
        results["gold"] = run_local_gold(config.gold)
        logger.info("Finished medallion gold step.")
    return results


def main() -> None:
    """IDE-friendly entrypoint for the local Praedixa medallion workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_local_medallion(build_default_local_medallion_run_config())


if __name__ == "__main__":
    main()
