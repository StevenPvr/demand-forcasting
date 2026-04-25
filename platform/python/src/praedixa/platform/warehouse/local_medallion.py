from __future__ import annotations

from dataclasses import dataclass
import logging

from praedixa.platform.warehouse.local_gold import (
    LocalGoldRunConfig,
    build_default_local_gold_run_config,
)
from praedixa.platform.warehouse.local_silver import (
    LocalSilverRunConfig,
    build_default_local_silver_run_config,
)
from praedixa.platform.warehouse.pipeline import (
    MedallionRunConfig,
    run_local_medallion_pipeline,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalMedallionRunConfig:
    """Runtime configuration for the local Praedixa medallion workflow."""

    run_silver: bool
    run_gold: bool
    silver: LocalSilverRunConfig
    gold: LocalGoldRunConfig

    def to_pipeline_config(self) -> MedallionRunConfig:
        """Convert the public local config to the explicit pipeline config."""

        return MedallionRunConfig(
            load_core_bronze=self.run_silver and self.silver.run_bronze_load,
            run_silver=self.run_silver,
            refresh_open_exogenous=self.run_gold
            and self.gold.refresh_open_exogenous,
            run_gold=self.run_gold,
            silver=self.silver,
            gold=self.gold,
        )


def build_default_local_medallion_run_config() -> LocalMedallionRunConfig:
    """Return the default local medallion workflow configuration."""

    return LocalMedallionRunConfig(
        run_silver=True,
        run_gold=True,
        silver=build_default_local_silver_run_config(),
        gold=build_default_local_gold_run_config(),
    )


def run_local_medallion(config: LocalMedallionRunConfig) -> dict[str, object]:
    """Run the explicit local Praedixa medallion workflow."""

    results = run_local_medallion_pipeline(config.to_pipeline_config())
    if "run_silver" in results:
        results["silver"] = results["run_silver"]
    if "run_gold" in results:
        results["gold"] = results["run_gold"]
    return results


def main() -> None:
    """IDE-friendly entrypoint for the local Praedixa medallion workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_local_medallion(build_default_local_medallion_run_config())


if __name__ == "__main__":
    main()
