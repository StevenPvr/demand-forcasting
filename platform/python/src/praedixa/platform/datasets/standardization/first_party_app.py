from __future__ import annotations

from dataclasses import dataclass
import logging

from praedixa.platform.datasets.standardization.first_party import build_first_party_onboarding_templates
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_OUTPUT_DIR = SOURCES_DIR / "commercial_datasets" / "raw" / "templates"


@dataclass(frozen=True)
class FirstPartyAppConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


def build_default_first_party_app_config() -> FirstPartyAppConfig:
    return FirstPartyAppConfig()


def main() -> None:
    """Generate the onboarding template bundle and log the written paths."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = build_default_first_party_app_config()
    artifacts = build_first_party_onboarding_templates(config.output_dir)
    logging.getLogger(__name__).info("First-party onboarding templates written to %s", artifacts)


if __name__ == "__main__":
    main()
