from __future__ import annotations

import argparse
import logging

from praedixa.platform.datasets.standardization.first_party import build_first_party_onboarding_templates
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_OUTPUT_DIR = SOURCES_DIR / "commercial_datasets" / "raw" / "templates"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for first-party onboarding template generation."""

    parser = argparse.ArgumentParser(description="Create first-party onboarding templates for Praedixa.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    """Generate the onboarding template bundle and log the written paths."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    artifacts = build_first_party_onboarding_templates(args.output_dir)
    logging.getLogger(__name__).info("First-party onboarding templates written to %s", artifacts)


if __name__ == "__main__":
    main()
