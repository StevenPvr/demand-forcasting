"""Load packaged YAML spec assets for the synthetic foodservice simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_SPEC_DIR: Path = Path(__file__).resolve().parent
_SCENARIOS_DIR: Path = _SPEC_DIR.parent / "scenarios"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def load_data_dictionary() -> dict[str, Any]:
    """Return the full data dictionary (tables, enums, conventions)."""
    return _read_yaml(_SPEC_DIR / "data_dictionary.yaml")


def load_export_layers() -> dict[str, Any]:
    """Return export layer corruption rates by data_quality_level."""
    return _read_yaml(_SPEC_DIR / "export_layers.yaml")


def load_activity_segmentation() -> dict[str, Any]:
    return _read_yaml(_SCENARIOS_DIR / "activity_segmentation.yaml")


def load_location_baseline_multipliers() -> dict[str, Any]:
    return _read_yaml(_SCENARIOS_DIR / "location_baseline_multipliers.yaml")


def load_taxonomy_profiles() -> dict[str, Any]:
    return _read_yaml(_SCENARIOS_DIR / "taxonomy_profiles.yaml")


def load_normal_scenarios() -> dict[str, Any]:
    return _read_yaml(_SCENARIOS_DIR / "normal_scenarios.yaml")


def load_extreme_shocks() -> dict[str, Any]:
    return _read_yaml(_SCENARIOS_DIR / "extreme_shocks.yaml")


def load_concrete_configurations() -> dict[str, Any]:
    return _read_yaml(_SCENARIOS_DIR / "concrete_configurations.yaml")
