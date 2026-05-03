"""Machine-readable specification for the restaurant POS / ops synthetic data simulator."""

from __future__ import annotations

from praedixa.platform.datasets.synthetic_foodservice.spec.loader import (
    load_activity_segmentation,
    load_concrete_configurations,
    load_data_dictionary,
    load_export_layers,
    load_extreme_shocks,
    load_location_baseline_multipliers,
    load_normal_scenarios,
    load_taxonomy_profiles,
)
from praedixa.platform.datasets.synthetic_foodservice.spec.pipeline import (
    PIPELINE_STEPS,
    ROUNDING_TOLERANCE_EUR_DEFAULT,
    PipelineStep,
)
from praedixa.platform.datasets.synthetic_foodservice.spec.schema_version import SCHEMA_VERSION
from praedixa.platform.datasets.synthetic_foodservice.spec.validator_rules import (
    VALIDATION_INVARIANTS,
    ValidationInvariant,
    invariant_ids,
)

__all__ = [
    "PIPELINE_STEPS",
    "ROUNDING_TOLERANCE_EUR_DEFAULT",
    "PipelineStep",
    "SCHEMA_VERSION",
    "VALIDATION_INVARIANTS",
    "ValidationInvariant",
    "invariant_ids",
    "load_activity_segmentation",
    "load_concrete_configurations",
    "load_data_dictionary",
    "load_export_layers",
    "load_extreme_shocks",
    "load_location_baseline_multipliers",
    "load_normal_scenarios",
    "load_taxonomy_profiles",
]
