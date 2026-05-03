from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.synthetic_foodservice.spec import (  # noqa: E402
    PIPELINE_STEPS,
    SCHEMA_VERSION,
    VALIDATION_INVARIANTS,
    invariant_ids,
    load_concrete_configurations,
    load_data_dictionary,
    load_export_layers,
    load_extreme_shocks,
    load_normal_scenarios,
)
from praedixa.platform.datasets.synthetic_foodservice.spec.schema_version import (  # noqa: E402
    SCHEMA_VERSION as SCHEMA_VERSION_DIRECT,
)


class SpecAssetsTests(unittest.TestCase):
    def test_schema_version_matches_dictionary(self) -> None:
        dd = load_data_dictionary()
        self.assertEqual(dd.get("schema_version"), SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION_DIRECT, SCHEMA_VERSION)

    def test_data_dictionary_has_core_tables(self) -> None:
        dd = load_data_dictionary()
        tables: dict[str, object] = dd["tables"]
        for name in (
            "sites",
            "products",
            "sales_tickets",
            "sales_lines",
            "latent_demand_internal",
            "simulation_parameters",
            "locations",
        ):
            self.assertIn(name, tables)

    def test_export_layers_quality_profiles(self) -> None:
        el = load_export_layers()
        profiles = el["quality_profiles"]
        for level in ("high", "medium", "medium_low", "low"):
            self.assertIn(level, profiles)

    def test_normal_scenarios_twelve(self) -> None:
        ns = load_normal_scenarios()
        scenarios = ns["scenarios"]
        self.assertGreaterEqual(len(scenarios), 12)

    def test_extreme_shocks_catalog(self) -> None:
        xs = load_extreme_shocks()
        shocks = xs["shocks"]
        ids = {s["id"] for s in shocks}
        self.assertIn("heatwave", ids)
        self.assertIn("pos_migration_cutover", ids)

    def test_concrete_configurations_three(self) -> None:
        cc = load_concrete_configurations()
        configs = cc["configurations"]
        self.assertEqual(len(configs), 3)

    def test_pipeline_steps_ordered(self) -> None:
        orders = [step.order for step in PIPELINE_STEPS]
        self.assertEqual(orders, sorted(orders))
        self.assertEqual(orders[0], 1)

    def test_validation_invariant_ids_unique(self) -> None:
        ids = invariant_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(VALIDATION_INVARIANTS), 10)


if __name__ == "__main__":
    unittest.main()
