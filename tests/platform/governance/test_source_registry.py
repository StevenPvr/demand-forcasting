from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.governance.source_registry import (  # noqa: E402
    allowed_training_dataset_sources,
    is_provider_runtime_enabled,
    load_source_registry_frame,
)


class SourceRegistryTests(unittest.TestCase):
    def test_allowed_training_dataset_sources_excludes_blocked_and_non_training_sources(self) -> None:
        allowed_sources = allowed_training_dataset_sources()

        self.assertEqual(
            allowed_sources,
            ["bakery", "first_party_daily", "freshretail", "freshretail_lt"],
        )
        self.assertNotIn("open_meteo_api", allowed_sources)

    def test_registry_seed_contains_expected_statuses(self) -> None:
        frame = load_source_registry_frame()
        self.assertFalse(frame["source_id"].duplicated().any())

        weather_row = frame.loc[frame["source_id"].eq("open_meteo_api")].iloc[0]
        self.assertEqual(weather_row["review_status"], "quarantine")
        self.assertTrue(bool(weather_row["contract_required"]))

        insee_row = frame.loc[frame["source_id"].eq("insee_bdm")].iloc[0]
        self.assertEqual(insee_row["review_status"], "allowed")
        self.assertTrue(bool(insee_row["commercial_use_allowed"]))
        self.assertTrue(bool(insee_row["ml_training_allowed"]))

    def test_provider_runtime_enabled_requires_explicit_contract_opt_in_for_quarantine_sources(self) -> None:
        self.assertFalse(is_provider_runtime_enabled("open_meteo_api", allow_contractual_providers=False))
        self.assertTrue(is_provider_runtime_enabled("open_meteo_api", allow_contractual_providers=True))
        self.assertTrue(is_provider_runtime_enabled("insee_bdm", allow_contractual_providers=False))


if __name__ == "__main__":
    unittest.main()
