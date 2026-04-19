from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.global_dataset.commercial_external import commercial_dataset_compatibility_matrix  # noqa: E402


class CommercialExternalSmokeTests(unittest.TestCase):
    def test_compatibility_matrix_excludes_ambiguous_hierarchical_sales_dataset(self) -> None:
        compatibility_by_source = {
            item.dataset_source: item for item in commercial_dataset_compatibility_matrix()
        }

        self.assertTrue(compatibility_by_source["freshretail_lt"].compatible_with_pipeline)
        self.assertTrue(compatibility_by_source["mendeley_pharmacy_id"].commercial_use_allowed)
        self.assertFalse(compatibility_by_source["uci_hierarchical_sales"].commercial_use_allowed)


if __name__ == "__main__":
    unittest.main()
