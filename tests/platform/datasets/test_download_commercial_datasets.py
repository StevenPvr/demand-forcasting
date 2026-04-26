from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.datasets import download_commercial_datasets as downloader  # noqa: E402


class DownloadCommercialDatasetsPlanTests(unittest.TestCase):
    def test_download_plan_adds_missing_open_training_datasets_without_pizza_mirror(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = downloader._download_plan(root, root / ".hf_cache")

        entries_by_name = {entry.name: entry for entry in entries}

        self.assertEqual(
            set(entries_by_name),
            {
                "freshretail_lt_train",
                "freshretail_lt_eval",
            },
        )
        self.assertNotIn("m5_forecasting_accuracy_zenodo", entries_by_name)
        self.assertNotIn("maven_pizza_place_sales", entries_by_name)
        self.assertNotIn("maven_coffee_shop_sales", entries_by_name)
        self.assertNotIn("maven_northwind_traders", entries_by_name)
        self.assertNotIn("pizza_place_sales_kaggle", entries_by_name)

    def test_download_public_function_writes_manifest_when_artifacts_exist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = downloader._download_plan(root, root / ".hf_cache")
            for entry in entries:
                entry.target_path.parent.mkdir(parents=True, exist_ok=True)
                entry.target_path.write_bytes(b"already-downloaded")

            outputs = downloader.download_commercial_datasets(raw_dir=root)
            manifest_path = root / "commercial_dataset_download_manifest.json"
            manifest_exists = manifest_path.exists()
            manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(set(outputs), {entry.name for entry in entries})
        self.assertTrue(manifest_exists)
        self.assertNotIn(
            "m5_forecasting_accuracy_zenodo",
            {str(row["name"]) for row in manifest_rows},
        )


if __name__ == "__main__":
    unittest.main()
