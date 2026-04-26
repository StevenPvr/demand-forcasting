from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


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

        self.assertIn("m5_forecasting_accuracy_zenodo", entries_by_name)
        self.assertEqual(
            set(entries_by_name),
            {
                "freshretail_lt_train",
                "freshretail_lt_eval",
                "m5_forecasting_accuracy_zenodo",
                "restaurant_sales_forecasting_zenodo",
                "uci_online_retail_ii",
                "uci_online_retail",
                "maven_cafe_rewards_offers",
                "perishable_goods_management_kaggle",
                "restaurant_sales_report_kaggle",
            },
        )
        self.assertIn(
            "zenodo.org/records/12636070",
            entries_by_name["m5_forecasting_accuracy_zenodo"].source_url,
        )
        self.assertNotIn(
            "kaggle",
            entries_by_name["m5_forecasting_accuracy_zenodo"].source_url.lower(),
        )
        self.assertNotIn("maven_pizza_place_sales", entries_by_name)
        self.assertNotIn("maven_coffee_shop_sales", entries_by_name)
        self.assertNotIn("maven_northwind_traders", entries_by_name)
        self.assertNotIn("pizza_place_sales_kaggle", entries_by_name)

    def test_zenodo_download_validates_license_and_uses_md5(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "record.zip"
            record = {
                "metadata": {"license": {"id": "cc-by-4.0"}},
                "files": [
                    {
                        "key": "record.zip",
                        "checksum": "md5:abc123",
                        "links": {"self": "https://zenodo.example/record.zip/content"},
                    }
                ],
            }
            with (
                mock.patch.object(downloader, "_read_json_url", return_value=record),
                mock.patch.object(
                    downloader, "_download_url", return_value=target_path
                ) as download_url,
            ):
                result = downloader._download_zenodo_record_file(
                    record_id=1,
                    file_name="record.zip",
                    output_path=target_path,
                    expected_license_id="cc-by-4.0",
                )

        self.assertEqual(result, target_path)
        download_url.assert_called_once_with(
            url="https://zenodo.example/record.zip/content",
            output_path=target_path,
            expected_md5="abc123",
        )

    def test_kaggle_download_validates_license_and_pins_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "dataset.zip"
            metadata = {"licenseName": "Apache 2.0", "currentVersionNumber": 2}
            with (
                mock.patch.object(downloader, "_read_json_url", return_value=metadata),
                mock.patch.object(
                    downloader, "_download_url", return_value=target_path
                ) as download_url,
            ):
                result = downloader._download_kaggle_dataset(
                    dataset_ref="owner/dataset",
                    dataset_version=2,
                    output_path=target_path,
                    expected_license_name="Apache 2.0",
                )

        self.assertEqual(result, target_path)
        download_url.assert_called_once_with(
            url="https://www.kaggle.com/api/v1/datasets/download/owner/dataset?datasetVersionNumber=2",
            output_path=target_path,
            expected_content_type="application/zip",
        )

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
        self.assertIn(
            "m5_forecasting_accuracy_zenodo",
            {str(row["name"]) for row in manifest_rows},
        )


if __name__ == "__main__":
    unittest.main()
