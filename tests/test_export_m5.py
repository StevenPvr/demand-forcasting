from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.export_m5 import (  # noqa: E402
    DEFAULT_FILES,
    DEFAULT_BASE_URL,
    build_download_plan,
    export_m5_dataset,
    sniff_csv_shape,
)


class ExportM5Tests(unittest.TestCase):
    def test_build_download_plan_returns_one_url_per_file(self) -> None:
        plan = build_download_plan()

        self.assertEqual(set(plan.keys()), set(DEFAULT_FILES))
        self.assertTrue(all(url.startswith(DEFAULT_BASE_URL) for url in plan.values()))

    def test_sniff_csv_shape_counts_rows_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "a,b,c",
                        "1,2,3",
                        "4,5,6",
                    ]
                ),
                encoding="utf-8",
            )

            shape = sniff_csv_shape(csv_path)

        self.assertEqual(shape, {"rows": 2, "columns": 3})

    def test_export_m5_dataset_writes_metadata_for_downloaded_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "m5"

            def fake_download_file(url: str, destination: Path, chunk_size: int = 0) -> int:
                file_name = destination.name
                destination.write_text(
                    "\n".join(
                        [
                            "col_a,col_b",
                            f"{file_name},1",
                            f"{file_name},2",
                        ]
                    ),
                    encoding="utf-8",
                )
                return destination.stat().st_size

            with mock.patch(
                "research_praedixa.export_m5.download_file",
                side_effect=fake_download_file,
            ):
                metadata = export_m5_dataset(output_dir=output_dir)

            metadata_path = output_dir / "m5_download_metadata.json"
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["dataset_name"], "m5-forecasting-accuracy")
        self.assertEqual(set(metadata["files"].keys()), set(DEFAULT_FILES))
        self.assertEqual(saved_metadata["metadata_path"], str(metadata_path))
        self.assertTrue(all(file_meta["rows"] == 2 for file_meta in metadata["files"].values()))
        self.assertTrue(all(file_meta["columns"] == 2 for file_meta in metadata["files"].values()))


if __name__ == "__main__":
    unittest.main()
