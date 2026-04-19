from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy"
DEFAULT_FILES: tuple[str, ...] = (
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sales_train_validation.csv",
    "sample_submission.csv",
    "sell_prices.csv",
)
DEFAULT_OUTPUT_DIR = "data/m5"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def build_download_plan(
    base_url: str = DEFAULT_BASE_URL,
    filenames: tuple[str, ...] = DEFAULT_FILES,
) -> dict[str, str]:
    """Return the source URL for each canonical M5 file."""
    return {filename: f"{base_url}/{filename}" for filename in filenames}


def download_file(url: str, destination: Path, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> int:
    """Download one file to disk and return its size in bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response:
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
    except HTTPError as exc:
        raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc.reason}") from exc

    return destination.stat().st_size


def sniff_csv_shape(csv_path: Path) -> dict[str, int]:
    """Read a CSV header and count data rows for a compact manifest summary."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {"rows": 0, "columns": 0}
        row_count = sum(1 for _ in reader)
    return {"rows": row_count, "columns": len(header)}


def export_m5_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    base_url: str = DEFAULT_BASE_URL,
    filenames: tuple[str, ...] = DEFAULT_FILES,
) -> dict[str, Any]:
    """Download the canonical M5 CSV files and persist a manifest."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    download_plan = build_download_plan(base_url=base_url, filenames=filenames)
    files_metadata: dict[str, dict[str, Any]] = {}

    for filename, source_url in download_plan.items():
        destination = output_path / filename
        LOGGER.info("Downloading M5 file %s", filename)
        byte_size = download_file(source_url, destination)
        csv_shape = sniff_csv_shape(destination)
        files_metadata[filename] = {
            "source_url": source_url,
            "path": str(destination),
            "bytes": byte_size,
            "rows": csv_shape["rows"],
            "columns": csv_shape["columns"],
        }

    metadata = {
        "dataset_name": "m5-forecasting-accuracy",
        "source": "Hugging Face dataset mirror",
        "base_url": base_url,
        "output_dir": str(output_path),
        "files": files_metadata,
    }
    metadata_path = output_path / "m5_download_metadata.json"
    metadata["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the M5 exporter."""
    parser = argparse.ArgumentParser(
        description="Download the canonical M5 Forecasting Accuracy CSV files."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for downloading M5 data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    metadata = export_m5_dataset(
        output_dir=args.output_dir,
        base_url=args.base_url,
    )
    LOGGER.info("M5 download complete: %s", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
