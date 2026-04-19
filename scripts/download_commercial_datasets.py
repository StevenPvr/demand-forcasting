from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

from datasets import load_dataset


logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = Path("data/commercial_datasets/raw")
MENDELEY_API_BASE = "https://data.mendeley.com/public-api/datasets"
DEFAULT_HTTP_TIMEOUT_SECONDS = 120


def _download_bytes(url: str, *, timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS) -> bytes:
    request = Request(url, headers={"User-Agent": "PraedixaCommercialDatasetBootstrap/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        return response.read()


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _download_hugging_face_split(*, dataset_name: str, split_name: str, output_path: Path, cache_dir: Path) -> Path:
    dataset_split = load_dataset(dataset_name, split=split_name, cache_dir=str(cache_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_split.to_parquet(str(output_path))
    logger.info("Downloaded Hugging Face split %s/%s to %s", dataset_name, split_name, output_path)
    return output_path


def _mendeley_download_url(dataset_id: str) -> str:
    payload = json.loads(_download_bytes(f"{MENDELEY_API_BASE}/{dataset_id}").decode("utf-8"))
    files = payload.get("files", [])
    if not files:
        raise ValueError(f"No downloadable files were exposed by Mendeley for dataset {dataset_id}.")
    return str(files[0]["content_details"]["download_url"])


def _download_to_path(url: str, output_path: Path) -> Path:
    path = _write_bytes(output_path, _download_bytes(url))
    logger.info("Downloaded %s to %s", url, path)
    return path


def _download_plan(root: Path, hf_cache_dir: Path) -> list[tuple[str, Path, callable]]:
    return [
        (
            "freshretail_lt_train",
            root / "freshretail_lt_train.parquet",
            lambda: _download_hugging_face_split(
                dataset_name="Dingdong-Inc/FreshRetailNet-LT",
                split_name="train",
                output_path=root / "freshretail_lt_train.parquet",
                cache_dir=hf_cache_dir,
            ),
        ),
        (
            "freshretail_lt_eval",
            root / "freshretail_lt_eval.parquet",
            lambda: _download_hugging_face_split(
                dataset_name="Dingdong-Inc/FreshRetailNet-LT",
                split_name="eval",
                output_path=root / "freshretail_lt_eval.parquet",
                cache_dir=hf_cache_dir,
            ),
        ),
        (
            "uci_online_retail",
            root / "uci_online_retail.xlsx",
            lambda: _download_to_path(
                "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
                root / "uci_online_retail.xlsx",
            ),
        ),
        (
            "uci_online_retail_ii",
            root / "uci_online_retail_ii.xlsx",
            lambda: _download_to_path(
                "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
                root / "uci_online_retail_ii.xlsx",
            ),
        ),
        (
            "mendeley_ecommerce",
            root / "mendeley_ecommerce.xlsx",
            lambda: _download_to_path(
                _mendeley_download_url("ggbkd8ck3x"),
                root / "mendeley_ecommerce.xlsx",
            ),
        ),
        (
            "mendeley_pharmacy",
            root / "mendeley_pharmacy.zip",
            lambda: _download_to_path(
                _mendeley_download_url("2ym7v78wtd"),
                root / "mendeley_pharmacy.zip",
            ),
        ),
        (
            "mendeley_bangladesh",
            root / "mendeley_bangladesh.xlsx",
            lambda: _download_to_path(
                _mendeley_download_url("xwmbk7n3c8"),
                root / "mendeley_bangladesh.xlsx",
            ),
        ),
    ]


def download_commercial_datasets(raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict[str, str]:
    """Download commercially usable public datasets used by the canonical pipeline."""

    root = Path(raw_dir)
    hf_cache_dir = root / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))

    outputs: dict[str, str] = {}
    for dataset_name, target_path, download_fn in _download_plan(root, hf_cache_dir):
        if target_path.exists():
            logger.info("Skipping dataset %s because %s already exists", dataset_name, target_path)
            outputs[dataset_name] = str(target_path)
            continue
        logger.info("Downloading dataset %s", dataset_name)
        outputs[dataset_name] = str(download_fn())
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the commercial dataset downloader."""

    parser = argparse.ArgumentParser(description="Download commercially usable public datasets for Praedixa.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    return parser.parse_args()


def main() -> None:
    """Download the approved commercial datasets into the configured raw directory."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    outputs = download_commercial_datasets(raw_dir=args.raw_dir)
    logger.info("Commercial dataset downloads complete: %s", json.dumps(outputs, ensure_ascii=True))


if __name__ == "__main__":
    main()
