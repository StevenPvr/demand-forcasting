from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

import datasets
from praedixa.platform.runtime.paths import SOURCES_DIR


logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = SOURCES_DIR / "commercial_datasets" / "raw"
MENDELEY_API_BASE = "https://data.mendeley.com/public-api/datasets"
DEFAULT_HTTP_TIMEOUT_SECONDS = 120
DATASETS_API: Any = datasets


def _download_bytes(url: str, *, timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS) -> bytes:
    request = Request(url, headers={"User-Agent": "PraedixaCommercialDatasetBootstrap/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        return bytes(response.read())


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _download_hugging_face_split(*, dataset_name: str, split_name: str, output_path: Path, cache_dir: Path) -> Path:
    dataset_split = DATASETS_API.load_dataset(dataset_name, split=split_name, cache_dir=str(cache_dir))
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


def _plan_entry(name: str, target_path: Path, download_fn: Callable[[], Path]) -> tuple[str, Path, Callable[[], Path]]:
    return (name, target_path, download_fn)


def _hugging_face_plan_entries(root: Path, hf_cache_dir: Path) -> list[tuple[str, Path, Callable[[], Path]]]:
    dataset_name = "Dingdong-Inc/FreshRetailNet-LT"
    split_specs = [
        ("freshretail_lt_train", "train"),
        ("freshretail_lt_eval", "eval"),
    ]
    entries: list[tuple[str, Path, Callable[[], Path]]] = []
    for artifact_name, split_name in split_specs:
        target_path = root / f"{artifact_name}.parquet"
        entries.append(
            _plan_entry(
                artifact_name,
                target_path,
                lambda split_name=split_name, target_path=target_path: _download_hugging_face_split(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    output_path=target_path,
                    cache_dir=hf_cache_dir,
                ),
            )
        )
    return entries


def _url_plan_entries(root: Path) -> list[tuple[str, Path, Callable[[], Path]]]:
    download_specs = [
        (
            "uci_online_retail",
            "uci_online_retail.xlsx",
            lambda: "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
        ),
        (
            "uci_online_retail_ii",
            "uci_online_retail_ii.xlsx",
            lambda: "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        ),
        ("mendeley_ecommerce", "mendeley_ecommerce.xlsx", lambda: _mendeley_download_url("ggbkd8ck3x")),
        ("mendeley_pharmacy", "mendeley_pharmacy.zip", lambda: _mendeley_download_url("2ym7v78wtd")),
        ("mendeley_bangladesh", "mendeley_bangladesh.xlsx", lambda: _mendeley_download_url("xwmbk7n3c8")),
    ]
    return [
        _plan_entry(
            artifact_name,
            root / file_name,
            lambda url_factory=url_factory, target_path=root / file_name: _download_to_path(url_factory(), target_path),
        )
        for artifact_name, file_name, url_factory in download_specs
    ]


def _download_plan(root: Path, hf_cache_dir: Path) -> list[tuple[str, Path, Callable[[], Path]]]:
    return _hugging_face_plan_entries(root, hf_cache_dir) + _url_plan_entries(root)


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
