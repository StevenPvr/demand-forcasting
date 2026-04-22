from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import datasets
from praedixa.platform.runtime.paths import SOURCES_DIR


logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = SOURCES_DIR / "commercial_datasets" / "raw"
DATASETS_API: Any = datasets


def _download_hugging_face_split(*, dataset_name: str, split_name: str, output_path: Path, cache_dir: Path) -> Path:
    dataset_split = DATASETS_API.load_dataset(dataset_name, split=split_name, cache_dir=str(cache_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_split.to_parquet(str(output_path))
    logger.info("Downloaded Hugging Face split %s/%s to %s", dataset_name, split_name, output_path)
    return output_path


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


def _download_plan(root: Path, hf_cache_dir: Path) -> list[tuple[str, Path, Callable[[], Path]]]:
    return _hugging_face_plan_entries(root, hf_cache_dir)


def download_commercial_datasets(raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict[str, str]:
    """Download the public FreshRetail-LT artifacts used by the canonical pipeline."""

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


def main() -> None:
    """Download the approved commercial datasets into the configured raw directory."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    outputs = download_commercial_datasets(raw_dir=DEFAULT_RAW_DIR)
    logger.info("Commercial dataset downloads complete: %s", json.dumps(outputs, ensure_ascii=True))


if __name__ == "__main__":
    main()
