from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

from praedixa.platform.runtime.paths import SOURCES_DIR


logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = SOURCES_DIR / "commercial_datasets" / "raw"
DEFAULT_DOWNLOAD_MANIFEST_PATH = (
    DEFAULT_RAW_DIR / "commercial_dataset_download_manifest.json"
)


@dataclass(frozen=True)
class DownloadPlanEntry:
    """One approved commercial dataset artifact download."""

    name: str
    target_path: Path
    source_url: str
    license_type: str
    download_fn: Callable[[], Path]


def _download_plan(root: Path, hf_cache_dir: Path) -> list[DownloadPlanEntry]:
    """Return approved automatic downloads.

    The active corpus no longer includes FreshRetail-LT or other automatic
    commercial downloads. M5 is expected as a governed local/Zenodo artifact.
    """

    _ = (root, hf_cache_dir)
    return []


def _manifest_row(entry: DownloadPlanEntry, output_path: Path) -> dict[str, object]:
    stat = output_path.stat() if output_path.exists() else None
    return {
        "name": entry.name,
        "path": str(output_path),
        "source_url": entry.source_url,
        "license_type": entry.license_type,
        "exists": output_path.exists(),
        "size_bytes": int(stat.st_size) if stat is not None else None,
    }


def _write_download_manifest(
    *,
    manifest_path: Path,
    plan: list[DownloadPlanEntry],
    outputs: dict[str, str],
) -> None:
    rows = [
        _manifest_row(entry, Path(outputs.get(entry.name, entry.target_path)))
        for entry in plan
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def download_commercial_datasets(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> dict[str, str]:
    """Download approved commercial-training artifacts from their source hosts."""

    root = Path(raw_dir)
    hf_cache_dir = root / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))

    outputs: dict[str, str] = {}
    plan = _download_plan(root, hf_cache_dir)
    for entry in plan:
        if entry.target_path.exists():
            logger.info(
                "Skipping dataset %s because %s already exists",
                entry.name,
                entry.target_path,
            )
            outputs[entry.name] = str(entry.target_path)
            continue
        logger.info("Downloading dataset %s from %s", entry.name, entry.source_url)
        outputs[entry.name] = str(entry.download_fn())
    _write_download_manifest(
        manifest_path=root / DEFAULT_DOWNLOAD_MANIFEST_PATH.name,
        plan=plan,
        outputs=outputs,
    )
    return outputs


def main() -> None:
    """Download the approved commercial datasets into the configured raw directory."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    outputs = download_commercial_datasets(raw_dir=DEFAULT_RAW_DIR)
    logger.info(
        "Commercial dataset downloads complete: %s",
        json.dumps(outputs, ensure_ascii=True),
    )


if __name__ == "__main__":
    main()
