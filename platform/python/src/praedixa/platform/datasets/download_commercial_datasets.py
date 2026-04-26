from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, cast
from urllib.request import Request
from urllib.request import urlopen

import datasets
from praedixa.platform.runtime.paths import SOURCES_DIR


logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = SOURCES_DIR / "commercial_datasets" / "raw"
DEFAULT_DOWNLOAD_MANIFEST_PATH = (
    DEFAULT_RAW_DIR / "commercial_dataset_download_manifest.json"
)
HTTP_TIMEOUT_SECONDS = 120
USER_AGENT = "praedixa-commercial-dataset-downloader/1.0"
DATASETS_API: Any = datasets


@dataclass(frozen=True)
class DownloadPlanEntry:
    """One approved commercial dataset artifact download."""

    name: str
    target_path: Path
    source_url: str
    license_type: str
    download_fn: Callable[[], Path]


def _download_hugging_face_split(
    *,
    dataset_name: str,
    split_name: str,
    output_path: Path,
    cache_dir: Path,
) -> Path:
    dataset_split = DATASETS_API.load_dataset(
        dataset_name, split=split_name, cache_dir=str(cache_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_split.to_parquet(str(output_path))
    logger.info(
        "Downloaded Hugging Face split %s/%s to %s",
        dataset_name,
        split_name,
        output_path,
    )
    return output_path


def _read_json_url(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8")
    return cast(dict[str, Any], json.loads(payload))


def _copy_url_to_temp(
    *,
    url: str,
    output_dir: Path,
    expected_content_type: str | None,
) -> Path:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        content_type = response.headers.get("Content-Type", "")
        if (
            expected_content_type is not None
            and expected_content_type not in content_type
        ):
            raise RuntimeError(f"Unexpected content type for {url}: {content_type}")
        with NamedTemporaryFile(
            "wb", delete=False, dir=output_dir, prefix=".download-", suffix=".tmp"
        ) as temp_file:
            shutil.copyfileobj(response, temp_file)
            return Path(temp_file.name)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(
    *,
    url: str,
    output_path: Path,
    expected_md5: str | None = None,
    expected_content_type: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _copy_url_to_temp(
        url=url,
        output_dir=output_path.parent,
        expected_content_type=expected_content_type,
    )
    try:
        if temp_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty artifact from {url}")
        if expected_md5 is not None and _file_md5(temp_path) != expected_md5:
            raise RuntimeError(f"Downloaded artifact checksum mismatch for {url}")
        temp_path.replace(output_path)
        output_path.chmod(0o644)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    logger.info("Downloaded %s to %s", url, output_path)
    return output_path


def _find_zenodo_file(record: dict[str, Any], file_name: str) -> dict[str, Any]:
    files = cast(list[dict[str, Any]], record.get("files", []))
    for file_item in files:
        if file_item.get("key") == file_name:
            return file_item
    raise RuntimeError(f"Zenodo record does not expose required file: {file_name}")


def _zenodo_md5(file_item: dict[str, Any]) -> str | None:
    checksum = str(file_item.get("checksum", ""))
    return checksum.removeprefix("md5:") if checksum.startswith("md5:") else None


def _download_zenodo_record_file(
    *,
    record_id: int,
    file_name: str,
    output_path: Path,
    expected_license_id: str,
) -> Path:
    record = _read_json_url(f"https://zenodo.org/api/records/{record_id}")
    metadata = cast(dict[str, Any], record.get("metadata", {}))
    license_info = cast(dict[str, Any], metadata.get("license", {}))
    if license_info.get("id") != expected_license_id:
        raise RuntimeError(
            f"Unexpected Zenodo license for record {record_id}: {license_info.get('id')}"
        )
    file_item = _find_zenodo_file(record, file_name)
    links = cast(dict[str, Any], file_item.get("links", {}))
    return _download_url(
        url=str(links["self"]),
        output_path=output_path,
        expected_md5=_zenodo_md5(file_item),
    )


def _kaggle_view_url(dataset_ref: str) -> str:
    return f"https://www.kaggle.com/api/v1/datasets/view/{dataset_ref}"


def _kaggle_download_url(dataset_ref: str, dataset_version: int) -> str:
    return (
        f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
        f"?datasetVersionNumber={dataset_version}"
    )


def _download_kaggle_dataset(
    *,
    dataset_ref: str,
    dataset_version: int,
    output_path: Path,
    expected_license_name: str,
) -> Path:
    metadata = _read_json_url(_kaggle_view_url(dataset_ref))
    if metadata.get("licenseName") != expected_license_name:
        raise RuntimeError(
            f"Unexpected Kaggle license for {dataset_ref}: {metadata.get('licenseName')}"
        )
    if int(metadata.get("currentVersionNumber", 0)) < dataset_version:
        raise RuntimeError(
            f"Kaggle dataset {dataset_ref} does not expose version {dataset_version}"
        )
    return _download_url(
        url=_kaggle_download_url(dataset_ref, dataset_version),
        output_path=output_path,
        expected_content_type="application/zip",
    )


def _plan_entry(
    *,
    name: str,
    target_path: Path,
    source_url: str,
    license_type: str,
    download_fn: Callable[[], Path],
) -> DownloadPlanEntry:
    return DownloadPlanEntry(
        name=name,
        target_path=target_path,
        source_url=source_url,
        license_type=license_type,
        download_fn=download_fn,
    )


def _hugging_face_plan_entries(
    root: Path, hf_cache_dir: Path
) -> list[DownloadPlanEntry]:
    dataset_name = "Dingdong-Inc/FreshRetailNet-LT"
    split_specs = [
        ("freshretail_lt_train", "train"),
        ("freshretail_lt_eval", "eval"),
    ]
    entries: list[DownloadPlanEntry] = []
    for artifact_name, split_name in split_specs:
        target_path = root / f"{artifact_name}.parquet"
        entries.append(
            _plan_entry(
                name=artifact_name,
                target_path=target_path,
                source_url="https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-LT",
                license_type="CC BY 4.0",
                download_fn=lambda split_name=split_name, target_path=target_path: (
                    _download_hugging_face_split(
                        dataset_name=dataset_name,
                        split_name=split_name,
                        output_path=target_path,
                        cache_dir=hf_cache_dir,
                    )
                ),
            )
        )
    return entries


def _zenodo_plan_entries(root: Path) -> list[DownloadPlanEntry]:
    return [
        _plan_entry(
            name="m5_forecasting_accuracy_zenodo",
            target_path=root / "m5_forecasting_accuracy_zenodo.zip",
            source_url="https://zenodo.org/records/12636070",
            license_type="CC BY 4.0",
            download_fn=lambda: _download_zenodo_record_file(
                record_id=12636070,
                file_name="m5-forecasting-accuracy.zip",
                output_path=root / "m5_forecasting_accuracy_zenodo.zip",
                expected_license_id="cc-by-4.0",
            ),
        ),
        _plan_entry(
            name="restaurant_sales_forecasting_zenodo",
            target_path=root / "restaurant_sales_forecasting_zenodo.docx",
            source_url="https://zenodo.org/records/5791480",
            license_type="CC BY 4.0",
            download_fn=lambda: _download_zenodo_record_file(
                record_id=5791480,
                file_name="ML_Sales_Forecasting_Schmidt_Supplement_Make.docx",
                output_path=root / "restaurant_sales_forecasting_zenodo.docx",
                expected_license_id="cc-by-4.0",
            ),
        ),
    ]


def _uci_plan_entries(root: Path) -> list[DownloadPlanEntry]:
    return [
        _plan_entry(
            name="uci_online_retail_ii",
            target_path=root / "uci_online_retail_ii.zip",
            source_url="https://archive.ics.uci.edu/dataset/502/online+retail+ii",
            license_type="CC BY 4.0",
            download_fn=lambda: _download_url(
                url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
                output_path=root / "uci_online_retail_ii.zip",
            ),
        ),
        _plan_entry(
            name="uci_online_retail",
            target_path=root / "uci_online_retail.zip",
            source_url="https://archive.ics.uci.edu/dataset/352/online+retail",
            license_type="CC BY 4.0",
            download_fn=lambda: _download_url(
                url="https://archive.ics.uci.edu/static/public/352/online+retail.zip",
                output_path=root / "uci_online_retail.zip",
            ),
        ),
    ]


def _maven_plan_entries(root: Path) -> list[DownloadPlanEntry]:
    return [
        _plan_entry(
            name="maven_cafe_rewards_offers",
            target_path=root / "maven_cafe_rewards_offers.zip",
            source_url="https://mavenanalytics.io/data-playground/cafe-rewards-offers",
            license_type="Public Domain",
            download_fn=lambda: _download_url(
                url="https://maven-datasets.s3.amazonaws.com/Cafe+Rewards+Offers/Cafe+Rewards+Offers.zip",
                output_path=root / "maven_cafe_rewards_offers.zip",
                expected_content_type="application/zip",
            ),
        ),
    ]


def _kaggle_plan_entries(root: Path) -> list[DownloadPlanEntry]:
    return [
        _plan_entry(
            name="perishable_goods_management_kaggle",
            target_path=root / "perishable_goods_management_kaggle.zip",
            source_url="https://www.kaggle.com/datasets/likithagedipudi/perishable-goods-management",
            license_type="CC0 Public Domain",
            download_fn=lambda: _download_kaggle_dataset(
                dataset_ref="likithagedipudi/perishable-goods-management",
                dataset_version=1,
                output_path=root / "perishable_goods_management_kaggle.zip",
                expected_license_name="CC0: Public Domain",
            ),
        ),
        _plan_entry(
            name="restaurant_sales_report_kaggle",
            target_path=root / "restaurant_sales_report_kaggle.zip",
            source_url="https://www.kaggle.com/datasets/rajatsurana979/fast-food-sales-report",
            license_type="Apache 2.0",
            download_fn=lambda: _download_kaggle_dataset(
                dataset_ref="rajatsurana979/fast-food-sales-report",
                dataset_version=2,
                output_path=root / "restaurant_sales_report_kaggle.zip",
                expected_license_name="Apache 2.0",
            ),
        ),
    ]


def _download_plan(root: Path, hf_cache_dir: Path) -> list[DownloadPlanEntry]:
    return [
        *_hugging_face_plan_entries(root, hf_cache_dir),
        *_zenodo_plan_entries(root),
        *_uci_plan_entries(root),
        *_maven_plan_entries(root),
        *_kaggle_plan_entries(root),
    ]


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
    """Download approved commercial-training artifacts from their source-of-truth hosts."""

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
