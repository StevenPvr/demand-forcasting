from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from praedixa.platform.runtime.paths import PROJECT_ROOT


DEFAULT_SOURCE_REGISTRY_PATH = PROJECT_ROOT / "platform" / "warehouse" / "seeds" / "source_registry.csv"
BOOL_COLUMNS = [
    "commercial_use_allowed",
    "ml_training_allowed",
    "caching_allowed",
    "attribution_required",
    "contract_required",
    "include_in_training",
]


@dataclass(frozen=True)
class SourceRegistryEntry:
    """Licensing and usage policy for one dataset or provider."""

    source_id: str
    source_kind: str
    dataset_source: str | None
    provider_name: str
    legal_basis: str
    license_type: str
    commercial_use_allowed: bool
    ml_training_allowed: bool
    caching_allowed: bool
    redistribution_mode: str
    attribution_required: bool
    attribution_text: str | None
    contract_required: bool
    review_status: str
    include_in_training: bool
    review_owner: str
    reviewed_at: str
    notes: str


def _coerce_bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in BOOL_COLUMNS:
        normalized[column] = (
            normalized[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )
    return normalized


def load_source_registry_frame(path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH) -> pd.DataFrame:
    """Load the shared source registry seed used by Python and dbt."""

    frame = pd.read_csv(Path(path), keep_default_na=False)
    return _coerce_bool_columns(frame)


def load_source_registry_entries(path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH) -> list[SourceRegistryEntry]:
    """Return typed source registry entries."""

    frame = load_source_registry_frame(path)
    records: list[SourceRegistryEntry] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            SourceRegistryEntry(
                source_id=str(row["source_id"]),
                source_kind=str(row["source_kind"]),
                dataset_source=str(row["dataset_source"]) if row["dataset_source"] else None,
                provider_name=str(row["provider_name"]),
                legal_basis=str(row["legal_basis"]),
                license_type=str(row["license_type"]),
                commercial_use_allowed=bool(row["commercial_use_allowed"]),
                ml_training_allowed=bool(row["ml_training_allowed"]),
                caching_allowed=bool(row["caching_allowed"]),
                redistribution_mode=str(row["redistribution_mode"]),
                attribution_required=bool(row["attribution_required"]),
                attribution_text=str(row["attribution_text"]) if row["attribution_text"] else None,
                contract_required=bool(row["contract_required"]),
                review_status=str(row["review_status"]),
                include_in_training=bool(row["include_in_training"]),
                review_owner=str(row["review_owner"]),
                reviewed_at=str(row["reviewed_at"]),
                notes=str(row["notes"]),
            )
        )
    return records


def allowed_training_dataset_sources(path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH) -> list[str]:
    """Return the dataset_source values allowed in commercial ML training."""

    frame = load_source_registry_frame(path)
    allowed = frame.loc[
        frame["source_kind"].eq("dataset")
        & frame["dataset_source"].ne("")
        & frame["commercial_use_allowed"]
        & frame["ml_training_allowed"]
        & frame["include_in_training"]
        & frame["review_status"].eq("allowed"),
        "dataset_source",
    ]
    return sorted(allowed.astype(str).unique().tolist())


def registry_entry_by_source_id(
    source_id: str,
    path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
) -> SourceRegistryEntry:
    """Return one registry entry by its source_id."""

    for entry in load_source_registry_entries(path):
        if entry.source_id == source_id:
            return entry
    raise KeyError(f"Unknown source registry id: {source_id}")


def _provider_review_allows_runtime(
    *,
    entry: SourceRegistryEntry,
    allow_contractual_providers: bool,
) -> bool:
    if entry.review_status == "blocked":
        return False
    if entry.review_status == "allowed":
        return True
    if entry.review_status == "quarantine" and entry.contract_required:
        return allow_contractual_providers
    return False


def is_provider_runtime_enabled(
    source_id: str,
    *,
    allow_contractual_providers: bool,
    path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
) -> bool:
    """Return whether one provider is enabled under the current commercial policy."""

    entry = registry_entry_by_source_id(source_id, path)
    if entry.source_kind != "provider":
        raise ValueError(f"Source `{source_id}` is not a provider.")
    return _provider_review_allows_runtime(
        entry=entry,
        allow_contractual_providers=allow_contractual_providers,
    )
