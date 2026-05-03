from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.config import (
    SyntheticFoodserviceConfig,
)
from praedixa.platform.datasets.synthetic_foodservice.taxonomy import (
    FRENCH_REGION_CODES,
    PRODUCT_FAMILIES_BY_SOURCE,
    SYNTHETIC_BAKERY_SOURCE,
    VERTICAL_SPEC_BY_SOURCE,
    VERTICAL_SPECS,
)


@dataclass(frozen=True)
class SyntheticFoodserviceEntities:
    """Generated static entities used by the demand simulator."""

    cities: pd.DataFrame
    locations: pd.DataFrame
    products: pd.DataFrame
    assortments: pd.DataFrame
    location_metadata: pd.DataFrame


class LocationEntityRecord(Protocol):
    """Typed view of a pandas ``itertuples`` location row."""

    dataset_source: str
    location_id: str
    site_volume_factor: float


class ProductEntityRecord(Protocol):
    """Typed view of a pandas ``itertuples`` product row."""

    product_id: str
    base_price: float
    unit_material_cost: float
    product_popularity: float
    category_level_1: str
    category_level_2: str
    category_level_3: str


def build_synthetic_foodservice_entities(
    config: SyntheticFoodserviceConfig,
) -> SyntheticFoodserviceEntities:
    """Create deterministic cities, sites, product catalogs and assortments."""

    rng = np.random.default_rng(config.seed)
    cities = _build_city_frame(config=config, rng=rng)
    locations = _build_location_frame(config=config, cities=cities, rng=rng)
    required_bakery_product_ids = _load_required_bakery_product_ids(config)
    products = _build_product_frame(
        config=config,
        required_bakery_product_ids=required_bakery_product_ids,
        rng=rng,
    )
    assortments = _build_assortment_frame(
        config=config,
        locations=locations,
        products=products,
        required_bakery_product_ids=required_bakery_product_ids,
        rng=rng,
    )
    metadata = _build_location_metadata_frame(locations=locations)
    return SyntheticFoodserviceEntities(
        cities=cities,
        locations=locations,
        products=products,
        assortments=assortments,
        location_metadata=metadata,
    )


def _base_city_names() -> tuple[str, ...]:
    return (
        "Paris",
        "Marseille",
        "Lyon",
        "Toulouse",
        "Nice",
        "Nantes",
        "Montpellier",
        "Strasbourg",
        "Bordeaux",
        "Lille",
        "Rennes",
        "Reims",
        "Toulon",
        "Saint-Etienne",
        "Le Havre",
        "Grenoble",
        "Dijon",
        "Angers",
        "Nimes",
        "Villeurbanne",
        "Clermont-Ferrand",
        "Le Mans",
        "Aix-en-Provence",
        "Brest",
        "Tours",
        "Amiens",
        "Limoges",
        "Annecy",
        "Perpignan",
        "Boulogne-Billancourt",
        "Metz",
        "Besancon",
        "Orleans",
        "Rouen",
        "Mulhouse",
        "Caen",
        "Nancy",
        "Argenteuil",
        "Montreuil",
        "Roubaix",
        "Tourcoing",
        "Avignon",
        "Poitiers",
        "Versailles",
        "Courbevoie",
        "Creteil",
        "Pau",
        "La Rochelle",
        "Cannes",
        "Antibes",
    )


def _region_center(region_code: str) -> tuple[float, float]:
    centers = {
        "ARA": (45.76, 4.84),
        "BFC": (47.32, 5.04),
        "BRE": (48.11, -1.68),
        "CVL": (47.90, 1.91),
        "GES": (48.58, 7.75),
        "HDF": (50.63, 3.06),
        "IDF": (48.86, 2.35),
        "NOR": (49.44, 1.10),
        "NAQ": (44.84, -0.58),
        "OCC": (43.60, 1.44),
        "PDL": (47.22, -1.55),
        "PAC": (43.30, 5.37),
    }
    return centers[region_code]


def _build_city_frame(
    *,
    config: SyntheticFoodserviceConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    names = _base_city_names()
    rows: list[dict[str, object]] = []
    for index in range(config.city_count):
        region_code = FRENCH_REGION_CODES[index % len(FRENCH_REGION_CODES)]
        latitude, longitude = _region_center(region_code)
        city_suffix = index // len(names)
        city_name = (
            names[index % len(names)]
            if city_suffix == 0
            else f"{names[index % len(names)]} {city_suffix + 1}"
        )
        rows.append(
            {
                "city_id": f"syn_city_{index + 1:04d}",
                "city_name": city_name,
                "region_code": region_code,
                "country_code": "FR",
                "latitude": round(float(latitude + rng.normal(0.0, 0.18)), 5),
                "longitude": round(float(longitude + rng.normal(0.0, 0.22)), 5),
                "city_weight": float(rng.lognormal(mean=0.0, sigma=0.8)),
                "school_zone": ("A", "B", "C")[index % 3],
            }
        )
    return pd.DataFrame(rows)


def _location_source_sequence(config: SyntheticFoodserviceConfig) -> list[str]:
    counts = [round(spec.site_share * config.site_count) for spec in VERTICAL_SPECS]
    counts[-1] += config.site_count - sum(counts)
    return [
        source
        for spec, count in zip(VERTICAL_SPECS, counts, strict=True)
        for source in [spec.dataset_source] * count
    ]


def _build_location_frame(
    *,
    config: SyntheticFoodserviceConfig,
    cities: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sources = np.array(_location_source_sequence(config))
    rng.shuffle(sources)
    city_probabilities = cities["city_weight"].to_numpy(dtype=float)
    city_probabilities = city_probabilities / city_probabilities.sum()
    city_indices = rng.choice(len(cities), size=config.site_count, p=city_probabilities)
    rows = [
        _location_row(
            index,
            str(sources[index]),
            cities.iloc[int(city_indices[index])],
            rng,
        )
        for index in range(config.site_count)
    ]
    return pd.DataFrame(rows).sort_values("location_id").reset_index(drop=True)


def _location_row(
    index: int,
    dataset_source: str,
    city: pd.Series[Any],
    rng: np.random.Generator,
) -> dict[str, object]:
    spec = VERTICAL_SPEC_BY_SOURCE[dataset_source]
    return {
        "dataset_source": dataset_source,
        "location_id": f"syn_site_{index + 1:05d}",
        "city_id": city["city_id"],
        "city_name": city["city_name"],
        "country_code": city["country_code"],
        "region_code": city["region_code"],
        "region_id": city["region_code"],
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "school_zone": city["school_zone"],
        "org_group_id": f"syn_group_{rng.integers(1, 251):04d}",
        "drive_through_flag": bool(rng.random() < spec.drive_through_probability),
        "delivery_flag": bool(rng.random() < spec.delivery_probability),
        "pickup_flag": bool(rng.random() < spec.pickup_probability),
        "mall_flag": bool(rng.random() < 0.11),
        "transit_hub_flag": bool(rng.random() < 0.08),
        "tourism_flag": bool(rng.random() < 0.12),
        "site_volume_factor": float(rng.lognormal(mean=0.0, sigma=0.45)),
        "capacity_factor": float(rng.lognormal(mean=0.0, sigma=0.28)),
    }


def _load_required_bakery_product_ids(
    config: SyntheticFoodserviceConfig,
) -> tuple[str, ...]:
    path = config.bakery_product_names_csv_path
    if path is None or not path.exists():
        return ()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    product_col = _bakery_product_source_column(header)
    if product_col is None:
        return ()
    products = pd.read_csv(path, usecols=[product_col])[product_col]
    normalized = products.astype(str).str.strip().str.upper()
    return tuple(
        product_id
        for product_id in normalized.drop_duplicates().tolist()
        if product_id and product_id != "NAN"
    )


def _bakery_product_source_column(columns: list[str]) -> str | None:
    for candidate in ("article", "article_raw", "product_id"):
        if candidate in columns:
            return candidate
    return None


def _build_product_frame(
    *,
    config: SyntheticFoodserviceConfig,
    required_bakery_product_ids: tuple[str, ...],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in VERTICAL_SPECS:
        families = PRODUCT_FAMILIES_BY_SOURCE[spec.dataset_source]
        family_probabilities = _family_probabilities(families)
        product_count = _product_count_for_source(
            spec.dataset_source,
            base_count=spec.product_count,
            required_bakery_product_ids=required_bakery_product_ids,
        )
        for index in range(product_count):
            family = str(rng.choice(families, p=family_probabilities))
            product_id = _product_id_for_source(
                spec.dataset_source,
                index=index,
                required_bakery_product_ids=required_bakery_product_ids,
            )
            rows.append(
                _product_row(spec.dataset_source, product_id, family, index, rng)
            )
    return pd.DataFrame(rows)


def _product_count_for_source(
    dataset_source: str,
    *,
    base_count: int,
    required_bakery_product_ids: tuple[str, ...],
) -> int:
    if dataset_source != SYNTHETIC_BAKERY_SOURCE:
        return base_count
    return max(base_count, len(required_bakery_product_ids))


def _product_id_for_source(
    dataset_source: str,
    *,
    index: int,
    required_bakery_product_ids: tuple[str, ...],
) -> str:
    _ = required_bakery_product_ids
    return f"{dataset_source.replace('synthetic_foodservice_', 'syn_')}_sku_{index + 1:04d}"


def _family_probabilities(families: tuple[str, ...]) -> np.ndarray:
    weights = np.linspace(1.4, 0.7, num=len(families))
    return weights / weights.sum()


def _product_row(
    dataset_source: str,
    product_id: str,
    family: str,
    index: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    price = float(rng.lognormal(mean=2.0, sigma=0.38))
    return {
        "dataset_source": dataset_source,
        "product_id": product_id,
        "category_level_1": family,
        "category_level_2": f"{family}_sub_{index % 7}",
        "category_level_3": f"{family}_item_{index % 19}",
        "base_price": round(price, 2),
        "unit_material_cost": round(price * float(rng.uniform(0.22, 0.42)), 2),
        "product_popularity": float(rng.lognormal(mean=0.0, sigma=0.75)),
        "perishable_flag": family not in {"beverage"},
    }


def _build_assortment_frame(
    *,
    config: SyntheticFoodserviceConfig,
    locations: pd.DataFrame,
    products: pd.DataFrame,
    required_bakery_product_ids: tuple[str, ...],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_location in locations.itertuples(index=False):
        location = cast(LocationEntityRecord, raw_location)
        source = str(location.dataset_source)
        spec = VERTICAL_SPEC_BY_SOURCE[source]
        source_products = products.loc[products["dataset_source"].eq(source)].copy()
        menu_size = int(
            rng.integers(spec.min_site_products, spec.max_site_products + 1)
        )
        rows.extend(_assortment_rows(config, location, source_products, menu_size, rng))
    rows = _with_required_bakery_assortment_coverage(
        config=config,
        rows=rows,
        locations=locations,
        products=products,
        required_bakery_product_ids=required_bakery_product_ids,
        rng=rng,
    )
    return pd.DataFrame(rows)


def _with_required_bakery_assortment_coverage(
    *,
    config: SyntheticFoodserviceConfig,
    rows: list[dict[str, object]],
    locations: pd.DataFrame,
    products: pd.DataFrame,
    required_bakery_product_ids: tuple[str, ...],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    if not required_bakery_product_ids:
        return rows
    present = {
        str(row["product_id"])
        for row in rows
        if row["dataset_source"] == SYNTHETIC_BAKERY_SOURCE
    }
    missing = [
        product_id
        for product_id in required_bakery_product_ids
        if product_id not in present
    ]
    if not missing:
        return rows
    bakery_locations = locations.loc[
        locations["dataset_source"].eq(SYNTHETIC_BAKERY_SOURCE)
    ]
    bakery_products = products.loc[
        products["dataset_source"].eq(SYNTHETIC_BAKERY_SOURCE)
    ]
    if bakery_locations.empty or bakery_products.empty:
        return rows
    rows.extend(
        _required_bakery_assortment_rows(
            config=config,
            missing_product_ids=missing,
            bakery_locations=bakery_locations,
            bakery_products=bakery_products,
            rng=rng,
        )
    )
    return rows


def _required_bakery_assortment_rows(
    *,
    config: SyntheticFoodserviceConfig,
    missing_product_ids: list[str],
    bakery_locations: pd.DataFrame,
    bakery_products: pd.DataFrame,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, product_id in enumerate(missing_product_ids):
        location = cast(
            LocationEntityRecord,
            bakery_locations.iloc[index % len(bakery_locations)],
        )
        product_row = bakery_products.loc[bakery_products["product_id"].eq(product_id)]
        if product_row.empty:
            continue
        product = cast(ProductEntityRecord, product_row.iloc[0])
        rows.append(_site_product_row(config, location, product, rng))
    return rows


def _assortment_rows(
    config: SyntheticFoodserviceConfig,
    location: LocationEntityRecord,
    source_products: pd.DataFrame,
    menu_size: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    probabilities = source_products["product_popularity"].to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum()
    selected_indices = rng.choice(
        len(source_products),
        size=min(menu_size, len(source_products)),
        replace=False,
        p=probabilities,
    )
    selected = source_products.iloc[selected_indices]
    rows: list[dict[str, object]] = []
    for raw_product in selected.itertuples(index=False):
        rows.append(
            _site_product_row(
                config,
                location,
                cast(ProductEntityRecord, raw_product),
                rng,
            )
        )
    return rows


def _site_product_row(
    config: SyntheticFoodserviceConfig,
    location: LocationEntityRecord,
    product: ProductEntityRecord,
    rng: np.random.Generator,
) -> dict[str, object]:
    active_start = config.start_date + timedelta(
        days=int(rng.integers(0, 90)) if rng.random() < 0.08 else 0
    )
    active_end = config.end_date - timedelta(
        days=int(rng.integers(0, 120)) if rng.random() < 0.06 else 0
    )
    return {
        "dataset_source": location.dataset_source,
        "location_id": location.location_id,
        "product_id": product.product_id,
        "active_start": min(active_start, active_end),
        "active_end": active_end,
        "base_price": product.base_price,
        "unit_material_cost": product.unit_material_cost,
        "base_daily_units": float(product.product_popularity)
        * float(location.site_volume_factor),
        "category_level_1": product.category_level_1,
        "category_level_2": product.category_level_2,
        "category_level_3": product.category_level_3,
    }


def _build_location_metadata_frame(locations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_source",
        "location_id",
        "country_code",
        "region_code",
        "city_name",
        "latitude",
        "longitude",
        "school_zone",
        "drive_through_flag",
        "delivery_flag",
        "pickup_flag",
        "mall_flag",
        "transit_hub_flag",
        "tourism_flag",
    ]
    metadata = locations.loc[:, columns].copy()
    metadata["weather_location_label"] = "synthetic_foodservice_city_weather"
    metadata["assumption_source"] = "synthetic_foodservice_v1"
    return metadata
