from __future__ import annotations

from collections.abc import Callable
from collections.abc import Hashable
import math
from typing import Any, cast
from urllib.parse import urlencode

import pandas as pd

from praedixa.platform.signals.open_data.http import HttpJsonReader


OVERPASS_BASE_URL = "https://overpass-api.de/api/interpreter"
RESIDENTIAL_POPULATION_WEIGHTS = {
    "apartments": 40.0,
    "residential": 18.0,
    "house": 3.0,
    "detached": 3.0,
    "semidetached_house": 4.0,
    "terrace": 6.0,
}
COMPETITOR_TAGS_BY_SITE_FORMAT = {
    "bakery": {("shop", "bakery"), ("shop", "pastry"), ("shop", "confectionery")},
    "grocery": {
        ("shop", "supermarket"),
        ("shop", "convenience"),
        ("shop", "greengrocer"),
        ("shop", "bakery"),
    },
    "pharmacy": {("amenity", "pharmacy"), ("shop", "chemist")},
    "counter_service": {("amenity", "fast_food"), ("amenity", "cafe"), ("amenity", "restaurant")},
    "synthetic_food_service": {("amenity", "fast_food"), ("amenity", "cafe"), ("amenity", "restaurant")},
}


def _overpass_url(query: str) -> str:
    return f"{OVERPASS_BASE_URL}?{urlencode({'data': query})}"


def _poi_query(*, latitude: float, longitude: float, radius_meters: int) -> str:
    return "\n".join(
        [
            "[out:json][timeout:25];",
            "(",
            f'  nwr(around:{radius_meters},{latitude},{longitude})[office];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[building=office];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[amenity~"school|college|university|kindergarten"];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[railway~"station|halt|tram_stop|subway_entrance"];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[public_transport~"station|stop_position|platform"];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[amenity=bus_station];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[shop=mall];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[tourism];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[amenity=parking];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[building~"apartments|residential|house|detached|semidetached_house|terrace"];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[landuse=residential];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[shop~"bakery|pastry|confectionery|supermarket|convenience|greengrocer|chemist"];',
            f'  nwr(around:{radius_meters},{latitude},{longitude})[amenity~"pharmacy|fast_food|cafe|restaurant"];',
            ");",
            "out center tags;",
        ]
    )


def _element_key(element: dict[str, object]) -> tuple[str, object]:
    return (str(element.get("type", "node")), element.get("id"))


def _tags(element: dict[str, object]) -> dict[str, str]:
    raw_tags = element.get("tags", {})
    if isinstance(raw_tags, dict):
        typed_tags = cast(dict[object, object], raw_tags)
        return {str(key): str(value) for key, value in typed_tags.items()}
    return {}


def _count_matching_elements(
    elements: list[dict[str, object]],
    predicate: Callable[[dict[str, str]], bool],
) -> int:
    seen: set[tuple[str, object]] = set()
    for element in elements:
        tags = _tags(element)
        if predicate(tags):
            seen.add(_element_key(element))
    return len(seen)


def _estimate_population_proxy(elements: list[dict[str, object]]) -> float:
    total = 0.0
    seen: set[tuple[str, object]] = set()
    for element in elements:
        tags = _tags(element)
        element_key = _element_key(element)
        if element_key in seen:
            continue
        building_kind = tags.get("building")
        if building_kind in RESIDENTIAL_POPULATION_WEIGHTS:
            total += RESIDENTIAL_POPULATION_WEIGHTS[building_kind]
            seen.add(element_key)
            continue
        if tags.get("landuse") == "residential":
            total += 120.0
            seen.add(element_key)
    return total


def _competitor_predicate(site_format: str) -> Callable[[dict[str, str]], bool]:
    exact_matches = COMPETITOR_TAGS_BY_SITE_FORMAT.get(site_format, COMPETITOR_TAGS_BY_SITE_FORMAT["counter_service"])

    def predicate(tags: dict[str, str]) -> bool:
        return any(tags.get(tag_key) == tag_value for tag_key, tag_value in exact_matches)

    return predicate


def _parking_score(*, parking_count: int, transit_count: int) -> str:
    if parking_count >= 4:
        return "high"
    if parking_count >= 1:
        return "medium" if transit_count <= 2 else "low"
    return "low" if transit_count > 0 else "unknown"


def _transit_predicate(tags: dict[str, str]) -> bool:
    return (
        tags.get("railway") in {"station", "halt", "tram_stop", "subway_entrance"}
        or tags.get("public_transport") in {"station", "stop_position", "platform"}
        or tags.get("amenity") == "bus_station"
    )


def _fetch_poi_payload(
    *,
    latitude: float,
    longitude: float,
    radius_meters: int,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_json_reader: HttpJsonReader,
) -> object:
    return http_json_reader(
        _overpass_url(_poi_query(latitude=latitude, longitude=longitude, radius_meters=radius_meters)),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def _payload_elements(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    typed_payload = cast(dict[str, object], payload)
    elements = typed_payload.get("elements", [])
    if not isinstance(elements, list):
        return []
    raw_elements = cast(list[object], elements)
    return [cast(dict[str, object], element) for element in raw_elements if isinstance(element, dict)]


def _poi_elements_for_radius(
    *,
    latitude: float,
    longitude: float,
    radius_meters: int,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_json_reader: HttpJsonReader,
) -> list[dict[str, object]]:
    payload = _fetch_poi_payload(
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        http_json_reader=http_json_reader,
    )
    return _payload_elements(payload)


def _poi_elements_for_radii(
    *,
    latitude: float,
    longitude: float,
    radii_meters: tuple[int, int, int],
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_json_reader: HttpJsonReader,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    radius_500m, radius_1000m, radius_3000m = radii_meters
    return (
        _poi_elements_for_radius(
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius_500m,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            http_json_reader=http_json_reader,
        ),
        _poi_elements_for_radius(
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius_1000m,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            http_json_reader=http_json_reader,
        ),
        _poi_elements_for_radius(
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius_3000m,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            http_json_reader=http_json_reader,
        ),
    )


def _fetch_location_poi_elements(
    *,
    latitude: float,
    longitude: float,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_json_reader: HttpJsonReader,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    return _poi_elements_for_radii(
        latitude=latitude,
        longitude=longitude,
        radii_meters=(500, 1000, 3000),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        http_json_reader=http_json_reader,
    )


def _location_catchment_row(
    location: dict[str, object],
    elements_500m: list[dict[str, object]],
    elements_1km: list[dict[str, object]],
    elements_3km: list[dict[str, object]],
) -> dict[str, object]:
    competitor_predicate = _competitor_predicate(str(location.get("site_format", "counter_service")))
    transit_count_1km = _count_matching_elements(elements_1km, _transit_predicate)
    return {
        "dataset_source": location["dataset_source"],
        "location_id": location["location_id"],
        "population_1km": _estimate_population_proxy(elements_1km),
        "population_3km": _estimate_population_proxy(elements_3km),
        "office_poi_count_1km": _count_matching_elements(
            elements_1km, lambda tags: "office" in tags or tags.get("building") == "office"
        ),
        "school_poi_count_1km": _count_matching_elements(
            elements_1km,
            lambda tags: tags.get("amenity") in {"school", "college", "university", "kindergarten"},
        ),
        "transit_station_count_1km": transit_count_1km,
        "mall_poi_count_1km": _count_matching_elements(elements_1km, lambda tags: tags.get("shop") == "mall"),
        "tourism_poi_count_1km": _count_matching_elements(elements_1km, lambda tags: "tourism" in tags),
        "competitor_count_500m": _count_matching_elements(elements_500m, competitor_predicate),
        "competitor_count_1km": _count_matching_elements(elements_1km, competitor_predicate),
        "parking_score": _parking_score(
            parking_count=_count_matching_elements(elements_1km, lambda tags: tags.get("amenity") == "parking"),
            transit_count=transit_count_1km,
        ),
        "source_name": "openstreetmap_overpass_odbl",
    }


def _string_key_record(record: dict[Hashable, Any]) -> dict[str, object]:
    return {str(key): cast(object, value) for key, value in record.items()}


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def fetch_location_catchment_frame(
    location_metadata: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_json_reader: HttpJsonReader,
) -> pd.DataFrame:
    """Derive catchment proxies from OSM POIs around each known location."""

    rows: list[dict[str, object]] = []
    for raw_location in location_metadata.to_dict(orient="records"):
        location = _string_key_record(raw_location)
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        if _is_missing_scalar(latitude) or _is_missing_scalar(longitude):
            continue
        elements_500m, elements_1km, elements_3km = _fetch_location_poi_elements(
            latitude=float(cast(Any, latitude)),
            longitude=float(cast(Any, longitude)),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            http_json_reader=http_json_reader,
        )
        rows.append(_location_catchment_row(location, elements_500m, elements_1km, elements_3km))

    return pd.DataFrame(rows)
