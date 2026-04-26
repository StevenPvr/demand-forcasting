from __future__ import annotations

from dataclasses import dataclass


SYNTHETIC_QSR_SOURCE: str = "synthetic_foodservice_qsr"
SYNTHETIC_BAKERY_SOURCE: str = "synthetic_foodservice_bakery"
SYNTHETIC_RESTAURANT_SOURCE: str = "synthetic_foodservice_restaurant"
SYNTHETIC_DATASET_SOURCES: tuple[str, ...] = (
    SYNTHETIC_QSR_SOURCE,
    SYNTHETIC_BAKERY_SOURCE,
    SYNTHETIC_RESTAURANT_SOURCE,
)


@dataclass(frozen=True)
class VerticalSpec:
    """Operational parameters for one synthetic foodservice vertical."""

    dataset_source: str
    vertical_level_1: str
    vertical_level_2: str
    site_share: float
    product_count: int
    min_site_products: int
    max_site_products: int
    min_daily_active_products: int
    max_daily_active_products: int
    base_units_low: float
    base_units_high: float
    promo_probability: float
    stockout_tightness: float
    zero_inflation: float
    delivery_probability: float
    pickup_probability: float
    drive_through_probability: float


VERTICAL_SPECS: tuple[VerticalSpec, ...] = (
    VerticalSpec(
        dataset_source=SYNTHETIC_QSR_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="qsr_fast_food",
        site_share=0.40,
        product_count=180,
        min_site_products=8,
        max_site_products=24,
        min_daily_active_products=7,
        max_daily_active_products=22,
        base_units_low=4.0,
        base_units_high=95.0,
        promo_probability=0.11,
        stockout_tightness=0.12,
        zero_inflation=0.03,
        delivery_probability=0.92,
        pickup_probability=0.86,
        drive_through_probability=0.38,
    ),
    VerticalSpec(
        dataset_source=SYNTHETIC_BAKERY_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="bakery_snacking",
        site_share=0.35,
        product_count=320,
        min_site_products=12,
        max_site_products=44,
        min_daily_active_products=10,
        max_daily_active_products=40,
        base_units_low=2.0,
        base_units_high=70.0,
        promo_probability=0.06,
        stockout_tightness=0.20,
        zero_inflation=0.10,
        delivery_probability=0.35,
        pickup_probability=0.94,
        drive_through_probability=0.03,
    ),
    VerticalSpec(
        dataset_source=SYNTHETIC_RESTAURANT_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="traditional_restaurant",
        site_share=0.25,
        product_count=220,
        min_site_products=5,
        max_site_products=18,
        min_daily_active_products=5,
        max_daily_active_products=16,
        base_units_low=1.0,
        base_units_high=38.0,
        promo_probability=0.035,
        stockout_tightness=0.08,
        zero_inflation=0.14,
        delivery_probability=0.48,
        pickup_probability=0.52,
        drive_through_probability=0.01,
    ),
)

VERTICAL_SPEC_BY_SOURCE: dict[str, VerticalSpec] = {
    spec.dataset_source: spec for spec in VERTICAL_SPECS
}

PRODUCT_FAMILIES_BY_SOURCE: dict[str, tuple[str, ...]] = {
    SYNTHETIC_QSR_SOURCE: (
        "burger",
        "sandwich",
        "fried_chicken",
        "salad",
        "side",
        "dessert",
        "beverage",
        "breakfast",
        "bundle",
    ),
    SYNTHETIC_BAKERY_SOURCE: (
        "bread",
        "viennoiserie",
        "pastry",
        "sandwich",
        "salad",
        "snacking",
        "dessert",
        "beverage",
        "seasonal",
    ),
    SYNTHETIC_RESTAURANT_SOURCE: (
        "starter",
        "main_course",
        "dessert",
        "beverage",
        "kids_menu",
        "side",
        "daily_special",
        "brunch",
    ),
}

FRENCH_REGION_CODES: tuple[str, ...] = (
    "ARA",
    "BFC",
    "BRE",
    "CVL",
    "GES",
    "HDF",
    "IDF",
    "NOR",
    "NAQ",
    "OCC",
    "PDL",
    "PAC",
)
