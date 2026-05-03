from __future__ import annotations

from dataclasses import dataclass


SYNTHETIC_QSR_SOURCE: str = "synthetic_foodservice_qsr"
SYNTHETIC_BAKERY_SOURCE: str = "synthetic_foodservice_bakery"
SYNTHETIC_RESTAURANT_SOURCE: str = "synthetic_foodservice_restaurant"
SYNTHETIC_BRASSERIE_SOURCE: str = "synthetic_foodservice_brasserie"
SYNTHETIC_COFFEE_SHOP_SOURCE: str = "synthetic_foodservice_coffee_shop"
SYNTHETIC_SNACK_SOURCE: str = "synthetic_foodservice_snack"
SYNTHETIC_DARK_KITCHEN_SOURCE: str = "synthetic_foodservice_dark_kitchen"
SYNTHETIC_SANDWICH_SHOP_SOURCE: str = "synthetic_foodservice_sandwich_shop"
SYNTHETIC_SEASONAL_TOURIST_SOURCE: str = "synthetic_foodservice_seasonal_tourist"

SYNTHETIC_DATASET_SOURCES: tuple[str, ...] = (
    SYNTHETIC_QSR_SOURCE,
    SYNTHETIC_BAKERY_SOURCE,
    SYNTHETIC_RESTAURANT_SOURCE,
    SYNTHETIC_BRASSERIE_SOURCE,
    SYNTHETIC_COFFEE_SHOP_SOURCE,
    SYNTHETIC_SNACK_SOURCE,
    SYNTHETIC_DARK_KITCHEN_SOURCE,
    SYNTHETIC_SANDWICH_SHOP_SOURCE,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE,
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
        site_share=0.22,
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
        site_share=0.20,
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
        site_share=0.12,
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
    VerticalSpec(
        dataset_source=SYNTHETIC_BRASSERIE_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="brasserie",
        site_share=0.10,
        product_count=180,
        min_site_products=8,
        max_site_products=22,
        min_daily_active_products=7,
        max_daily_active_products=20,
        base_units_low=2.0,
        base_units_high=45.0,
        promo_probability=0.05,
        stockout_tightness=0.10,
        zero_inflation=0.10,
        delivery_probability=0.30,
        pickup_probability=0.65,
        drive_through_probability=0.01,
    ),
    VerticalSpec(
        dataset_source=SYNTHETIC_COFFEE_SHOP_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="coffee_shop",
        site_share=0.10,
        product_count=120,
        min_site_products=10,
        max_site_products=28,
        min_daily_active_products=8,
        max_daily_active_products=25,
        base_units_low=3.0,
        base_units_high=55.0,
        promo_probability=0.07,
        stockout_tightness=0.08,
        zero_inflation=0.06,
        delivery_probability=0.25,
        pickup_probability=0.92,
        drive_through_probability=0.02,
    ),
    VerticalSpec(
        dataset_source=SYNTHETIC_SNACK_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="snack",
        site_share=0.08,
        product_count=100,
        min_site_products=6,
        max_site_products=18,
        min_daily_active_products=5,
        max_daily_active_products=16,
        base_units_low=2.0,
        base_units_high=40.0,
        promo_probability=0.09,
        stockout_tightness=0.14,
        zero_inflation=0.08,
        delivery_probability=0.55,
        pickup_probability=0.88,
        drive_through_probability=0.05,
    ),
    VerticalSpec(
        dataset_source=SYNTHETIC_DARK_KITCHEN_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="dark_kitchen",
        site_share=0.07,
        product_count=150,
        min_site_products=10,
        max_site_products=30,
        min_daily_active_products=8,
        max_daily_active_products=28,
        base_units_low=3.0,
        base_units_high=50.0,
        promo_probability=0.10,
        stockout_tightness=0.10,
        zero_inflation=0.05,
        delivery_probability=0.98,
        pickup_probability=0.15,
        drive_through_probability=0.01,
    ),
    VerticalSpec(
        dataset_source=SYNTHETIC_SANDWICH_SHOP_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="sandwich_shop",
        site_share=0.06,
        product_count=80,
        min_site_products=8,
        max_site_products=20,
        min_daily_active_products=6,
        max_daily_active_products=18,
        base_units_low=3.0,
        base_units_high=55.0,
        promo_probability=0.06,
        stockout_tightness=0.15,
        zero_inflation=0.07,
        delivery_probability=0.40,
        pickup_probability=0.90,
        drive_through_probability=0.02,
    ),
    VerticalSpec(
        dataset_source=SYNTHETIC_SEASONAL_TOURIST_SOURCE,
        vertical_level_1="food_service",
        vertical_level_2="seasonal_tourist",
        site_share=0.05,
        product_count=140,
        min_site_products=6,
        max_site_products=16,
        min_daily_active_products=5,
        max_daily_active_products=14,
        base_units_low=1.5,
        base_units_high=60.0,
        promo_probability=0.04,
        stockout_tightness=0.18,
        zero_inflation=0.12,
        delivery_probability=0.20,
        pickup_probability=0.45,
        drive_through_probability=0.01,
    ),
)

VERTICAL_SPEC_BY_SOURCE: dict[str, VerticalSpec] = {
    spec.dataset_source: spec for spec in VERTICAL_SPECS
}

PRODUCT_FAMILIES_BY_SOURCE: dict[str, tuple[str, ...]] = {
    SYNTHETIC_QSR_SOURCE: (
        "burger", "sandwich", "fried_chicken", "salad",
        "side", "dessert", "beverage", "breakfast", "bundle",
    ),
    SYNTHETIC_BAKERY_SOURCE: (
        "bread", "viennoiserie", "pastry", "sandwich",
        "salad", "snacking", "dessert", "beverage", "seasonal",
    ),
    SYNTHETIC_RESTAURANT_SOURCE: (
        "starter", "main_course", "dessert", "beverage",
        "kids_menu", "side", "daily_special", "brunch",
    ),
    SYNTHETIC_BRASSERIE_SOURCE: (
        "starter", "main_course", "dessert", "beverage",
        "wine", "side", "daily_special", "apero",
    ),
    SYNTHETIC_COFFEE_SHOP_SOURCE: (
        "hot_beverage", "cold_beverage", "pastry", "sandwich",
        "salad", "snack", "cake", "seasonal_special",
    ),
    SYNTHETIC_SNACK_SOURCE: (
        "kebab", "pizza", "sandwich", "burger",
        "fries", "beverage", "dessert",
    ),
    SYNTHETIC_DARK_KITCHEN_SOURCE: (
        "bowl", "burger", "poke", "salad",
        "dessert", "beverage", "bundle", "side",
    ),
    SYNTHETIC_SANDWICH_SHOP_SOURCE: (
        "sandwich", "wrap", "salad", "soup",
        "beverage", "dessert", "side",
    ),
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: (
        "main_course", "seafood", "dessert", "beverage",
        "ice_cream", "snack", "cocktail",
    ),
}

FRENCH_REGION_CODES: tuple[str, ...] = (
    "ARA", "BFC", "BRE", "CVL", "GES", "HDF",
    "IDF", "NOR", "NAQ", "OCC", "PDL", "PAC",
)
