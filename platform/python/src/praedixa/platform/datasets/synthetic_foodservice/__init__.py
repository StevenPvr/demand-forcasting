"""Synthetic foodservice data generation engine.

Generates realistic multi-vertical restaurant/QSR/bakery data with:
- Latent demand → observed sales censoring
- 9 business verticals (QSR, bakery, restaurant, brasserie, coffee_shop,
  snack, dark_kitchen, sandwich_shop, seasonal_tourist)
- Multiple granularity outputs (daily, tickets, lines, 15min, hourly,
  halfday, weekly, monthly)
- Operational tables (stock snapshots, inventory movements, staff schedules)
- POS data corruption by quality level
- Extreme shock injection (22 shock types)
- Statistical validation
"""

from __future__ import annotations

from praedixa.platform.datasets.synthetic_foodservice.config import (
    SyntheticFoodserviceConfig,
    build_default_synthetic_foodservice_config,
    build_smoke_synthetic_foodservice_config,
)
from praedixa.platform.datasets.synthetic_foodservice.exporter import (
    SyntheticFoodserviceArtifacts,
    generate_synthetic_foodservice_dataset,
)

__all__ = [
    "SyntheticFoodserviceArtifacts",
    "SyntheticFoodserviceConfig",
    "build_default_synthetic_foodservice_config",
    "build_smoke_synthetic_foodservice_config",
    "generate_synthetic_foodservice_dataset",
]
