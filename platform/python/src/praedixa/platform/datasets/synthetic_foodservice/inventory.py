"""Stock snapshot and inventory movement generation.

Produces daily stock positions and movements from production/sales/waste data
without ingredient-level BOM (per project decision: no ingredients needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STOCK_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "snapshot_id",
    "dataset_source",
    "location_id",
    "product_id",
    "dt",
    "theoretical_stock_qty",
    "observed_stock_qty",
    "stock_quality_flag",
    "inventory_error_level",
)

INVENTORY_MOVEMENT_COLUMNS: tuple[str, ...] = (
    "movement_id",
    "dataset_source",
    "location_id",
    "product_id",
    "dt",
    "movement_type",
    "quantity",
    "unit",
    "reason",
    "theoretical_flag",
    "manual_adjustment_flag",
)


def generate_stock_data(
    daily: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate stock snapshots and inventory movements from daily simulation."""

    if daily.empty or oracle.empty:
        return (
            pd.DataFrame(columns=STOCK_SNAPSHOT_COLUMNS),
            pd.DataFrame(columns=INVENTORY_MOVEMENT_COLUMNS),
        )
    rng = np.random.default_rng(seed + 77777)
    merged = daily.merge(
        oracle[
            [
                "dataset_source",
                "dt",
                "location_id",
                "product_id",
                "waste_qty_debug",
                "production_qty_debug",
            ]
        ],
        on=["dataset_source", "dt", "location_id", "product_id"],
        how="left",
    )
    snapshots = _build_snapshots(merged, rng)
    movements = _build_movements(merged, rng)
    return snapshots, movements


def _build_snapshots(merged: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 0
    for _, row in merged.iterrows():
        counter += 1
        production = float(row.get("production_qty_debug", 0) or 0)
        sold = float(row.get("observed_demand_qty", 0) or 0)
        waste = float(row.get("waste_qty_debug", 0) or 0)
        theoretical = max(production - sold - waste, 0.0)
        error = float(rng.normal(0, 0.08))
        observed = max(theoretical * (1.0 + error), 0.0)
        rows.append(
            {
                "snapshot_id": f"snap_{counter:010d}",
                "dataset_source": row["dataset_source"],
                "location_id": row["location_id"],
                "product_id": row["product_id"],
                "dt": row["dt"],
                "theoretical_stock_qty": round(theoretical, 2),
                "observed_stock_qty": round(observed, 2),
                "stock_quality_flag": "ok" if abs(error) < 0.15 else "discrepancy",
                "inventory_error_level": round(abs(error), 4),
            }
        )
    return pd.DataFrame(rows, columns=STOCK_SNAPSHOT_COLUMNS)


def _build_movements(merged: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 0
    for _, row in merged.iterrows():
        base = {
            "dataset_source": row["dataset_source"],
            "location_id": row["location_id"],
            "product_id": row["product_id"],
            "dt": row["dt"],
            "unit": "units",
        }
        production = float(row.get("production_qty_debug", 0) or 0)
        if production > 0:
            counter += 1
            rows.append(
                {
                    **base,
                    "movement_id": f"mv_{counter:010d}",
                    "movement_type": "production_output",
                    "quantity": round(production, 2),
                    "reason": "daily_production",
                    "theoretical_flag": True,
                    "manual_adjustment_flag": False,
                }
            )
        sold = float(row.get("observed_demand_qty", 0) or 0)
        if sold > 0:
            counter += 1
            rows.append(
                {
                    **base,
                    "movement_id": f"mv_{counter:010d}",
                    "movement_type": "outbound_sale",
                    "quantity": round(-sold, 2),
                    "reason": "pos_sale",
                    "theoretical_flag": True,
                    "manual_adjustment_flag": False,
                }
            )
        waste = float(row.get("waste_qty_debug", 0) or 0)
        if waste > 0:
            counter += 1
            rows.append(
                {
                    **base,
                    "movement_id": f"mv_{counter:010d}",
                    "movement_type": "outbound_waste",
                    "quantity": round(-waste, 2),
                    "reason": _waste_reason(rng),
                    "theoretical_flag": True,
                    "manual_adjustment_flag": False,
                }
            )
        if rng.random() < 0.03:
            counter += 1
            adj = float(rng.normal(0, 2.0))
            movement_type = "adjustment_up" if adj > 0 else "adjustment_down"
            rows.append(
                {
                    **base,
                    "movement_id": f"mv_{counter:010d}",
                    "movement_type": movement_type,
                    "quantity": round(adj, 2),
                    "reason": "inventory_count_correction",
                    "theoretical_flag": False,
                    "manual_adjustment_flag": True,
                }
            )
    return pd.DataFrame(rows, columns=INVENTORY_MOVEMENT_COLUMNS)


def _waste_reason(rng: np.random.Generator) -> str:
    reasons = ["expiry", "overproduction", "damaged", "quality", "other"]
    weights = [0.40, 0.30, 0.10, 0.10, 0.10]
    return str(rng.choice(reasons, p=weights))
