"""Generate ticket-level and line-level DataFrames from daily demand.

Distributes daily observed quantities across intraday time slots using the
vertical-specific profiles, then materialises individual tickets and lines.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.intraday import (
    SLOTS_PER_DAY,
    intraday_profile,
    slot_to_time_label,
)

TICKET_COLUMNS: tuple[str, ...] = (
    "ticket_id", "dataset_source", "location_id", "dt", "time_slot",
    "time_bucket_15min", "channel", "total_amount_ttc", "total_discount",
    "cancelled_flag", "refunded_flag", "correction_flag", "line_count",
)

LINE_COLUMNS: tuple[str, ...] = (
    "sales_line_id", "ticket_id", "dataset_source", "location_id",
    "product_id", "dt", "time_slot", "quantity", "unit_price_ttc",
    "line_amount_ttc", "discount_amount", "cancelled_flag",
    "menu_component_flag",
)


class TicketRow(TypedDict):
    ticket_id: str
    dataset_source: str
    location_id: str
    dt: object
    time_slot: int
    time_bucket_15min: str
    channel: str
    total_amount_ttc: float
    total_discount: float
    cancelled_flag: bool
    refunded_flag: bool
    correction_flag: bool
    line_count: int


class LineRow(TypedDict):
    sales_line_id: str
    ticket_id: str
    dataset_source: str
    location_id: str
    product_id: str
    dt: object
    time_slot: int
    quantity: float
    unit_price_ttc: float
    line_amount_ttc: float
    discount_amount: float
    cancelled_flag: bool
    menu_component_flag: bool


def generate_tickets_from_daily(
    daily: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert daily aggregates into ticket and line-level frames."""

    if daily.empty:
        return (
            pd.DataFrame(columns=TICKET_COLUMNS),
            pd.DataFrame(columns=LINE_COLUMNS),
        )
    rng = np.random.default_rng(seed + 31415)
    ticket_rows: list[TicketRow] = []
    line_rows: list[LineRow] = []
    ticket_counter: int = 0

    for source, source_frame in daily.groupby("dataset_source"):
        profile = intraday_profile(str(source))
        for (loc_id, dt), day_group in source_frame.groupby(["location_id", "dt"]):
            tickets, lines, ticket_counter = _distribute_day(
                day_group, profile, str(loc_id), dt, str(source),
                rng, ticket_counter,
            )
            ticket_rows.extend(tickets)
            line_rows.extend(lines)

    return (
        pd.DataFrame(ticket_rows, columns=TICKET_COLUMNS) if ticket_rows
        else pd.DataFrame(columns=TICKET_COLUMNS),
        pd.DataFrame(line_rows, columns=LINE_COLUMNS) if line_rows
        else pd.DataFrame(columns=LINE_COLUMNS),
    )


def _distribute_day(
    day_group: pd.DataFrame,
    profile: np.ndarray,
    location_id: str,
    dt: object,
    source: str,
    rng: np.random.Generator,
    ticket_counter: int,
) -> tuple[list[TicketRow], list[LineRow], int]:
    """Distribute a single site-day into tickets and lines."""

    tickets: list[TicketRow] = []
    lines: list[LineRow] = []
    products = day_group[day_group["observed_demand_qty"].astype(float) > 0]
    if products.empty:
        return tickets, lines, ticket_counter

    total_units = float(products["observed_demand_qty"].astype(float).sum())
    avg_basket = max(float(rng.lognormal(mean=0.8, sigma=0.3)), 1.2)
    n_tickets = max(1, int(round(total_units / avg_basket)))
    slot_probs = profile / profile.sum()
    slots: NDArray[np.int64] = rng.choice(SLOTS_PER_DAY, size=n_tickets, p=slot_probs)

    product_ids: NDArray[np.object_] = products["product_id"].to_numpy(dtype=object)
    quantities: NDArray[np.float64] = products["observed_demand_qty"].to_numpy(dtype=float)
    prices: NDArray[np.float64] = products["observed_revenue_net"].to_numpy(dtype=float)
    safe_qty = np.maximum(quantities, 1e-6)
    unit_prices = prices / safe_qty
    discounts: NDArray[np.float64] = products["observed_discount_amount"].to_numpy(dtype=float)
    unit_discounts = discounts / safe_qty

    qty_pool = quantities.copy()

    for i in range(n_tickets):
        ticket_counter += 1
        tid = f"tkt_{ticket_counter:010d}"
        slot = int(slots[i])
        t_lines, qty_pool = _build_ticket_lines(
            tid, source, location_id, dt, slot,
            product_ids, unit_prices, unit_discounts, qty_pool, rng,
        )
        if not t_lines:
            continue
        ticket_total = sum(float(ln["line_amount_ttc"]) for ln in t_lines)
        ticket_disc = sum(float(ln["discount_amount"]) for ln in t_lines)
        tickets.append({
            "ticket_id": tid, "dataset_source": source,
            "location_id": location_id, "dt": dt,
            "time_slot": slot, "time_bucket_15min": slot_to_time_label(slot),
            "channel": _random_channel(source, rng),
            "total_amount_ttc": round(ticket_total, 2),
            "total_discount": round(ticket_disc, 2),
            "cancelled_flag": False, "refunded_flag": False,
            "correction_flag": False, "line_count": len(t_lines),
        })
        lines.extend(t_lines)

    return tickets, lines, ticket_counter


def _build_ticket_lines(
    ticket_id: str,
    source: str,
    location_id: str,
    dt: object,
    slot: int,
    product_ids: NDArray[np.object_],
    unit_prices: NDArray[np.float64],
    unit_discounts: NDArray[np.float64],
    qty_pool: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[list[LineRow], NDArray[np.float64]]:
    """Allocate items from the remaining pool to one ticket."""

    available_mask = qty_pool > 0.5
    if not available_mask.any():
        return [], qty_pool

    n_lines = min(int(rng.integers(1, 5)), int(available_mask.sum()))
    chosen_indices = rng.choice(
        np.nonzero(available_mask)[0], size=n_lines, replace=False,
    )
    result: list[LineRow] = []
    line_counter = 0
    for idx in chosen_indices:
        product_index = int(idx)
        available_qty = float(qty_pool[product_index])
        take = min(float(rng.integers(1, 4)), available_qty)
        if take < 0.5:
            continue
        qty_pool[product_index] -= take
        line_counter += 1
        line_id = f"{ticket_id}_L{line_counter:02d}"
        unit_price = float(unit_prices[product_index])
        unit_discount = float(unit_discounts[product_index])
        line_amt = round(take * unit_price, 2)
        line_disc = round(take * unit_discount, 2)
        result.append({
            "sales_line_id": line_id, "ticket_id": ticket_id,
            "dataset_source": source, "location_id": location_id,
            "product_id": str(product_ids[product_index]), "dt": dt,
            "time_slot": slot, "quantity": round(take, 1),
            "unit_price_ttc": round(unit_price, 2),
            "line_amount_ttc": line_amt, "discount_amount": line_disc,
            "cancelled_flag": False, "menu_component_flag": False,
        })
    return result, qty_pool


def _random_channel(source: str, rng: np.random.Generator) -> str:
    roll = float(rng.random())
    if source == "synthetic_foodservice_dark_kitchen":
        return "delivery_platform" if roll < 0.92 else "takeaway"
    if roll < 0.50:
        return "on_site"
    if roll < 0.75:
        return "takeaway"
    return "delivery_platform"
