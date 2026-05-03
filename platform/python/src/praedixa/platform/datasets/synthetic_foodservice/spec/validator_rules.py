"""Business invariants for post-generation validation of synthetic restaurant data.

These rules mirror the specification §21. Implementations run them on ``internal_clean``
first; ``pos_synth_export`` may violate some rules **by design** — then checks gate on
rows flagged ``data_anomaly_flags`` or ``correction_flag``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ValidationInvariant", "VALIDATION_INVARIANTS"]


@dataclass(frozen=True)
class ValidationInvariant:
    """Single check codified for automated test suites."""

    id: str
    description: str
    severity: str  # "error" | "warning"
    applies_to_layers: tuple[str, ...]


VALIDATION_INVARIANTS: tuple[ValidationInvariant, ...] = (
    ValidationInvariant(
        "INV-01-product-active-window",
        "No positive sale for product_id outside [launch_date, end_date] unless ticket carries explicit anomaly metadata.",
        "error",
        ("internal_clean", "ops_realistic"),
    ),
    ValidationInvariant(
        "INV-02-line-sum-vs-ticket-total",
        "sum(line_amount_ttc) equals total_amount_ttc - total_discount within rounding tolerance unless correction_flag.",
        "error",
        ("internal_clean",),
    ),
    ValidationInvariant(
        "INV-03-closure-coherence",
        "During closure intervals, observed tickets are absent or zero; export_gap pattern must match visibility flags.",
        "error",
        ("internal_clean", "ops_realistic"),
    ),
    ValidationInvariant(
        "INV-04-price-window",
        "unit_price_ttc/ht matches an effective prices row for (site, product[, channel], timestamp).",
        "error",
        ("internal_clean", "ops_realistic"),
    ),
    ValidationInvariant(
        "INV-05-promotion-window",
        "If promotion_id set, promotion interval contains ticket.datetime unless tagged promo error.",
        "warning",
        ("internal_clean", "ops_realistic"),
    ),
    ValidationInvariant(
        "INV-06-stockout-block",
        "During finished-good stockout without substitution, quantity sold for SKU is zero.",
        "error",
        ("internal_clean",),
    ),
    ValidationInvariant(
        "INV-07-stock-conservation",
        "Theoretical stock obeys conservation law modulo declared inventory_error shocks.",
        "error",
        ("internal_clean",),
    ),
    ValidationInvariant(
        "INV-08-holiday-calendar",
        "holidays rows align with scenario country/region and statutory calendar version recorded in simulation_parameters.",
        "error",
        ("internal_clean",),
    ),
    ValidationInvariant(
        "INV-09-seasonal-sku-months",
        "seasonal_flag products only have sales in allowed calendar months unless lifecycle exception logged.",
        "warning",
        ("internal_clean", "ops_realistic"),
    ),
    ValidationInvariant(
        "INV-10-latent-dominates-observed",
        "For each (site, product, time cell) without substitution inflow, observed qty <= latent qty + numerical eps.",
        "error",
        ("internal_clean",),
    ),
)


def invariant_ids() -> list[str]:
    return [inv.id for inv in VALIDATION_INVARIANTS]
