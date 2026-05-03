"""Latent demand → observed sales pipeline: ordered steps and conservation identities.

This module documents the **nominal** execution order for a Python implementation.
It does not run the simulator; it exposes structured metadata for code generation and tests.

Conservation (theoretical stock, continuous time):
    S_obs(t+) = S_obs(t-) + Inbound(t) + Produced(t) - Sold_obs(t) - Waste(t) - Adjustments(t)

Latent vs observed (per site, product, time cell):
    D_obs ≤ D_latent + Substitute_inflow_from_group
    where Substitute_inflow is demand rerouted from sibling SKUs in the same substitution_group.

Ticket coherence:
    sum(line_amount_ttc for lines in ticket) ≈ total_amount_ttc - total_discount  (± rounding_tol)

See also: docs/data_engineering/restaurant_simulator/PIPELINE_SPEC.md
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PIPELINE_STEPS",
    "PipelineStep",
    "ROUNDING_TOLERANCE_EUR_DEFAULT",
]

ROUNDING_TOLERANCE_EUR_DEFAULT: float = 0.02


@dataclass(frozen=True)
class PipelineStep:
    """One ordered transformation stage."""

    order: int
    key: str
    description: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]


# 1-based order: must run sequentially when building observed sales from the same random draws.
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        1,
        "calendar_open_intervals",
        "Intersect opening_hours with not(closures), timezone, DST; label export_gap vs zero_sales.",
        ("opening_hours", "closures", "holidays"),
        ("tradeable_intervals_by_site",),
    ),
    PipelineStep(
        2,
        "exogenous_grids",
        "Attach weather, local_events, holiday flags to each time cell.",
        ("weather", "local_events", "holidays"),
        ("exogenous_multipliers",),
    ),
    PipelineStep(
        3,
        "latent_intensity",
        "Compute log-multiplicative λ_site(t) then λ_{site,product,channel}(t) with seasonality.",
        ("sites", "products", "simulation_parameters"),
        ("latent_intensity",),
    ),
    PipelineStep(
        4,
        "latent_draws",
        "Draw counts via Poisson / NB / ZINB; compose tickets and basket structure (menus, complements).",
        ("latent_intensity",),
        ("latent_demand_internal", "synthetic_ticket_skeleton"),
    ),
    PipelineStep(
        5,
        "staff_capacity",
        "Map staff_schedules → K_kitchen(t), K_service(t); cap throughput (queues, abandonment).",
        ("staff_schedules",),
        ("capacity_envelope",),
    ),
    PipelineStep(
        6,
        "inventory_evolution",
        "Apply inbound movements, production_batches, shelf life, theoretical consumption.",
        ("inventory_movements", "production_batches", "products"),
        ("theoretical_stock_trace",),
    ),
    PipelineStep(
        7,
        "stockout_and_substitution",
        "Clip sales to availability; route unmet units to substitution_group or record lost sales.",
        ("theoretical_stock_trace", "latent_demand_internal"),
        ("stockouts", "observed_sales_pre_pos", "substitution_audit"),
    ),
    PipelineStep(
        8,
        "promotions_and_pricing",
        "Apply active prices + promo lifts; reconcile discounts to lines (pre-human-error).",
        ("prices", "promotions"),
        ("priced_lines",),
    ),
    PipelineStep(
        9,
        "pos_human_error",
        "Mis-keyed quantities, missing lines, wrong promotion_id, rare pre-close timestamps.",
        (),
        ("sales_tickets_pre_export", "sales_lines_pre_export"),
    ),
    PipelineStep(
        10,
        "waste_and_snapshot_close",
        "Finalize waste from invendus rules; optional stock_snapshots at day boundary.",
        ("observed_sales_pre_pos",),
        ("waste", "stock_snapshots"),
    ),
    PipelineStep(
        11,
        "layer_export_internal_clean",
        "Emit internal_clean bundle (full coherence, optional strip latent to separate path).",
        (),
        ("internal_clean_tables",),
    ),
    PipelineStep(
        12,
        "layer_export_ops_realistic",
        "Inject light inventory noise, partial stockout visibility.",
        (),
        ("ops_realistic_tables",),
    ),
    PipelineStep(
        13,
        "layer_export_pos_synth",
        "Apply data_quality_level rates: MCAR/MAR missing, duplicates, TZ drift, ID churn.",
        ("export_profiles",),
        ("pos_synth_tables",),
    ),
)
