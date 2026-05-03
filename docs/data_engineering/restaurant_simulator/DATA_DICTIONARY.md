# Restaurant simulator — data dictionary

This document is the human-facing index for the **machine-readable** schema shipped with the codebase.

## Schema version

- Current: **`1.0.0`** — defined in [`schema_version.py`](../../../../platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/schema_version.py) and echoed in [`data_dictionary.yaml`](../../../../platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/data_dictionary.yaml).

When you change columns, FK targets, or enums, bump **`SCHEMA_VERSION`** and the `schema_version` key in the YAML together.

## Authoritative files

| Asset | Path |
|-------|------|
| Full column dictionary + enums + FK | `platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/data_dictionary.yaml` |
| Version constant | `platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/schema_version.py` |

## Tables (A–T + extensions)

Mandatory tables from the product spec:

- **A** `sites` — establishments (with `location_id` FK for shared geography).
- **B** `products` — SKUs and lifecycle metadata.
- **C** `product_categories` — hierarchy and demand pattern tags.
- **D** `prices` — SCD2 price history (optional `channel` granularity).
- **E** `promotions` — planned vs actually applied offers.
- **F** `opening_hours` — theoretical hours by DOW and service type.
- **G** `closures` — exceptional non-trading intervals.
- **H** `sales_tickets` — POS headers.
- **I** `sales_lines` — line items.
- **J** `weather` — preferably keyed by `location_id`, optional `site_id` denorm.
- **K** `local_events` — shocks and demand drivers.
- **L** `holidays` — calendar (composite key `date`, `country`, `region`).
- **M** `inventory_movements` — receipts, adjustments, production I/O.
- **N** `stock_snapshots` — theoretical vs observed counts.
- **O** `stockouts` — OOS episodes and lost-sales diagnostics.
- **P** `production_batches` — batches (bakery / central prep).
- **Q** `waste` — shrinkage and invendus.
- **R** `staff_schedules` — capacity drivers only (not HR optimization).
- **S** `latent_demand_internal` — pre-censoring demand audit (`internal_only_flag`).
- **T** `simulation_parameters` — atomic run parameters (prefer one row per name).

**Extensions** (recommended for multi-site, dark kitchen, POS churn):

- `locations` — metro / timezone anchor for weather and events.
- `delivery_zones` — simplified routing geometry.
- `pos_versions` — software cutovers and mapping quality.
- `product_id_mappings` — SKU lineage and renames.
- `reservations` — optional traditional restaurant covers.

## Conventions (normative)

- **Money**: `decimal(10,2)`; TTC and HT columns explicit per field name.
- **VAT**: `vat_rate` `decimal(5,4)`.
- **Flags**: boolean in clean layers; dirty export may coerce to `0/1`.
- **Dependency levels**: integers `0–3` on `sites` for tourism / office / student / weather / event sensitivity unless scenario overrides to floats.
- **`data_quality_level`** on `sites`: drives export corruption profile — see `export_layers.yaml`.

## Python access

```python
from praedixa.platform.datasets.synthetic_foodservice.spec.loader import load_data_dictionary
from praedixa.platform.datasets.synthetic_foodservice.spec.schema_version import SCHEMA_VERSION

dd = load_data_dictionary()
assert dd["schema_version"] == SCHEMA_VERSION
```

## Related specs

- [Pipeline (latent → observed)](./PIPELINE_SPEC.md)
- [Export layers](./EXPORT_LAYERS.md)
- [Validation invariants](./VALIDATION.md)
