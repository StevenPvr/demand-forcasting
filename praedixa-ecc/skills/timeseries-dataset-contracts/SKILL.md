---
name: timeseries-dataset-contracts
description: Define, validate, or review time series dataset contracts for forecasting
  and ML pipelines. Use when a task touches grain definition, schema validation,
  series keys, target columns, missingness, stockouts, censoring, or dataset manifests.
---

# Time Series Dataset Contracts

Use this skill before trusting a forecasting dataset.

## Read First

- `praedixa-ecc/rules/timeseries/data-contracts.md`
- `praedixa-ecc/rules/timeseries/leakage.md`
- `praedixa-ecc/templates/dataset-manifest.md`

## Repo Rules To Apply

- Declare an explicit and stable canonical grain such as `date x store_id x sku_id`.
- Keep IDs as columns, not hidden in a persistent multi-index.
- Validate uniqueness at the canonical grain before trusting the dataset.
- The target must stay explicit and visible; do not hide it in the index.
- Distinguish observed demand from censored demand:
  - no sale
  - out of stock / stockout
  - closure
  - missing data
- Returns, cancellations, and negative quantities must be explained by an explicit business rule.
- Never forward-fill the target.
- Log missingness before and after imputation.
- Prefer Parquet for primary datasets, CSV only for samples/debug, JSON for manifests/metadata.
- Dates must stay typed as dates or timestamps, not free-form strings.
- Every persisted dataset should carry a schema contract and a manifest.

## Workflow

1. Declare the canonical grain, series key, target, cadence, and horizon.
2. Verify sort order, uniqueness at grain, and required columns.
3. Separate missing data from closures, stockouts, and true zero demand.
4. Check that joins do not silently mix grains.
5. Write or update a dataset manifest when the task is non-trivial.

## Output

Prefer a compact manifest plus a short list of contract violations or confirmed checks.
