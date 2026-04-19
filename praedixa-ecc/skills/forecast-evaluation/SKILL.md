---
name: forecast-evaluation
description: Evaluate forecasting outputs with business-relevant metrics, bias checks,
  and segmented analysis. Use when interpreting WAPE, MAE, RMSE, bias, segment performance,
  or translating forecast quality into operational meaning.
---

# Forecast Evaluation

## Read First

- `praedixa-ecc/rules/timeseries/metrics.md`
- `praedixa-ecc/templates/backtest-report.md`
- `praedixa-ecc/templates/business-translation-memo.md`

## Repo Rules To Apply

- Always report at least:
  - one level error metric
  - one bias metric
- Preferred metrics in this repo include `MAE`, `RMSE`, `WAPE` or `WMAPE`, `RMSSE`, and `Bias`.
- Do not stop at a global average.
- Segment results at least by:
  - horizon
  - product/category
  - store/region
  - volume bucket
  - promo vs non-promo
  - dense vs intermittent demand
- Make top contributors to volume or revenue visible.
- Evaluation must answer business questions:
  - where do we under-forecast?
  - where do we over-forecast?
  - which errors are most expensive?
  - does the model help the actual operational decision?
- A lower WAPE with materially worse bias is not a win.

## Workflow

1. Report at least one error metric and one bias metric.
2. Segment performance by horizon and by operationally relevant slices.
3. Identify where the model over-forecasts and under-forecasts.
4. Translate the evaluation into decision quality, not just score movement.
5. State clearly whether evidence is strong enough to keep the model.

## Output

Prefer a backtest summary followed by a short business translation memo.
