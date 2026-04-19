---
name: forecast-baselines
description: Select, implement, or review baseline forecasts before comparing complex
  models. Use when a task needs naive, seasonal naive, moving-average, or business
  baselines for time series forecasting.
---

# Forecast Baselines

## Read First

- `praedixa-ecc/rules/timeseries/backtesting.md`
- `praedixa-ecc/rules/timeseries/metrics.md`

## Repo Rules To Apply

- Every candidate model must beat an explicit baseline, not an implicit previous model.
- Baselines to consider first:
  - naive
  - seasonal naive
  - moving average
  - simple business baseline if operationally relevant
- Baselines must be scored on the exact same splits and metrics as the candidate.
- For intermittent demand, consider dedicated baselines such as Croston, SBA, or TSB when relevant.
- Start simple before adding model complexity.
- A more complex model is justified only if gains are stable across backtests and meaningful in business terms.

## Workflow

1. Match the baseline family to the cadence and seasonality.
2. Start with naive and seasonal naive unless there is a better business baseline.
3. Add a simple operational baseline when promo or assortment rules matter.
4. Score all baselines on the same folds as the candidate model.
5. Reject any win that is not robust against baseline comparison.

## Output

Return a baseline set with rationale and the minimum comparison table to use.
