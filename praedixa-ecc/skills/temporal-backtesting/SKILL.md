---
name: temporal-backtesting
description: Design or review forecasting backtests with strict temporal splits, rolling-origin
  evaluation, and production-like refresh logic. Use when a task involves validation
  windows, fold design, horizon evaluation, or walk-forward reporting.
---

# Temporal Backtesting

## Read First

- `praedixa-ecc/rules/timeseries/backtesting.md`
- `praedixa-ecc/rules/timeseries/metrics.md`
- `praedixa-ecc/templates/experiment-spec.md`
- `praedixa-ecc/templates/backtest-report.md`

## Repo Rules To Apply

- No random split for forecasting tasks.
- Test stays untouched until the final evaluation.
- Split boundaries must live in config or an experiment spec.
- Backtesting must simulate production:
  - real horizon
  - real refresh cadence
  - real information availability
- Prefer rolling-origin or walk-forward over a single holdout.
- Evaluate on multiple windows, not just one lucky split.
- If lags or rolling windows need it, keep an explicit gap.
- If the use case needs uncertainty, support quantiles and evaluate interval quality, not just point forecasts.

## Workflow

1. State the business horizon and forecast cadence.
2. Choose train, validation, and test windows that match deployment reality.
3. Prefer walk-forward or rolling-origin folds over a single holdout.
4. Define refresh or retrain cadence explicitly.
5. Ensure baselines and candidate models are scored on identical folds.

## Output

Produce a concrete split plan and, when needed, a backtest report skeleton.
