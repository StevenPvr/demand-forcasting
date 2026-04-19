---
name: timeseries-feature-engineering
description: Build or review time series feature engineering with safe lags, rolling
  features, calendar signals, and exogenous variables. Use when creating features
  for forecasting or checking whether proposed features are available at prediction time.
---

# Time Series Feature Engineering

## Read First

- `praedixa-ecc/rules/timeseries/leakage.md`
- `praedixa-ecc/rules/timeseries/data-contracts.md`

## Repo Rules To Apply

- Every feature must be available at the actual decision timestamp.
- Always `shift(1)` or stricter before rolling any statistic on the target.
- Lags and rolling windows must only use the past, never `t` or `t+h`.
- Features must be reproducible at inference time with the same information boundary as in backtest.
- Use business calendar signals, not only raw civil calendar:
  - holidays
  - vacations
  - end of month
  - payday
  - Black Friday
  - seasonal local events
- Exogenous variables are allowed only if their availability is real:
  - planned promo known in advance
  - weather forecast if actually available in the scenario
  - local events if known before the decision
- Reject magic features that cannot exist in production.
- If a DataFrame pipeline becomes opaque, break it into named, testable transformations.

## Workflow

1. Confirm the prediction timestamp and information actually available then.
2. Design lags, rolling statistics, calendar features, price/promo features, and exogenous signals.
3. Enforce `shift(1)` or stricter alignment before any rolling statistic on the target.
4. Reject any feature that cannot be recomputed in inference.
5. Call out fragile features, hidden future knowledge, and missing business semantics.

## Output

Return a feature plan or review with:

- accepted features
- rejected features
- leakage risks
- inference requirements
