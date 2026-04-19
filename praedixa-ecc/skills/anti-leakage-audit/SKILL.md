---
name: anti-leakage-audit
description: Audit a forecasting or ML pipeline for temporal leakage, hidden future
  information, and invalid evaluation logic. Use when reviewing features, transforms,
  joins, scalers, rolling windows, or suspicious offline performance.
---

# Anti-Leakage Audit

## Read First

- `praedixa-ecc/rules/timeseries/leakage.md`
- `praedixa-ecc/rules/timeseries/backtesting.md`

## Repo Rules To Apply

- Look first for the classic leakage paths:
  - random split
  - scaler fit on full data
  - lag misalignment
  - rolling windows that include the present or future
  - future promo knowledge not actually available
  - stockout interpreted as weak demand
- Verify that transforms and encoders are fit on train only.
- Confirm that the prediction timestamp is explicit and consistent across the pipeline.
- Treat censored demand as a first-class issue, not an afterthought.
- Suspicious offline gains should be treated as possible leakage until disproven.

## Workflow

1. Trace the exact prediction timestamp and the information available then.
2. Inspect feature generation, joins, rolling windows, imputations, and transforms.
3. Check that train-only fitting is respected for scalers and encoders.
4. Look for random splits, target leakage, future promo knowledge, and stockout confusion.
5. Report concrete leakage paths and propose the narrowest safe correction.

## Output

Lead with findings ordered by severity. Cite the leakage path precisely.
