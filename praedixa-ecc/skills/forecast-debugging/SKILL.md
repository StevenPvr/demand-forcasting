---
name: forecast-debugging
description: Debug weak or unstable forecasting results. Use when performance regresses,
  bias drifts, segments collapse, or a model behaves well offline but poorly under
  realistic validation.
---

# Forecast Debugging

## Read First

- `praedixa-ecc/rules/timeseries/leakage.md`
- `praedixa-ecc/rules/timeseries/backtesting.md`
- `praedixa-ecc/rules/timeseries/metrics.md`

## Workflow

1. Verify the protocol before blaming the model.
2. Compare against baselines and isolate where the regression appears.
3. Check data drift, censoring, missingness, feature breakage, and horizon mismatch.
4. Separate global score issues from segment-specific failures.
5. Recommend the smallest next diagnostic step that increases confidence.

## Output

Return a ranked list of likely failure modes, supporting evidence, and next checks.
