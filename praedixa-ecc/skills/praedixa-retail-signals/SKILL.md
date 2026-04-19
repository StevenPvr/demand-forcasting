---
name: praedixa-retail-signals
description: Select and frame internal and external signals for Praedixa retail and
  perishable demand use cases. Use when choosing exogenous variables such as weather,
  calendar, events, promo, price, assortment, or local demand signals.
---

# Praedixa Retail Signals

## Read First

- `praedixa-ecc/rules/timeseries/leakage.md`
- `praedixa-ecc/rules/praedixa/retail-perishable-demand.md`

## Repo Rules To Apply

- Internal sources typically include:
  - POS
  - ERP
  - WFM / planning
  - CRM
  - stock systems
  - sales history
  - recipe sheets when useful for matter translation
- External sources typically include:
  - weather
  - calendar
  - local events
  - seasonality
  - Google Trends
  - macro or economic signals
- Prefer signals with operational plausibility and real-time availability.
- Reject signals that look impressive offline but are not truly available at decision time.

## Workflow

1. Separate internal signals from external signals.
2. Check real-time availability of each signal at prediction time.
3. Prioritize signals with plausible operational causality and good data quality.
4. Reject vanity signals that add story but no deployable value.
5. Note whether the signal helps demand, staffing, or both.

## Output

Return a ranked signal map with expected value and availability constraints.
