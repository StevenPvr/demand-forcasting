---
name: praedixa-staffing-forecast
description: Translate Praedixa demand signals into staffing forecasts and operational
  capacity questions. Use when a task touches labor planning, service coverage, staffing
  under/over-allocation, or the bridge between demand and workforce needs.
---

# Praedixa Staffing Forecast

## Read First

- `praedixa-ecc/rules/praedixa/business-framing.md`
- `praedixa-ecc/rules/praedixa/retail-perishable-demand.md`

## Repo Rules To Apply

- Staffing forecasting is downstream from demand, not independent from it.
- Make the translation assumptions explicit:
  - labor unit
  - service target
  - productivity assumption
  - time bucket
- The trade-off is always cost vs service level, not staffing in the abstract.
- Separate current observed staffing practice from recommended operational need.
- If the translation logic is heuristic, say so explicitly.

## Workflow

1. Start from the demand signal, then translate to staffing only with explicit assumptions.
2. State the staffing unit clearly: hours, shifts, heads, or service windows.
3. Distinguish observed staffing practice from recommended staffing need.
4. Call out where staffing translation is still heuristic rather than measured.
5. Express the trade-off between labor cost and service level.

## Output

Return staffing assumptions, translation logic, and operational implications.
