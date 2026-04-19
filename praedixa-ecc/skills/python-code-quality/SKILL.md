---
name: python-code-quality
description: Apply the repo's Python code-quality rules for structure, typing, dataframes,
  error handling, and tests. Use when writing, reviewing, or refactoring Python code
  in this repository, especially for pipelines, scripts, and tests.
---

# Python Code Quality

## Read First

- `praedixa-ecc/rules/python/coding-style.md`
- `praedixa-ecc/rules/python/patterns.md`
- `praedixa-ecc/rules/python/testing.md`
- `praedixa-ecc/scripts/python_doctor.py`
- `praedixa-ecc/scripts/quality_gate.py`

## Repo Rules To Apply

- Put `from __future__ import annotations` first in new source modules.
- Type public functions, config structures, and important variables.
- Use modern Python types when possible.
- Prefer `Path`, `dataclass`, and `Protocol` where they clarify the contract.
- Keep functions focused and modules reasonably small.
- Keep I/O, transformation, training, and evaluation clearly separated.
- Use `logging` in production code, not `print`.
- Catch specific exceptions and chain them when raising a business error.
- Avoid hardcoded secrets, tokens, or user-specific absolute paths.
- Keep DataFrame transformations explicit, staged, and testable.
- Stay consistent with the repo's current test style: `unittest`.
- Build tests from small but realistic time-series/retail fixtures.
- Add regression tests for leakage, lags/rolling, splits, schema, and metrics when relevant.

## Workflow

1. Apply the style and structure rules while editing.
2. Use `python_doctor` on touched Python files to catch rule violations quickly.
3. Use `quality_gate` before calling non-trivial Python work done.
4. Fix warnings deliberately or make the trade-off explicit.

## Output

Prefer code that is easy to test, easy to audit, and easy to reuse in the forecasting pipeline.
