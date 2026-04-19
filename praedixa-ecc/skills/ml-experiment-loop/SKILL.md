---
name: ml-experiment-loop
description: Run ML work as a hypothesis-driven experiment loop instead of opportunistic
  tweaking. Use when planning, executing, or reviewing experiments for model, feature,
  or protocol changes.
---

# ML Experiment Loop

## Read First

- `praedixa-ecc/rules/ml/experiment-integrity.md`
- `praedixa-ecc/templates/experiment-spec.md`
- `praedixa-ecc/templates/decision-log.md`

## Repo Rules To Apply

- Fix all randomness through config and log the seed.
- Log windows, horizon, granularity, and protocol for every serious experiment.
- Use versionable data artifacts and manifests for raw/intermediate/model-ready datasets.
- Dependencies should stay locked via `uv.lock`.
- Python should run through `.venv/bin/python` or `uv run python`.
- A notebook is not the source of truth for production pipeline logic.
- Do not change the protocol after seeing partial results just to salvage a run.
- End every experiment with an explicit decision:
  - keep
  - reject
  - investigate
- Prefer a simple, interpretable pipeline before escalating to heavier models.

## Workflow

1. Write the hypothesis before touching the code.
2. Freeze the protocol: data slice, backtest design, baselines, metrics, and success rule.
3. Run the comparison without changing the target after seeing partial results.
4. Summarize the evidence, not just the winning run.
5. End with an explicit decision log.

## Output

An experiment spec plus a `keep`, `reject`, or `investigate` decision.
