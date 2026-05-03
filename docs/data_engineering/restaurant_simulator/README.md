# Restaurant data simulator — specifications

This folder indexes the **normative** simulator specification (relational schema, scenarios, pipeline, exports, validation) for POS / stocks / planning / météo–style synthetic data.

| Document | Purpose |
|----------|---------|
| [DATA_DICTIONARY.md](./DATA_DICTIONARY.md) | Pointer to `data_dictionary.yaml` (tables A–T + extensions) |
| [PIPELINE_SPEC.md](./PIPELINE_SPEC.md) | Latent → observed → export cascade |
| [EXPORT_LAYERS.md](./EXPORT_LAYERS.md) | `internal_clean`, `ops_realistic`, `pos_synth_export` + MCAR/MAR |
| [VALIDATION.md](./VALIDATION.md) | Post-generation invariants (`validator_rules.py`) |

Python entrypoints live under `platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/` and `scenarios/`.

See also: [foodservice_synthetic_data_protocol.md](../foodservice_synthetic_data_protocol.md) for broader product context.
