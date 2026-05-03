# Scenario reference YAML

YAML files in this directory encode **calibration starting points** for the restaurant POS simulator (normal operations, taxonomy defaults, extreme shocks, concrete configs).

Load via `praedixa.platform.datasets.synthetic_foodservice.spec.loader`:

- `load_activity_segmentation()`
- `load_location_baseline_multipliers()`
- `load_taxonomy_profiles()`
- `load_normal_scenarios()`
- `load_extreme_shocks()`
- `load_concrete_configurations()`

Human overview: [`docs/data_engineering/restaurant_simulator/DATA_DICTIONARY.md`](../../../../../../../../docs/data_engineering/restaurant_simulator/DATA_DICTIONARY.md) (sibling `PIPELINE_SPEC.md`, `EXPORT_LAYERS.md`, `VALIDATION.md`).
