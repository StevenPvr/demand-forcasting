# Foodservice Synthetic Data Protocol

Generated: 2026-04-25

## Decision

The best methodology for Praedixa is a **mechanistic, scenario-controlled simulator** that generates a first-party-like CSV bundle at the grain:

```text
dt x location_id x product_id
```

It should not start as a pure GAN/VAE/black-box synthetic-data model. For the current commercial objective, Praedixa needs a dataset that can prove operational value: demand forecasting, staffing needs, stockout/waste tradeoffs, and label quality under realistic restaurant, fast-food, and bakery constraints.

The simulator must therefore generate both:

1. **latent demand**: what would have been sold without operational constraints;
2. **observed sales**: what the POS would show after stockouts, closures, production caps, kitchen saturation, and channel constraints.

This is the key distinction. A synthetic CSV that only contains clean sales is too easy and commercially misleading.

## Why This Fits The Current Repo

The repo already expects a daily canonical demand panel and first-party onboarding feed:

- `platform/python/src/praedixa/platform/datasets/standardization/schema.py`
- `platform/python/src/praedixa/platform/datasets/standardization/first_party.py`
- `platform/python/src/praedixa/platform/datasets/standardization/quality.py`
- `platform/warehouse/models/silver/silver_daily_product_demand.sql`
- `platform/warehouse/models/gold/gold_base_panel_d1.sql`
- `platform/warehouse/macros/praedixa_gold_feature_slice.sql`
- `products/demand_forecast/src/praedixa/demand_forecast/training/validation/eligibility.py`

Important repo facts:

- The canonical global schema already includes `target_semantics`, `censor_flag`, `target_source`, `label_quality_score`, `usable_for_training_flag`, weather, promo, holiday, stockout, revenue and discount fields.
- The warehouse already computes D+1 target, lag features, rolling windows, cold-start buckets, split buckets, and operational flags.
- Training eligibility already excludes censored / low-quality rows.
- The README references `apps/platform/generate_synthetic_cold_start/main.py`, but that app is not present in the current file tree. A new implementation should therefore create this app rather than assume it exists.

## Research Synthesis

### Foodservice and bakery demand are operational forecasting problems

Restaurant forecasting research emphasizes internal POS/menu sales plus external variables. A 2024 Decision Support Systems paper on a large US restaurant chain reports the use of internal and external data, ML/DL models, stable vs turbulent periods, and macro/pandemic factors for restaurant demand forecasting: https://www.sciencedirect.com/science/article/pii/S0167923624001246

Bakery forecasting is especially relevant to Praedixa. Huber and Stuckenschmidt frame daily bakery demand forecasting at store/product-category level as input for production, ordering, and staffing decisions, with special calendar days causing demand patterns that differ sharply from regular days: https://www.sciencedirect.com/science/article/pii/S0169207020300224

Food-service waste reduction evidence supports the commercial wedge. Rodrigues et al. used real daily data from three catering services over 3 to 9 years, compared models to service estimates, and reported potential food waste reductions of 14% to 52% and unmet-demand reductions of 3% to 16%: https://www.sciencedirect.com/science/article/pii/S0959652623044232

FAO notes that better data availability on where food loss/waste occurs and its causes supports targeted interventions: https://www.fao.org/policy-support/policy-themes/food-loss-and-food-waste/fao-policy-series--food-loss---food-waste

### Stockout metadata is useful but not the current target

FreshRetailNet-50K is the closest methodological benchmark for stockout-aware
fresh retail data. The paper introduces a stockout-annotated fresh-retail dataset
and uses a two-stage approach for latent-demand recovery. That is useful
literature for a later experiment, but the current Praedixa pretraining contract
keeps the target as observed sales. The dataset contains 50,000 store-product
time series, 898 stores, 18 cities, perishable SKUs, hourly stock status, promo
discounts, precipitation, and temporal features: https://arxiv.org/abs/2505.16319

The Hugging Face dataset card states that FreshRetailNet-50K contains about 20% organically occurring stockout data and lists fields including `sale_amount`, hourly sales, stock status, discount, holiday/activity flags, precipitation, temperature, humidity and wind: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

Implication for Praedixa: the simulated CSV must include stockout/constraint
flags and, ideally, retain the hidden true latent demand in a restricted debug
column or manifest for simulator validation. The model-facing dataset should not
learn from this hidden ground truth unless explicitly running a controlled
experiment.

### Retail benchmarks support hierarchy, covariates, intermittency, and uncertainty

The M5 competition focused on 42,840 hierarchical Walmart retail sales time series, evaluated point and uncertainty forecasts, included exogenous variables, grouped/correlated series, and many intermittent series: https://www.sciencedirect.com/science/article/pii/S0169207021001187

Implication: the simulation must not only generate one average store. It needs a hierarchy:

- organization / franchise group
- location
- product category
- product
- channel / service model where useful

It also needs intermittent low-volume products, cold-start products, and correlated demand across products and sites.

### Synthetic time-series data must be validated, not only generated

Bahrpeyma et al. argue that synthetic time-series datasets need a methodological validation framework for diversity and coverage; otherwise they may favor one model type and undermine evaluation: https://www.sciencedirect.com/science/article/pii/S2215016121002521

A 2026 systematic review of longitudinal/time-series synthetic data notes that utility evaluations often focus on descriptive statistics and predictive performance, while inferential validity and realism are less consistently assessed: https://link.springer.com/article/10.1186/s12911-025-03326-8

The Synthetic Data Vault `PARSynthesizer` can learn multi-sequence data with a `sequence_key`, but it needs real multi-sequence data and creates new sequences by learning from them: https://docs.sdv.dev/sdv/sequential-data/modeling/parsynthesizer

Implication: SDV/PAR or VAE/GAN methods can become a later augmentation once Praedixa has first-party data. They are not the right primary methodology for the first commercial simulator, because Praedixa needs controllable ground truth, explicit censoring, and scenario stress tests.

### TFT alignment

Temporal Fusion Transformers are designed for multi-horizon forecasting with static covariates, known future inputs, and observed historical exogenous time series: https://arxiv.org/abs/1912.09363

Implication: the simulator should generate features with explicit availability semantics:

- static: site/product metadata
- known future: calendar, planned promo, planned closure, planned event
- observed historical: sales, stockout, day completeness, weather observations if only known after the day
- point-in-time forecast: weather forecast if used before decision time

This directly supports the repo's `feature_contract.py` and prevents temporal leakage.

## Methodological Choice

### Recommended approach

Use a **hybrid structural simulator**:

1. **Entity generator**
   Creates restaurant, fast-food, and bakery archetypes with realistic site/product metadata.

2. **Exogenous calendar/weather/event generator**
   Creates known-at-decision signals and observed-after-the-fact signals separately.

3. **Latent demand generator**
   Generates daily product demand from a zero-inflated / hurdle negative-binomial process with seasonality, site/product random effects, promotions, weather, holidays, events, trend, shocks, and cross-product cannibalization.

4. **Operational constraint layer**
   Converts latent demand to observed sales via production caps, stockouts, closures, kitchen saturation, channel disablement, product availability, and day completeness.

5. **Economic layer**
   Computes revenue, discounts, waste proxy, lost-sales proxy, material-cost proxy, and staffing workload.

6. **CSV export layer**
   Writes a first-party-compatible daily CSV plus optional profile CSVs and a simulation manifest.

7. **Validation gate**
   Checks schema, leakage, time-series diversity, business realism, and baseline/model behavior.

### Why not pure synthetic-data learning first

Pure SDV/PAR/GAN/TVAE generation is attractive later, but weak for the first high-stakes commercial simulation because:

- it needs enough real multi-site sequences to learn from;
- it does not naturally expose true latent demand vs observed sales;
- it can reproduce artifacts from real data without explaining the operational mechanism;
- it is harder to guarantee point-in-time feature semantics;
- it is weaker for counterfactual demos such as "what if staffing capacity drops 20%" or "what if promo causes stockout."

The simulator should be mechanistic first, calibrated with real/public references second, and only later hybridized with learned generative models.

## Core Data Generating Process

For each site `s`, product `p`, and date `t`, generate latent demand:

```text
log(mu_spt) =
    alpha_vertical
  + alpha_site_s
  + alpha_product_p
  + alpha_site_product_sp
  + dow_effect_vertical[t]
  + annual_seasonality[t]
  + holiday_effect[s,t]
  + bridge_day_effect[s,t]
  + school_holiday_effect[s,t]
  + weather_effect[vertical, product_family, t]
  + event_effect[s,t]
  + promo_effect[p,t]
  + price_discount_effect[p,t]
  + trend_spt
  + ar_state_spt
  + shared_site_shock_st
  + shared_vertical_shock_vt
  + cannibalization_effect_spt
```

Then:

```text
latent_demand_spt ~ hurdle_negative_binomial(mu_spt, dispersion_spt, zero_prob_spt)
```

Observed sales:

```text
available_qty_spt =
    production_qty_spt
  + opening_stock_spt
  - spoilage_spt

capacity_adjusted_demand_spt =
    latent_demand_spt * kitchen_capacity_factor_st * channel_capacity_factor_st

observed_demand_qty_spt =
    min(capacity_adjusted_demand_spt, available_qty_spt)
```

Censure:

```text
censor_flag = observed_demand_qty < latent_demand
observed_stockout_flag = available_qty < latent_demand
observed_stockout_intensity = (latent_demand - observed_demand_qty) / max(latent_demand, 1)
```

Training semantics:

```text
target_semantics = "observed_sales" by default
target_source = "observed_sales"
label_quality_score = 1.0 complete POS observation, 0.0 missing/closed/incomplete
usable_for_training_flag = label_quality_score >= 0.75 and target_source not missing/closed
```

For internal simulator diagnostics only, keep hidden columns in a debug output:

- `latent_demand_qty_debug`
- `lost_sales_qty_debug`
- `waste_qty_debug`
- `production_qty_debug`
- `staffing_required_hours_debug`

These should not be included in the model-facing CSV unless running a deliberate oracle experiment.

## Recommended CSV Bundle

### 1. Required daily feed

File:

```text
var/sources/commercial_datasets/raw/synthetic_foodservice_daily.csv
```

Minimum columns, aligned with current first-party onboarding:

```text
dt
location_id
product_id
observed_demand_qty
observed_revenue_net
observed_discount_amount
promo_flag
holiday_flag
activity_flag
observed_stockout_flag
observed_stockout_available
observed_stockout_intensity
day_complete_flag
event_name_1
event_type_1
event_name_2
event_type_2
weather_precipitation
weather_temperature
weather_humidity
weather_wind_level
```

Recommended additional columns for commercial realism:

```text
dataset_source
source_partition
source_run_id
series_id
vertical_level_1
vertical_level_2
region_id
org_group_id
category_level_1
category_level_2
category_level_3
target_semantics
censor_flag
target_source
label_quality_score
usable_for_training_flag
avg_selling_price
location_open_flag
product_active_flag
missing_sales_flag
anomaly_flag
location_closed_flag
kitchen_saturation_flag
channel_disabled_flag
assortment_restriction_flag
order_cutoff_flag
service_channel
daypart_mix_breakfast
daypart_mix_lunch
daypart_mix_dinner
delivery_share
pickup_share
onsite_share
planned_staff_hours
actual_staff_hours
staffing_need_hours
staffing_gap_hours
material_cost_estimate
gross_margin_estimate
waste_qty_observed
```

If strict repo compatibility is the priority, start with the existing first-party columns and let the warehouse/gold layer derive the rest. If commercial demo richness is the priority, include the additional operational fields now and update standardization/bronze specs accordingly.

### 2. Location profile

File:

```text
var/sources/commercial_datasets/raw/synthetic_location_profile.csv
```

Columns:

```text
location_id
country_code
city_name
region_id
org_group_id
vertical_level_1
vertical_level_2
site_format
service_model
drive_through_flag
delivery_flag
pickup_flag
late_night_flag
trade_area_type
mall_flag
transit_hub_flag
tourism_flag
office_density_bucket
residential_density_bucket
competition_intensity_bucket
seating_capacity
kitchen_capacity_units_per_hour
baseline_staff_hours
```

### 3. Product profile

File:

```text
var/sources/commercial_datasets/raw/synthetic_product_profile.csv
```

Columns:

```text
product_id
product_family
product_subfamily
category_level_1
category_level_2
category_level_3
menu_role
price_band
base_price
unit_material_cost
shelf_life_hours
prep_time_minutes
service_time_seconds
bundle_flag
core_menu_flag
add_on_flag
beverage_flag
dessert_flag
breakfast_flag
lunch_flag
perishable_flag
substitution_group_id
```

### 4. Manifest

File:

```text
var/sources/commercial_datasets/raw/synthetic_foodservice_manifest.json
```

Must include:

- seed
- date range
- number of locations/products/series
- vertical mix
- generator version
- parameter file hash
- stockout/constraint rate by vertical
- missingness rate
- stockout rate
- mean/median demand by vertical/product family
- list of columns hidden from model-facing export
- source policy metadata: `dataset_source=synthetic_v1`, commercial use allowed, ML training allowed

## Archetypes To Simulate

### Fast-food / QSR

Patterns:

- lunch and dinner peaks
- strong delivery/pickup channel share
- promotions and bundles
- weather and events affect traffic/channel mix
- kitchen saturation and labor gaps matter
- stockouts can be product-specific or packaging/channel-specific

Important fields:

- `delivery_flag`, `pickup_flag`, `drive_through_flag`
- `service_channel`
- `kitchen_saturation_flag`
- `planned_staff_hours`, `actual_staff_hours`, `staffing_gap_hours`
- `bundle_flag`, `core_menu_flag`, `beverage_flag`, `add_on_flag`

### Bakery / pastry / snacking

Patterns:

- strong morning peak
- same-day perishability
- special days and bridge days matter
- weather can change footfall
- overproduction creates waste, underproduction creates stockout/lost sales
- many products have short shelf life and daily production caps

Important fields:

- `shelf_life_hours`
- `production_qty_debug`
- `waste_qty_observed`
- `observed_stockout_flag`
- `holiday_flag`, `bridge_day`, `pre_holiday`, `post_holiday`

### Restaurant / traditional

Patterns:

- dinner/weekend concentration
- lower SKU count or product families rather than full ingredient-level menu
- reservations/events can shift demand
- labor constraints and service level matter
- menu substitutions/cannibalization matter

Important fields:

- `service_model`
- `seating_capacity`
- `event_name_1`, `event_type_1`
- `planned_staff_hours`, `actual_staff_hours`
- `assortment_restriction_flag`

## Validation Gates

### Gate 1: Contract and schema

Run the simulator output through:

- `standardize_first_party_daily_frame`
- `validate_canonical_frame`
- silver/gold dbt models once bronze support exists

Required invariants:

- unique `dataset_source x dt x location_id x product_id`
- non-negative demand and revenue unless returns are explicitly modeled
- `series_id = location_id__product_id`
- `censor_flag` not null
- `target_semantics in {"observed_sales", "latent_demand_estimated"}`
- stockout/constraint flags are descriptive for the observed-sales contract, not a
  default training exclusion
- no future target columns in model-facing export

### Gate 2: Time-series realism

Compute by vertical and product family:

- zero rate
- ADI / intermittency proxy
- coefficient of variation
- autocorrelation at lags 1, 7, 14, 28
- day-of-week lift
- holiday/special-day lift
- promo uplift
- weather elasticity by family
- cross-product correlation inside substitution groups
- stockout/constraint rate
- missingness / anomaly rate

Compare against:

- local `bakery_sales` aggregate behavior
- FreshRetailNet-inspired stockout and covariate structure
- M5-inspired hierarchy/intermittency logic
- manually defined business plausibility ranges

### Gate 3: Forecasting utility

Run at least:

- seasonal naive D-7
- moving average 7/28
- simple global XGBoost or existing backend surface
- future TFT once enabled

Read metrics by:

- vertical
- product family
- cold-start bucket
- clean vs censored
- stockout-prone vs normal
- high-margin/high-volume products

Metrics:

- WAPE
- MAE
- bias / WPE
- pinball loss if quantiles are generated
- stockout-weighted error
- waste-weighted error
- economic cost proxy

The simulator is acceptable only if better models beat naive baselines in scenarios where true exogenous drivers exist, and do not show impossible gains when drivers are intentionally removed.

### Gate 4: Leakage and point-in-time checks

Every generated column must be classified as:

- static
- known at decision time
- known only after the day
- hidden oracle/debug

No observed-after-the-fact variable should enter a pre-decision model. Weather must be split between:

- `weather_forecast_*` for pre-close/pre-production decisions
- `weather_observed_*` for post-close analysis

The current repo has a `post_close_d_plus_1` profile. If the synthetic dataset is meant to support morning production decisions, add a second profile:

```text
pre_open_same_day_or_d_plus_1
```

and block current-day observed sales, current-day stockout, and observed weather.

## Implementation Plan

### Phase 1: Minimal simulator compatible with current first-party path

Add:

```text
platform/python/src/praedixa/platform/datasets/synthetic_foodservice/
  __init__.py
  config.py
  entities.py
  exogenous.py
  demand.py
  operations.py
  economics.py
  exporter.py
  quality.py
  main.py

apps/platform/generate_synthetic_cold_start/
  __init__.py
  main.py

tests/platform/datasets/synthetic_foodservice/
  test_config.py
  test_entities.py
  test_demand.py
  test_operations.py
  test_exporter.py
  test_quality.py
```

Output first:

```text
synthetic_foodservice_daily.csv
synthetic_location_profile.csv
synthetic_product_profile.csv
synthetic_foodservice_manifest.json
```

Run the generator once, before the medallion pipeline:

```bash
/Users/steven/Programmation/research_praedixa/.venv/bin/python /Users/steven/Programmation/research_praedixa/platform/python/src/praedixa/platform/datasets/synthetic_foodservice/main.py
```

Operational contract:

- this command is a one-shot source-generation step;
- it writes the stable source CSV under `var/sources/commercial_datasets/raw/synthetic_foodservice_daily.csv`;
- the medallion pipeline ingests that CSV afterward;
- the medallion pipeline must not regenerate synthetic data on each run.
- synthetic foodservice rows are training-only augmentation data in gold:
  `split_bucket = 'train'` only, never `val` and never `test`;
- the default generation scale is intentionally reduced to roughly one tenth of
  the previous reduced run, and about one three-hundredth of the first full-scale
  run: 180 sites, 180 cities, 12 months of history, compact realistic top-SKU
  assortments, and about 1 million daily source rows.

### Phase 2: Canonical standardization

Add a standardizer that maps synthetic output into the existing canonical schema, similar to:

- `first_party.py`
- `supplemental_corpus_standardizers.py`

Recommended file:

```text
platform/python/src/praedixa/platform/datasets/standardization/synthetic_foodservice.py
```

Then add `synthetic_v1` to source registry if not already present in current seeds.

### Phase 3: Warehouse integration

Add bronze specs and silver model only if synthetic data should enter the warehouse automatically. If this is only an onboarding/test feed, keep it as a first-party-compatible CSV and standardize directly.

### Phase 4: Scenario library

Create parameter presets:

```text
configs/synthetic_foodservice/bakery_small_chain.yaml
configs/synthetic_foodservice/qsr_franchise_10_sites.yaml
configs/synthetic_foodservice/mixed_foodservice_pilot.yaml
configs/synthetic_foodservice/stress_stockout_weather.yaml
configs/synthetic_foodservice/cold_start_new_products.yaml
```

Each preset should define:

- location count
- product count
- date range
- vertical mix
- stockout/constraint target rate
- missingness target rate
- event intensity
- promo frequency
- capacity tightness
- staffing tightness
- seed

## Critical Risks

1. **Synthetic data too clean**
   If there are no stockouts, anomalies, closures, missing days, low-volume SKUs, and noisy events, the model will look better than it is.

2. **Latent demand leaked**
   If `latent_demand_qty_debug`, future weather, or target-day observed variables enter training, offline metrics become commercially useless.

3. **One average restaurant**
   A single generic demand curve will not prepare Praedixa for franchised multi-site operations. Site/product heterogeneity is mandatory.

4. **No operational bottleneck**
   Forecasting value comes from better production, ordering, and staffing decisions. Without capacity, waste, stockout, and staffing variables, the simulation misses the ROI layer.

5. **No validation against external references**
   The simulator must be calibrated against observed bakery data in the repo and public fresh-retail/forecasting literature. Pure intuition is not enough.

## Final Recommendation

Implement a **calibrated structural simulator** with hidden latent demand and visible observed sales.

For the first version, target:

- 12 months daily history
- hundreds of locations
- hundreds of cities
- compact vertical-specific product assortments
- 3 vertical archetypes: bakery, QSR/fast-food, traditional restaurant
- 5% to 20% stockout/constraint-flagged rows depending scenario
- 5% to 15% promo days
- realistic holidays, bridge days, weather and events
- at least 10% cold-start or low-history series
- output compatible with first-party onboarding and canonical schema

This gives Praedixa a credible commercial demo asset and a serious engineering testbed for the wedge: demand and staffing forecast for foodservice/perishable operations, with measurable ROI logic.
