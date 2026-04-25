# Couche Silver

Silver transforme les tables bronze en vérité opérationnelle standardisée. Elle
reste non agrégée au-delà du grain utile pour la prévision: pas de features ML
lourdes, pas de target future, pas de forward-fill de labels.

## Table Centrale

`silver_daily_product_demand`

Grain:

```text
dataset_source x dt x location_id x product_id
```

Colonnes critiques:

- `observed_demand_qty`
- `target_semantics`
- `censor_flag`
- `target_source`
- `label_quality_score`
- `usable_for_training_flag`

Aujourd'hui, la sémantique active est principalement `observed_sales`. La
demande latente ne doit être déclarée que si elle est réellement reconstruite.

## Gouvernance Source

Tables:

- `silver_source_registry`
- `silver_allowed_training_dataset_sources`
- `silver_allowed_provider_sources`
- `silver_quarantined_provider_sources`
- `silver_bronze_source_manifest`
- `silver_feature_registry`

Règles:

- les datasets training doivent être `allowed`, commercialement utilisables et
  autorisés pour ML;
- seules les sources provider présentes dans `silver_allowed_provider_sources`
  entrent dans les tables exogènes;
- les providers en quarantaine et les sources inconnues restent visibles via le
  manifest/staging, mais ne passent pas en silver exogène.

## Exogènes

Silver prépare les signaux externes au niveau propre:

- `silver_open_location_metadata`
- `silver_open_location_catchment`
- `silver_open_public_holiday_calendar_daily`
- `silver_open_school_holidays_daily`
- `silver_open_weather_daily`
- `silver_open_macro_country_daily`
- `silver_location_calendar`

Les signaux non disponibles restent absents ou `NULL`; ils ne doivent pas être
inventés. Gold décidera ensuite comment les transformer en covariables.

## Règles De Qualité

- pas de demande négative;
- `label_quality_score` entre 0 et 1;
- `censor_flag` explicite;
- provider non autorisé absent des tables open;
- unicité du grain canonique;
- matérialisation `table` par défaut, `incremental` uniquement si le modèle a un
  vrai `is_incremental()` et une clé non nulle.
