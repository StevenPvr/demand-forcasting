# Couche Bronze

La bronze locale est une projection tabulaire DuckDB des sources disponibles. Elle
ne prétend pas encore remplacer une vraie ingestion POS/ERP/WFM, mais elle porte
déjà les garanties nécessaires au développement commercial: audit, replay,
lignage source et refus des schémas cassés.

## Tables Actives

- `bronze_freshretail_daily`
- `bronze_bakery_order_lines`
- `bronze_supplemental_corpus_daily`
- `bronze_open_location_metadata`
- `bronze_open_location_catchment`
- `bronze_open_public_holidays`
- `bronze_open_school_holidays`
- `bronze_open_weather_daily`
- `bronze_open_macro_annual`
- `bronze_open_macro_timeseries`
- `bronze_source_manifest`

## Contrat Source

Chaque chargement ajoute ou conserve:

- `source_name`
- `source_policy_id`
- `source_file_path`
- `loaded_at`

Le manifest persistant `bronze_source_manifest` contient:

- `source_run_id`
- chemin source
- hash SHA-256
- taille fichier
- lignes chargées
- statut `required` / `allow_empty`
- `source_policy_id`

## Règles

- Les sources requises manquantes ou vides échouent avant mutation warehouse.
- Les colonnes attendues par une spec bronze échouent explicitement si absentes.
- Le chargement d'un batch de specs bronze est transactionnel: anciennes tables
  conservées si un remplacement tardif échoue.
- Les sources open exogenous peuvent être absentes, mais elles créent des tables
  vides contractuelles pour garder dbt stable.
- `source_policy_id` est la vérité légale aval: il doit rester stable et
  explicite pour chaque source autorisée.

## Frontière

Bronze ne fait pas de logique métier. Elle prépare des tables proches source,
auditables et suffisamment typées pour que staging/silver puissent normaliser.
