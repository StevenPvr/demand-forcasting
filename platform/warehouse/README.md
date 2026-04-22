# Praedixa Warehouse

Projet `dbt + DuckDB` pour les couches `silver` et `gold` SQL-first de Praedixa.

## Cible

- moteur principal : `DuckDB`
- mode local : `duckdb` embarque dans un fichier `.duckdb`
- mode cloud : chemin DuckDB distant compatible `md:` / MotherDuck
- usage : couches `silver` et `gold` multi-source

## Variables d'environnement

Les 2 variables les plus utiles au quotidien sont :

- `PRAEDIXA_DUCKDB_LOCAL_PATH`
- `PRAEDIXA_ENABLE_CLOUD_WAREHOUSE`

Toutes les autres servent surtout aux cas de configuration avances.

- `PRAEDIXA_ENABLE_LOCAL_WAREHOUSE`
- `PRAEDIXA_ENABLE_CLOUD_WAREHOUSE`
- `PRAEDIXA_ENABLE_LOCAL_BACKUP`
- `PRAEDIXA_LOCAL_BACKUP_DIR`
- `PRAEDIXA_DUCKDB_LOCAL_PATH`
- `PRAEDIXA_DUCKDB_CLOUD_PATH`
- `PRAEDIXA_DUCKDB_CLOUD_TOKEN`
- `PRAEDIXA_DUCKDB_TARGET_PATH`
- `PRAEDIXA_DUCKDB_BRONZE_SCHEMA`
- `PRAEDIXA_DUCKDB_SILVER_SCHEMA`
- `PRAEDIXA_DUCKDB_GOLD_SCHEMA`
- `PRAEDIXA_GOLD_HORIZON_DAYS`
- `PRAEDIXA_GOLD_BAKERY_TEST_MONTHS`
- `PRAEDIXA_BRONZE_DATA_DIR`
- `PRAEDIXA_OPEN_EXOGENOUS_DIR`

## Mode local par defaut

Le cloud reste optionnel et desactive par defaut.

Par defaut, le chargeur bronze fonctionne avec :

- `PRAEDIXA_ENABLE_LOCAL_WAREHOUSE=true`
- `PRAEDIXA_ENABLE_CLOUD_WAREHOUSE=false`
- `PRAEDIXA_ENABLE_LOCAL_BACKUP=false`
- `PRAEDIXA_DUCKDB_LOCAL_PATH=var/warehouse/praedixa.duckdb`
- `PRAEDIXA_LOCAL_BACKUP_DIR=var/sources/local_backup`

Dans ce mode :

- les tables bronze sont chargees dans un DuckDB local
- aucun chargement cloud n'est tente
- les exogenes `gold` viennent uniquement de sources open-source publiques
- le runner local bootstrap automatiquement `platform/warehouse/profiles.yml` si besoin
- le runner local lance automatiquement `dbt deps` si `dbt_packages/` manque
- la `gold` peut etre materialisee localement dans le schema `gold`

Le backup local reste possible, mais seulement sur opt-in explicite via `PRAEDIXA_ENABLE_LOCAL_BACKUP=true`.

## Sources exogenes open-source

La `gold` V1 enrichit la silver avec des sources publiques uniquement :

- jours feries : `Nager.Date`
- vacances scolaires France : calendrier ICS ministeriel open data
- meteo historique : `Open-Meteo Archive API`
- macro annuelle lente : `World Bank Indicators API`
- macro France mensuelle / trimestrielle : `Insee BDM SDMX`

Important :

- la couche macro actuelle est `as-of` avec lags de publication conservateurs
- ce n'est pas encore une vraie couche `vintage` avec revisions historiques

Les fichiers telecharges sont ecrits localement par defaut dans :

```bash
var/datasets/open_exogenous
```

Puis rechargees dans le schema `bronze` avant le run dbt `gold`.

## FreshRetail

La pipeline SQL `dbt` conserve maintenant `FreshRetail` dans son echelle native observee.

Important :

- aucun multiplicateur runtime n'est applique dans `warehouse/models/staging/stg_freshretail_daily.sql`
- aucun multiplicateur runtime n'est applique dans `platform/warehouse/models/staging/stg_freshretail_daily.sql`
- les colonnes de compatibilite `freshretail_rescaled_flag` et `target_scale_assumption` restent exposees dans la `gold`
- elles valent desormais respectivement `false` et `native_observed`

## Activer le cloud DuckDB

Le mode cloud est prevu pour un endpoint DuckDB distant, par exemple un chemin MotherDuck du type `md:praedixa`.

Exemple :

```bash
export PRAEDIXA_ENABLE_CLOUD_WAREHOUSE=true
export PRAEDIXA_DUCKDB_CLOUD_PATH=md:praedixa
export PRAEDIXA_DUCKDB_CLOUD_TOKEN=ton_token
```

Tant que `PRAEDIXA_ENABLE_CLOUD_WAREHOUSE=false`, ce chemin n'est pas utilise.

## Sequence de run

1. Charger les bronze locales de substitution :

```bash
.venv/bin/python -m apps.warehouse.load_bronze.main
```

2. Lancer la silver avec la cible locale par defaut :

```bash
dbt run --project-dir platform/warehouse --profiles-dir platform/warehouse --select +tag:silver
```

3. Valider :

```bash
dbt test --project-dir platform/warehouse --profiles-dir platform/warehouse --select +tag:silver
```

## Sequence de run gold

1. Materialiser la `silver` :

```bash
.venv/bin/python -m apps.warehouse.run_silver.main
```

2. Materialiser la `gold` :

```bash
.venv/bin/python -m apps.warehouse.run_gold.main
```

3. Lancer manuellement dbt si besoin :

```bash
dbt run --project-dir platform/warehouse --profiles-dir platform/warehouse --select +tag:gold
dbt test --project-dir platform/warehouse --profiles-dir platform/warehouse --select tag:gold
```

## Workflow local simplifie

Pour lancer la chaine locale de bout en bout en une commande :

```bash
.venv/bin/python -m apps.warehouse.run_silver.main
```

Cette commande :

- recharge le schema `bronze` local dans DuckDB
- bootstrap `profiles.yml` si necessaire
- installe `dbt_utils` si necessaire via `dbt deps`
- lance `dbt run`
- lance `dbt test`

Pour la `gold` :

```bash
.venv/bin/python -m apps.warehouse.run_gold.main
```

Cette commande :

- telecharge / rafraichit les fichiers exogenes open-source
- recharge uniquement les tables bronze exogenes correspondantes
- bootstrap `profiles.yml` si necessaire
- installe `dbt_utils` si necessaire via `dbt deps`
- lance `dbt run` sur `+tag:gold`
- lance `dbt test` sur `tag:gold`

Si tu veux rejouer la `gold` sans refetch des sources ouvertes :

```bash
.venv/bin/python -m apps.warehouse.run_gold.main --skip-open-exogenous-refresh
```

## Basculer dbt vers une cible cloud

Pour faire pointer dbt vers une base DuckDB cloud, il suffit de surcharger le chemin cible :

```bash
export PRAEDIXA_DUCKDB_TARGET_PATH=md:praedixa
dbt run --project-dir platform/warehouse --profiles-dir platform/warehouse --select +tag:silver
```

Si le moteur distant demande un token, le fournir via `PRAEDIXA_DUCKDB_CLOUD_TOKEN` ou le mecanisme natif du provider.

Le meme mecanisme vaut pour la `gold` :

```bash
export PRAEDIXA_DUCKDB_TARGET_PATH=md:praedixa
dbt run --project-dir platform/warehouse --profiles-dir platform/warehouse --select +tag:gold
dbt test --project-dir platform/warehouse --profiles-dir platform/warehouse --select tag:gold
```
