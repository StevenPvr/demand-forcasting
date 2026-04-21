# Workflow Local Praedixa

## Objectif

Figer un workflow local simple pour developper les couches `silver` et `gold` sans dependre du cloud.

Le mode local de reference est :

- `DuckDB local` pour la base warehouse
- `backup local` des inputs bronze
- `dbt` pour materialiser la silver et la gold
- `cloud DuckDB` desactive par defaut

## Commande unique

La commande la plus simple pour lancer la chaine locale est :

```bash
.venv/bin/python -m apps.warehouse.run_silver.main
```

Cette commande fait, dans l'ordre :

1. charge les datasets locaux dans le schema `bronze` DuckDB
2. lance `dbt run` sur la silver
3. lance `dbt test` sur la silver

Pour la `gold` :

```bash
.venv/bin/python -m apps.warehouse.run_gold.main
```

Cette commande :

1. suppose la `silver` deja disponible dans DuckDB
2. rafraichit les fichiers exogenes open-source
3. recharge uniquement les tables bronze exogenes
4. lance `dbt run` sur la `gold`
5. lance `dbt test` sur la `gold` uniquement

## Base locale utilisee

Par defaut, la base locale est :

```bash
var/warehouse/praedixa.duckdb
```

Schemas logiques :

- `bronze`
- `silver`
- `gold`

## Variables d'environnement utiles

Par defaut, le runner local force une configuration raisonnable :

- `PRAEDIXA_ENABLE_LOCAL_WAREHOUSE=true`
- `PRAEDIXA_ENABLE_CLOUD_WAREHOUSE=false`
- `PRAEDIXA_ENABLE_LOCAL_BACKUP=true`
- `PRAEDIXA_DUCKDB_LOCAL_PATH=var/warehouse/praedixa.duckdb`
- `PRAEDIXA_DUCKDB_TARGET_PATH=var/warehouse/praedixa.duckdb`
- `PRAEDIXA_DUCKDB_BRONZE_SCHEMA=bronze`
- `PRAEDIXA_DUCKDB_SILVER_SCHEMA=silver`
- `PRAEDIXA_DUCKDB_GOLD_SCHEMA=gold`
- `PRAEDIXA_GOLD_HORIZON_DAYS=1`
- `PRAEDIXA_GOLD_BAKERY_TEST_MONTHS=3`
- `PRAEDIXA_OPEN_EXOGENOUS_DIR=var/datasets/open_exogenous`

Tu peux les surcharger de l'exterieur si besoin.

Les 2 variables les plus utiles en pratique sont :

- `PRAEDIXA_DUCKDB_LOCAL_PATH`
- `PRAEDIXA_ENABLE_CLOUD_WAREHOUSE`

Notes utiles :

- la pipeline SQL `dbt` garde maintenant `FreshRetail` dans son echelle native observee
- le cloud DuckDB est implemente mais reste desactive par defaut
- la `gold` ne depend que de sources exogenes open-source publiques
- le runner bootstrap automatiquement `platform/warehouse/profiles.yml`
- le runner lance automatiquement `dbt deps` si `dbt_packages/` n'existe pas

## Variantes utiles

### Rejouer la silver sans recharger la bronze

```bash
.venv/bin/python -m apps.warehouse.run_silver.main --skip-bronze-load
```

### Lancer seulement le `dbt run` sans `dbt test`

```bash
.venv/bin/python -m apps.warehouse.run_silver.main --skip-dbt-tests
```

Version `gold` :

```bash
.venv/bin/python -m apps.warehouse.run_gold.main --skip-dbt-tests
```

### Rejouer la gold sans refetch des sources ouvertes

```bash
.venv/bin/python -m apps.warehouse.run_gold.main --skip-open-exogenous-refresh
```

### Cibler un sous-ensemble dbt

```bash
.venv/bin/python -m apps.warehouse.run_silver.main --select silver_commercial_external_daily_product_demand
```

En pratique, le runner ajoute les parents dbt automatiquement. Donc tu peux cibler un modele silver sans gerer a la main les dependances `staging`.

Version `gold` :

```bash
.venv/bin/python -m apps.warehouse.run_gold.main --select tag:gold
```

## Ce que signifie "bronze" localement

Dans l'etat actuel du repo :

- on n'a pas encore de vraie pipeline bronze depuis des JSON POS
- on a une **zone bronze DuckDB de substitution**
- cette zone bronze est alimentee depuis `var/sources/`

Donc le workflow actuel est :

- `var/sources/` -> `bronze` DuckDB -> `silver` dbt

et non encore :

- `raw JSON POS` -> `bronze tabulaire` -> `silver`

## Fichiers clefs

- runner local : [apps/warehouse/run_silver/main.py](/Users/steven/Programmation/research_praedixa/apps/warehouse/run_silver/main.py)
- chargeur bronze DuckDB : [apps/warehouse/load_bronze/main.py](/Users/steven/Programmation/research_praedixa/apps/warehouse/load_bronze/main.py)
- spec bronze : [docs/data_engineering/medaillon/bronze.md](/Users/steven/Programmation/research_praedixa/docs/data_engineering/medaillon/bronze.md)
- spec silver : [docs/data_engineering/medaillon/silver.md](/Users/steven/Programmation/research_praedixa/docs/data_engineering/medaillon/silver.md)
- spec gold : [docs/data_engineering/medaillon/gold.md](/Users/steven/Programmation/research_praedixa/docs/data_engineering/medaillon/gold.md)
- projet dbt : [platform/warehouse/README.md](/Users/steven/Programmation/research_praedixa/platform/warehouse/README.md)
