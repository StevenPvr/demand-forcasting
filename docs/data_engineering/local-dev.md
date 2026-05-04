# Workflow Local Praedixa

Ce document décrit le workflow local réellement supporté. Les wrappers `main.py`
sont volontairement lançables sans argument depuis l'IDE: pas de `argparse`, pas
de flags CLI.

## Commande De Référence

```bash
PYTHONPATH="$PWD:$PWD/platform/python/src:$PWD/products/demand_forecast/src" \
.venv/bin/python -m apps.warehouse.main
```

Ordre réel:

1. `load_core_bronze`: charge les sources locales coeur dans DuckDB.
2. `run_silver`: exécute dbt sur la silver canonique.
3. `refresh_open_exogenous`: lit `silver.silver_daily_product_demand`, produit les CSV open data, puis charge les `bronze_open_*`.
4. `run_gold`: matérialise le panel gold et le contrat `gold_training_matrix_d1`.

## Variantes Sans CLI

Les variantes se font dans le code importable via les dataclasses:

- `LocalSilverRunConfig`
- `LocalGoldRunConfig`
- `LocalMedallionRunConfig`
- `MedallionRunConfig`

Les wrappers sous `apps/` restent minces et sans flags. Pour un debug ciblé,
appeler la fonction importable correspondante dans un test ou un petit script
local temporaire non committé.

## Base Locale

Par défaut:

```text
var/warehouse/praedixa.duckdb
```

Schemas:

- `bronze`
- `silver`
- `gold`

Variables utiles:

- `PRAEDIXA_DUCKDB_LOCAL_PATH`
- `PRAEDIXA_DUCKDB_TARGET_PATH`
- `PRAEDIXA_DUCKDB_BRONZE_SCHEMA`
- `PRAEDIXA_DUCKDB_SILVER_SCHEMA`
- `PRAEDIXA_DUCKDB_GOLD_SCHEMA`
- `PRAEDIXA_OPEN_EXOGENOUS_DIR`
- `PRAEDIXA_DBT_THREADS`

Le cloud DuckDB/MotherDuck existe mais reste désactivé par défaut.

## Validation Légère Avant Run Long

Ne pas utiliser le médaillon complet comme outil de debug principal. Préférer:

```bash
PYTHONPATH="$PWD:$PWD/platform/python/src:$PWD/products/demand_forecast/src" \
.venv/bin/python -m unittest \
  tests.platform.warehouse.test_local_bronze \
  tests.platform.warehouse.test_local_silver \
  tests.platform.warehouse.test_local_gold \
  tests.platform.warehouse.test_local_medallion \
  tests.products.demand_forecast.training_bundle.test_bundle_builder

.venv/bin/dbt parse --project-dir platform/warehouse --profiles-dir platform/warehouse
```

Puis seulement:

```bash
PYTHONPATH="$PWD:$PWD/platform/python/src:$PWD/products/demand_forecast/src" \
.venv/bin/python -m apps.warehouse.main
```
