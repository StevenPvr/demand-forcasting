# Runbook Médaillon

## Vérification Rapide

```bash
PYTHONPATH="$PWD:$PWD/platform/python/src:$PWD/products/demand_forecast/src" \
.venv/bin/python -m unittest \
  tests.platform.warehouse.test_local_bronze \
  tests.platform.warehouse.test_local_silver \
  tests.platform.warehouse.test_local_gold \
  tests.platform.warehouse.test_local_medallion

.venv/bin/dbt parse --project-dir platform/warehouse --profiles-dir platform/warehouse
```

## Run Complet

```bash
PYTHONPATH="$PWD:$PWD/platform/python/src:$PWD/products/demand_forecast/src" \
.venv/bin/python -m apps.warehouse.main
```

## Contrôles DuckDB Utiles

```sql
select dataset_source, split_bucket, count(*)
from gold.gold_training_matrix_d1
group by 1, 2
order by 1, 2;

-- Contrat attendu:
-- sources d'entraînement autorisées: train + val uniquement, split chronologique 60/40.
-- bakery: test uniquement, fenêtre des 3 derniers mois.
-- Dans le bundle, gold.val devient tuning.parquet et gold.test devient valid.parquet.

select
    dataset_source,
    min(case when split_bucket = 'train' then dt end) as train_min_dt,
    max(case when split_bucket = 'train' then dt end) as train_max_dt,
    min(case when split_bucket = 'val' then dt end) as val_min_dt,
    max(case when split_bucket = 'val' then dt end) as val_max_dt,
    min(case when split_bucket = 'test' then dt end) as test_min_dt,
    max(case when split_bucket = 'test' then dt end) as test_max_dt
from gold.gold_training_matrix_d1
group by 1
order by 1;

select source_name, source_policy_id, loaded_rows, loaded_at
from bronze.bronze_source_manifest
order by loaded_at desc;

select *
from gold.gold_training_matrix_d1
where decision_timestamp >= cast(target_dt as timestamp);
```

## Règle D'Incident

Après un échec:

1. Lire l'erreur compilée dbt ou le test Python ciblé.
2. Corriger le contrat le plus proche de la cause.
3. Relancer la validation légère.
4. Ne relancer le médaillon complet qu'après stabilisation locale.
