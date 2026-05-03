# Praedixa Research

Repo de recherche, data engineering et evaluation pour le wedge Praedixa de
prevision de demande. Le repo sert a construire un pipeline auditable pour des
operations alimentaires perissables: boulangerie, restauration, snacking,
coffee shops et concepts multi-sites.

Praedixa ne cherche pas ici a produire un simple dashboard. L'objectif court
terme est de mieux anticiper la demande et les besoins operationnels; l'objectif
long terme est de relier ces previsions a de meilleures decisions economiques.

## Documentation

Point d'entree recommande:

- [Documentation repo](docs/README.md)
- [Architecture demand forecast](docs/demand_forecast/architecture.md)
- [Protocole d'evaluation](docs/demand_forecast/evaluation_protocol.md)
- [Resultats XGBoost 2026-04-30](docs/demand_forecast/results_xgboost_2026_04_30.md)
- [Runbook local](docs/demand_forecast/runbook.md)
- [Architecture medaillon](docs/data_engineering/medaillon/architecture_actuelle.md)

## Etat Actuel

- Source de verite data: `gold.gold_model_training_panel_d1`.
- Warehouse local: DuckDB + dbt, couches `bronze -> silver -> gold`.
- Backend courant exploitable: `xgboost`.
- Backend cible: TFT, encore en durcissement.
- Evaluation reference: bakery holdout comparable ARIMA.
- Cible ideale produit: demande latente quand les signaux de censure existent;
  cible de repli: ventes observees.

## Resultat Courant

Snapshot XGBoost du 2026-04-30:

| Mesure | Valeur |
| --- | ---: |
| test | 3 395 lignes |
| produits | 35 |
| fenetre test | 2022-06-26 -> 2022-09-30 |
| MAE modele | 6.191286637 |
| MAE baseline robuste | 5.954050074 |
| economie estimee vs baseline | 3 274.27 EUR |

Lecture correcte: le modele ne bat pas la baseline en MAE globale, mais il
reduit la perte economique sous la fonction de cout actuelle. Le resultat doit
donc etre presente comme un gain decisionnel / ROI, pas comme une victoire pure
sur l'erreur moyenne.

## Architecture Courte

```text
sources locales / open data / bakery / synthetic foodservice
-> bronze DuckDB
-> silver_daily_product_demand
-> gold.gold_model_training_panel_d1
-> training bundle / feature selection
-> XGBoost ou TFT
-> evaluation bakery reference
-> metrics + economic gain
```

## Commandes Essentielles

Depuis la racine:

```bash
export PYTHONPATH="$PWD:$PWD/platform/python/src:$PWD/products/demand_forecast/src"
```

Warehouse complet:

```bash
.venv/bin/python -m apps.warehouse.main
```

Gold seul:

```bash
.venv/bin/python -m apps.warehouse.run_gold.main
```

Bundle demand forecast:

```bash
.venv/bin/python -m apps.demand_forecast.build_training_bundle.main
```

Evaluation XGBoost bakery reference:

```bash
.venv/bin/python products/demand_forecast/src/praedixa/demand_forecast/evaluation/xgboost/main.py
```

Lecture rapide des resultats:

```bash
jq '{overall_metrics, business_impact}' var/experiments/demand_forecast/evaluation/xgboost_test_metrics.json
```

## Principes Non Negociables

- Split temporel strict.
- Pas de fuite du futur dans les features.
- Evaluation out-of-sample par date.
- Lecture economique separee de la MAE.
- Distinction explicite entre demande latente et ventes observees.
- Documentation des contrats cible, sources et artefacts avant promotion d'un
  resultat.

## Dossiers A Connaitre

- `apps/`: wrappers executables.
- `platform/warehouse/`: projet dbt + DuckDB.
- `platform/python/src/praedixa/platform/`: runtime, datasets, gouvernance,
  signaux open data.
- `products/demand_forecast/src/praedixa/demand_forecast/`: training,
  evaluation, contrats et backends.
- `tests/`: tests d'invariants repo.

## Verification Rapide

```bash
.venv/bin/dbt parse --project-dir platform/warehouse --profiles-dir platform/warehouse
.venv/bin/python -m unittest discover -s tests
```

Pour les commandes detaillees et les checks anti-leakage cibles, voir
[Runbook local](docs/demand_forecast/runbook.md).
