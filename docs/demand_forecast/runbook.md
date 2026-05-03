# Runbook Demand Forecast

Ce runbook donne les commandes locales utiles pour reconstruire les surfaces
data et lire les resultats XGBoost courants. Il evite les runs longs inutiles:
verifier d'abord les artefacts et contrats avant de relancer une evaluation.

## Environnement

Depuis la racine du repo:

```bash
cd /Users/steven/Programmation/research_praedixa
export PYTHONPATH="$PWD:$PWD/platform/python/src:$PWD/products/demand_forecast/src"
```

Python local attendu:

```bash
.venv/bin/python
```

## Warehouse

Runner complet medaillon:

```bash
.venv/bin/python -m apps.warehouse.main
```

Runners specialises:

```bash
.venv/bin/python -m apps.warehouse.run_silver.main
.venv/bin/python -m apps.warehouse.run_gold.main
```

Verifier dbt sans reconstruire tout le pipeline:

```bash
.venv/bin/dbt parse --project-dir platform/warehouse --profiles-dir platform/warehouse
```

## Bundle Et Features

Construire le bundle demand forecast:

```bash
.venv/bin/python -m apps.demand_forecast.build_training_bundle.main
```

Selectionner les features du bundle:

```bash
.venv/bin/python -m apps.demand_forecast.select_training_bundle_features.main
```

Entrainer / tuner XGBoost:

```bash
.venv/bin/python -m apps.demand_forecast.run_xgboost_training.main
```

## Evaluation XGBoost Reference

Evaluation bakery reference XGBoost:

```bash
.venv/bin/python products/demand_forecast/src/praedixa/demand_forecast/evaluation/xgboost/main.py
```

Cette commande peut etre longue parce qu'elle refit un modele par jour test.
Avant de la relancer, verifier que les artefacts existants repondent deja a la
question.

## Lire Les Resultats

Metrices principales:

```bash
jq '{overall_metrics, business_impact}' \
  var/experiments/demand_forecast/evaluation/xgboost_test_metrics.json
```

`business_impact` dans ce fichier donne surtout la comparaison MAE et une
lecture simplifiée. Pour le gain en euros réaliste, utiliser le fichier
`xgboost_economic_gain_vs_best_baseline.json`.

Gain economique:

```bash
jq '{total_model_loss_eur,total_baseline_loss_eur,total_estimated_savings_eur_vs_best_baselines,total_absolute_mae_saved_vs_best_baselines,product_count,field_baseline_name}' \
  var/experiments/demand_forecast/evaluation/xgboost_economic_gain_vs_best_baseline.json
```

Contrat cible:

```bash
jq '.' var/experiments/demand_forecast/evaluation/xgboost_target_contract.json
```

Top produits par gain economique:

```bash
jq '.per_product_gain
  | to_entries
  | sort_by(.value.estimated_realistic_savings_eur_vs_best_baseline)
  | reverse
  | .[0:8]' \
  var/experiments/demand_forecast/evaluation/xgboost_economic_gain_vs_best_baseline.json
```

## Checks Anti-Leakage Cibles

Checks rapides a executer apres changement de split, feature engineering ou
contrat cible:

```bash
.venv/bin/python -m unittest \
  tests.products.demand_forecast.evaluation.test_reference_mode \
  tests.products.demand_forecast.evaluation.test_bakery_reference_dataset \
  tests.products.demand_forecast.evaluation.test_orchestrator_context \
  tests.products.demand_forecast.training_bundle.test_bundle_builder \
  tests.products.demand_forecast.contracts.test_targets
```

Pour un gate plus large:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Regles D'Execution

- Ne pas relancer une evaluation quotidienne complete pour une simple question
  de lecture de metrique.
- Lire d'abord les JSON et CSV d'artefacts.
- Apres un echec de run long, analyser le schema, les colonnes et le protocole
  avant toute relance.
- Documenter toute modification de la fonction de cout economique avant de
  comparer les gains en euros.
