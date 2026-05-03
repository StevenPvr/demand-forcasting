# Couche Gold

Gold produit le dataset ML D+1 du wedge Praedixa: prévoir la demande du lendemain
à partir d'un état de connaissance disponible à la fin du jour D.

## Tables Actives

- `gold_base_panel_d1`
- `gold_feature_bakery_d1`
- `gold_feature_synthetic_foodservice_d1`
- `gold_feature_panel_d1`
- `gold_source_weights_d1`
- `gold_daily_product_forecast_panel_d1`
- `gold_model_training_panel_d1`
- `gold_run_manifest`

La table consommée par le bundle est `gold_model_training_panel_d1`.

`gold_feature_supplemental_corpus_d1` existe comme surface de modèle mais est
désactivée dans l'état courant. L'union active de `gold_feature_panel_d1`
combine `gold_feature_bakery_d1` et `gold_feature_synthetic_foodservice_d1`.

## Contrat D+1

Pour chaque ligne:

- `dt` = jour de décision;
- `decision_timestamp` = fin du jour D;
- `target_dt = dt + 1`;
- `forecast_horizon_days = 1`;
- `target_demand_qty_d_plus_1` = demande observée du lendemain, sauf futur contrat
  latent explicite.

Tests obligatoires:

- `target_dt = dt + 1`;
- `decision_timestamp < target_dt`;
- split chronologique;
- unicité `dataset_source x dt x location_id x product_id`;
- pas de ligne marquée utilisable si elle est low-quality, fermée ou issue d'un
  zéro dense non prouvé.

## Éligibilité Entraînement

Gold conserve les lignes et expose `usable_for_training_flag`. Le bundle filtre
ensuite `train` et `val`.

Une ligne n'est pas utilisable si:

- `label_quality_score < 0.75`;
- `target_source in ('closed_or_missing_observation', 'dense_calendar_zero_fill')`;
- la source n'est pas autorisée par la registry.

`censor_flag` / `observed_stockout_flag` sont des signaux descriptifs. Dans le
contrat actuel, la cible est la vente observée: une vente sous contrainte reste
donc un label valide si l'observation POS elle-même est complète.

## Contrat Résiduel XGBoost

Le run XGBoost bakery reference courant peut apprendre un résidu autour de la
baseline terrain:

| Champ | Valeur |
| --- | --- |
| cible d'apprentissage | `target_residual_field_blend_lag_1_lag_7_d_plus_1` |
| cible absolue | `target_demand_qty_d_plus_1` |
| transformation | `additive_residual` |
| ancre | `field_baseline_blend_lag_1_lag_7_d_plus_1` |

Gold doit donc conserver à la fois la cible absolue, l'ancre et le résidu.
L'évaluation finale se lit toujours sur la cible absolue reconstruite.

## Split

Le split gold est volontairement asymétrique par source.

### Optimisation Modèle

Les sources d'entraînement autorisées alimentent l'optimisation des modèles
selon leur stratégie de split déclarée dans la slice Gold.

Stratégies actuelles:

- `synthetic_foodservice%`: `pilot_ready_chrono_60_20_20`, soit 60 % train,
  20 % val et 20 % test chronologiques par `dataset_source`;
- `bakery`: `bakery_test_only`, soit holdout final sur les 3 derniers mois;
- `m5_forecasting_accuracy`: surface supplemental `chrono_60_40`, mais modèle
  Gold désactivé dans l'état courant.

Dans le bundle, `val` devient `tuning.parquet`. C'est le split de validation
utilisé par Optuna/HPO.

### Évaluation Finale

`bakery` est réservé au holdout d'évaluation finale. Le modèle ne doit pas voir
Bakery pendant l'optimisation générale.

La stratégie `bakery_test_only` conserve uniquement:

- `split_bucket = 'test'`;
- les lignes Bakery dont `dt` est dans les 3 derniers mois disponibles de
  `silver_bakery_daily_product_demand`;
- une fenêtre configurable via `PRAEDIXA_GOLD_BAKERY_TEST_MONTHS`, avec `3` mois
  par défaut.

Dans le bundle, `test` devient `valid.parquet` et `optimisation_valid.parquet`.
Malgré le nom historique `valid`, ce fichier correspond au holdout final
d'évaluation/test, pas au split de validation HPO.

Le protocole bakery reference XGBoost ajoute ensuite une nuance contrôlée pour
simuler un déploiement réel: une seed bakery pre-test d'un mois peut être
utilisée avant la première prédiction, puis le refit quotidien n'ajoute que les
jours test déjà observés. Cette seed n'est pas une validation HPO et ne doit pas
être confondue avec un accès au futur.

Résumé du mapping:

| Niveau gold | Source | Usage bundle |
| --- | --- | --- |
| `train` | synthetic foodservice actif | `train.parquet`, optimisation fit |
| `val` | synthetic foodservice actif | `tuning.parquet`, validation HPO |
| `test` | `bakery` uniquement | `valid.parquet`, évaluation finale |

Dans le protocole XGBoost bakery reference du 2026-04-30, la validation HPO
reste synthetic foodservice; Bakery n'est pas utilisé comme validation HPO.

## Feature Registry

`platform/warehouse/seeds/feature_registry.csv` documente les features stables:

- famille;
- rôle TFT;
- disponibilité à la prédiction;
- système source;
- politique de missing/default;
- inclusion training;
- tolérance aux constantes.

Ce registre évite de coder les décisions de feature engineering dans plusieurs
endroits incompatibles.
