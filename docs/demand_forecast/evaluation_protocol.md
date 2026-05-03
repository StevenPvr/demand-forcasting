# Protocole D'Evaluation Demand Forecast

Ce protocole documente l'evaluation bakery reference utilisee pour le snapshot
XGBoost du 2026-04-30.

## Objectif

Mesurer si un modele de prevision D+1 peut ameliorer une decision operationnelle
par rapport a une baseline statistique robuste, sans fuite temporelle.

Deux lectures sont obligatoires:

- precision predictive: MAE, WAPE, RMSE, bias;
- impact economique: cout d'erreur sous loss asymetrique surproduction /
  sous-production.

## Dataset Reference Bakery

Reference:

| Champ | Valeur |
| --- | ---: |
| produits | 35 |
| lignes completes reference | 22 295 |
| lignes train reference | 15 575 |
| lignes validation reference | 3 325 |
| lignes test reference | 3 395 |
| jours test | 97 |
| debut test | 2022-06-26 |
| fin test | 2022-09-30 |

Le split reference est comparable au protocole ARIMA bakery historique:

```text
70% train / 15% validation / 15% test
```

## Protocole Transfer Holdout

Le run XGBoost courant est en mode:

```text
bakery_reference_transfer_holdout
```

Le principe:

1. Les sources synthetic foodservice alimentent le train et la validation.
2. Le dataset bakery complet n'est pas utilise comme train/validation general.
3. Une graine bakery pre-test d'un mois est autorisee avant la premiere
   prediction.
4. Pendant le test, le modele est refitte chaque jour avec uniquement les jours
   bakery deja observes.

Parametres observes dans le run:

| Champ | Valeur |
| --- | ---: |
| train total charge | 612 996 |
| validation | 217 057 |
| test | 3 395 |
| features modele | 78 |
| seed bakery pre-test total | 1 085 lignes |
| seed bakery pre-test eligible | 1 016 lignes |
| seed bakery pre-test jours | 31 |
| debut seed | 2022-05-26 |
| fin seed | 2022-06-25 |

La premiere prediction du 2022-06-26 voit donc 31 jours bakery eligibles avant
test, puis le walk-forward ajoute les jours test une fois observes.

## Baseline

La baseline reference est:

```text
blend_lag_1_lag_7_50_50
```

Elle approxime une regle terrain robuste en combinant le niveau recent et la
saisonnalite hebdomadaire. Elle est aussi l'ancre de reconstruction du contrat
residuel XGBoost courant.

## Contrat Cible

| Champ | Valeur |
| --- | --- |
| `learning_target_col` | `target_residual_field_blend_lag_1_lag_7_d_plus_1` |
| `absolute_target_col` | `target_demand_qty_d_plus_1` |
| `target_transform` | `additive_residual` |
| `reconstruction_anchor_col` | `field_baseline_blend_lag_1_lag_7_d_plus_1` |

Le score final doit toujours etre lu sur la prediction absolue reconstruite,
pas seulement sur le residu.

## Refit Quotidien

Chaque fold correspond a une date test:

```text
fold 1  -> 2022-06-26
...
fold 97 -> 2022-09-30
```

A la date `D`, l'historique autorise inclut:

- train/validation transfer autorises;
- seed bakery pre-test;
- jours bakery test strictement anterieurs a `D`.

Il ne doit pas inclure:

- la cible du jour `D`;
- les jours bakery futurs;
- une feature calculee avec une fenetre qui regarde apres `D`.

## Economie

Le protocole economique compare le cout total modele au cout total baseline.
Sur le snapshot courant:

| Mesure | Valeur |
| --- | ---: |
| loss modele | 18 482.24 EUR |
| loss baseline | 21 756.52 EUR |
| economie estimee | 3 274.27 EUR |

Cette lecture est differente de la MAE. Un modele peut avoir une MAE plus haute
mais une perte economique plus basse si ses erreurs evitent les cas les plus
couteux.

## Checklist Anti-Leakage

Avant de promouvoir un resultat, verifier:

- les bornes train/validation/test en dates absolues;
- le nombre de jours bakery pre-test avant la premiere prediction;
- l'absence de lignes bakery test futures dans le train du fold;
- le sens des colonnes `d_plus_1`, `lag_*`, `rolling_*`;
- les features connues futures vs inconnues;
- la coherence entre `xgboost_target_contract.json` et les colonnes du panel;
- les artefacts `daily_refit_metrics` pour confirmer le walk-forward.
