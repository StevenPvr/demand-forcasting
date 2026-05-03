# Resultats XGBoost - 2026-04-30

Snapshot local du run XGBoost bakery reference execute le 2026-04-30.

Commande principale observee:

```bash
/Users/steven/Programmation/research_praedixa/.venv/bin/python \
  /Users/steven/Programmation/research_praedixa/products/demand_forecast/src/praedixa/demand_forecast/evaluation/xgboost/main.py
```

Artefacts sources:

- `var/experiments/demand_forecast/evaluation/xgboost_test_metrics.json`
- `var/experiments/demand_forecast/evaluation/xgboost_economic_gain_vs_best_baseline.json`
- `var/experiments/demand_forecast/evaluation/xgboost_target_contract.json`

Attention: `xgboost_test_metrics.json` contient aussi une lecture business
simplifiée alignée MAE / arrondi, qui peut afficher `0.0 EUR` de gain. La
lecture économique réaliste utilisée ici est celle de
`xgboost_economic_gain_vs_best_baseline.json`.

## Resume Executif

Le modele XGBoost ne bat pas la baseline statistique en MAE globale. En revanche,
il bat la baseline sous la fonction de cout economique actuelle:

```text
gain economique estime: 3 274.27 EUR sur 97 jours test
```

Lecture correcte: c'est un resultat fort pour la logique ROI, mais il faut le
presenter comme un gain decisionnel sous loss asymetrique, pas comme une victoire
predictive pure.

## Protocole

| Champ | Valeur |
| --- | --- |
| mode | `bakery_reference_transfer_holdout` |
| protocole reference | `bakery_product_arima_equivalent_from_gold` |
| backend | `xgboost` |
| test | 3 395 lignes |
| produits | 35 |
| jours test | 97 |
| fenetre test | 2022-06-26 -> 2022-09-30 |
| train charge | 612 996 lignes |
| validation chargee | 217 057 lignes |
| features | 78 |
| seed bakery pre-test | 31 jours, 1 016 lignes eligibles |

## Contrat Cible

| Champ | Valeur |
| --- | --- |
| cible d'apprentissage | `target_residual_field_blend_lag_1_lag_7_d_plus_1` |
| cible absolue | `target_demand_qty_d_plus_1` |
| transformation | `additive_residual` |
| ancre | `field_baseline_blend_lag_1_lag_7_d_plus_1` |

## Metriques Globales

| Mesure | XGBoost | Baseline `blend_lag_1_lag_7_50_50` | Lecture |
| --- | ---: | ---: | --- |
| MAE | 6.191286637 | 5.954050074 | baseline meilleure |
| delta MAE | -0.237236563 | - | modele moins bon |
| relative MAE improvement | -3.984456976 % | - | negatif |
| WAPE | 0.291099452 | n/a | mesure modele |
| RMSE | 15.517708826 | n/a | mesure modele |
| Bias | 2.185426104 | n/a | sur-prevision moyenne |
| lignes mieux que baseline | 1 529 | - | modele meilleur ligne par ligne |
| lignes egales | 238 | - | egalite |
| lignes pires que baseline | 1 628 | - | baseline meilleure ligne par ligne |

## Impact Economique

Source: `xgboost_economic_gain_vs_best_baseline.json`.

| Mesure | Valeur |
| --- | ---: |
| loss modele | 18 482.24 EUR |
| loss baseline | 21 756.52 EUR |
| economie estimee vs baseline | 3 274.27 EUR |
| total absolute MAE saved vs baseline | -18.519388193 |
| produits couverts | 35 |

Interpretation: le modele fait plus d'erreur absolue moyenne, mais ses erreurs
sont moins couteuses selon la fonction economique actuelle. C'est exactement la
difference Praedixa entre "predire un chiffre" et "ameliorer une decision".

## Produits Qui Portent Le Gain

Top produits par economie estimee:

| Produit | MAE modele | MAE baseline | delta MAE | gain estime |
| --- | ---: | ---: | ---: | ---: |
| TRADITIONAL BAGUETTE | 53.703530 | 52.500000 | -1.203530 | 610.41 EUR |
| FORMULE SANDWICH | 4.351313 | 3.907216 | -0.444096 | 456.75 EUR |
| TARTELETTE | 4.863058 | 4.856752 | -0.006306 | 185.25 EUR |
| SAND JB EMMENTAL | 2.411391 | 2.005446 | -0.405945 | 181.80 EUR |
| ECLAIR | 4.155552 | 3.191110 | -0.964442 | 179.53 EUR |
| SANDWICH COMPLET | 2.861855 | 2.814433 | -0.047422 | 177.50 EUR |
| FINANCIER X5 | 2.267057 | 1.628866 | -0.638191 | 134.66 EUR |
| BAGUETTE | 10.801133 | 9.932990 | -0.868143 | 124.95 EUR |

Point important: ces lignes montrent bien que le gain economique n'est pas une
simple consequence d'une meilleure MAE produit. La loss penalise differemment
surproduction et sous-production.

## Ce Qui Est Solide

- Le test est chronologique et out-of-sample sur la fenetre bakery reference.
- Le protocole daily refit simule une evaluation operationnelle jour par jour.
- Le modele ne dispose que d'un mois bakery pre-test avant la premiere
  prediction.
- Les artefacts conservent le contrat cible, les metriques et la lecture
  economique.
- Le resultat business est compare a une baseline terrain non triviale.

## Limites

- La MAE globale reste moins bonne que la baseline.
- Le gain depend fortement de la fonction de cout economique actuelle.
- La seed bakery pre-test de 31 jours doit etre documentee dans toute
  communication externe.
- Les resultats portent sur bakery et 35 produits; ils ne prouvent pas encore
  une generalisation multi-enseignes en production.
- La cible reste une vente observee / demande observee selon la qualite des
  signaux disponibles, pas une demande latente reconstruite universelle.

## Prochaines Ablations

1. XGBoost sans seed bakery pre-test.
2. XGBoost avec seed 7 jours, 14 jours, 31 jours.
3. XGBoost optimise sur MAE vs XGBoost optimise sur loss economique.
4. Comparaison par familles produit a forte marge vs faible marge.
5. Test de sensibilite sur les couts de surproduction et rupture.
6. Backtest identique avec TFT quand le backend est promouvable.
