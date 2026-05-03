# Résultats précédents - modèles foundation

Date de synthèse : 2026-04-27.

> Note 2026-04-30 : le snapshot courant promouvable pour le backend opérationnel
> XGBoost est documenté dans
> `docs/demand_forecast/results_xgboost_2026_04_30.md`. Le présent fichier
> conserve le contexte historique des runs foundation zero-shot.

Ces résultats viennent des artefacts locaux déjà présents dans `var/experiments/demand_forecast/evaluation/`. Ils correspondent aux runs précédents TimesFM, Chronos-2 et Moirai, avant les derniers changements de recalibrage local avec contexte FreshRetail + bakery et avant l'ajout de la progression `tqdm`.

## Protocole

- Données de référence : bakery, protocole `bakery_product_arima_equivalent_from_gold`.
- Split de référence :
  - train : 15 575 lignes
  - validation : 3 325 lignes
  - test : 3 395 lignes
- Fenêtre test : 2022-06-26 -> 2022-09-30.
- Produits testés : 35.
- Baseline économique / terrain utilisée : `blend_lag_1_lag_7_50_50`.
- Baseline MAE globale : `5.954050`.
- Baseline loss économique totale : `21 756.52 EUR`.
- Unité économique utilisée dans les artefacts : coût unitaire baguette `1.0 EUR`, ratio coût production `0.3`.

Important : ces runs sont en mode zero-shot foundation. Les poids des modèles ne sont pas fine-tunés. Le refit quotidien de ces artefacts repose sur le contexte historique walk-forward, pas sur une mise à jour des poids réseau.

## Résumé global

| Modèle | MAE | WAPE | Bias | RMSE | Gain MAE vs baseline | Gain économique vs baseline | Lignes mieux / pire |
|---|---:|---:|---:|---:|---:|---:|---:|
| TimesFM | 6.716537 | 0.315795 | -0.196313 | 18.786298 | -0.762487 | -2 360.77 EUR | 1 747 / 1 648 |
| Chronos-2 | 6.516132 | 0.306373 | 0.037588 | 17.126679 | -0.562081 | -1 495.83 EUR | 1 722 / 1 673 |
| Moirai | 6.507670 | 0.305975 | 0.067969 | 17.607762 | -0.553620 | -1 469.48 EUR | 1 727 / 1 668 |
| Ensemble 1/3 | 6.509129 | 0.306044 | -0.030252 | 17.612347 | -0.555079 | -1 581.47 EUR | 1 742 / 1 653 |

Lecture directe : aucun des trois modèles foundation zero-shot ne bat la baseline terrain sur ce protocole. Moirai est le meilleur des trois en MAE/WAPE, Chronos-2 est très proche et a le RMSE le plus bas. L'ensemble égal réduit le bias mais ne bat pas Moirai.

## Détail par modèle

### TimesFM

- Modèle : `google/timesfm-2.0-500m-pytorch`
- Backend déclaré : `gpu`
- Contexte : `2048`
- Horizon : `128`
- Batch size : `32`
- Zero-shot : oui
- Fine-tuning : non

Métriques :

- MAE : `6.716537`
- WAPE : `0.315795`
- Bias : `-0.196313`
- RMSE : `18.786298`
- sMAPE : `0.510630`
- MAE baseline : `5.954050`
- Gain économique estimé : `-2 360.77 EUR`
- Loss modèle : `24 117.29 EUR`
- Loss baseline : `21 756.52 EUR`

Artefacts principaux :

- `var/experiments/demand_forecast/evaluation/timesfm_test_metrics.json`
- `var/experiments/demand_forecast/evaluation/timesfm_economic_gain_vs_best_baseline.json`
- `var/experiments/demand_forecast/evaluation/timesfm_test_predictions.csv`
- `var/experiments/demand_forecast/evaluation/timesfm_daily_refit_metrics.csv`

### Chronos-2

- Modèle : `amazon/chronos-2`
- Device map déclaré : `cpu`
- Zero-shot : oui
- Fine-tuning : non

Métriques :

- MAE : `6.516132`
- WAPE : `0.306373`
- Bias : `0.037588`
- RMSE : `17.126679`
- sMAPE : `0.509608`
- MAE baseline : `5.954050`
- Gain économique estimé : `-1 495.83 EUR`
- Loss modèle : `23 252.34 EUR`
- Loss baseline : `21 756.52 EUR`

Artefacts principaux :

- `var/experiments/demand_forecast/evaluation/chronos2_test_metrics.json`
- `var/experiments/demand_forecast/evaluation/chronos2_economic_gain_vs_best_baseline.json`
- `var/experiments/demand_forecast/evaluation/chronos2_test_predictions.csv`
- `var/experiments/demand_forecast/evaluation/chronos2_daily_refit_metrics.csv`

### Moirai

- Modèle : `Salesforce/moirai-2.0-R-small`
- Contexte : `1680`
- Batch size : `32`
- Samples : `100`
- Zero-shot : oui
- Fine-tuning : non

Métriques :

- MAE : `6.507670`
- WAPE : `0.305975`
- Bias : `0.067969`
- RMSE : `17.607762`
- sMAPE : `0.513222`
- MAE baseline : `5.954050`
- Gain économique estimé : `-1 469.48 EUR`
- Loss modèle : `23 225.99 EUR`
- Loss baseline : `21 756.52 EUR`

Artefacts principaux :

- `var/experiments/demand_forecast/evaluation/moirai_test_metrics.json`
- `var/experiments/demand_forecast/evaluation/moirai_economic_gain_vs_best_baseline.json`
- `var/experiments/demand_forecast/evaluation/moirai_test_predictions.csv`
- `var/experiments/demand_forecast/evaluation/moirai_daily_refit_metrics.csv`

## Ensemble égal 1/3

Composition :

- TimesFM : `1/3`
- Chronos-2 : `1/3`
- Moirai : `1/3`

Métriques :

- MAE : `6.509129`
- WAPE : `0.306044`
- Bias : `-0.030252`
- RMSE : `17.612347`
- sMAPE : `0.508108`
- MAE baseline : `5.954050`
- Gain économique estimé : `-1 581.47 EUR`
- Loss modèle : `23 337.99 EUR`
- Loss baseline : `21 756.52 EUR`

Artefacts principaux :

- `var/experiments/demand_forecast/evaluation/ensemble_equal_test_metrics.json`
- `var/experiments/demand_forecast/evaluation/ensemble_equal_economic_gain_vs_best_baseline.json`
- `var/experiments/demand_forecast/evaluation/ensemble_equal_test_predictions.csv`

## Conclusion provisoire

Sur ces runs précédents, la baseline `blend_lag_1_lag_7_50_50` reste meilleure que les trois modèles foundation zero-shot et que leur ensemble égal. L'écart économique est négatif pour tous les modèles.

La suite logique n'est donc pas de conclure que les modèles foundation sont inutiles, mais de comparer avec les nouveaux runs qui ajoutent :

- contexte FreshRetail + bakery ;
- recalibrage local sur train + validation ;
- suivi de progression pendant l'initialisation de la calibration ;
- prédiction par lots de séries pour éviter les longues phases silencieuses.

## Comparaison nouvelle run TimesFM - 2026-04-27 15:28

Nouvelle run locale :

- commande : `apps/demand_forecast/run_timesfm_evaluation/main.py`
- protocole : `freshretail_plus_bakery_supervised_reference_train_val`
- train chargé : 2 915 575 lignes
- validation chargée : 1 903 325 lignes
- test : 3 395 lignes
- calibration initialisée sur : 1 900 000 lignes, 50 000 séries
- calibration initiale : slope `1.364181`, intercept `-0.115232`
- calibration finale fold 97 : slope `1.025059`, intercept `0.187595`

Comparaison avec l'ancienne run TimesFM :

| Métrique | Ancienne TimesFM | Nouvelle TimesFM | Delta | Lecture |
|---|---:|---:|---:|---|
| MAE | 6.716537 | 7.149542 | +0.433005 | pire |
| WAPE | 0.315795 | 0.336154 | +0.020359 | pire |
| Bias | -0.196313 | 1.706609 | +1.902922 | passage en sur-prévision |
| RMSE | 18.786298 | 20.863825 | +2.077527 | pire |
| sMAPE | 0.510630 | 0.508589 | -0.002041 | quasi stable |
| Lignes mieux que baseline | 1 747 | 1 691 | -56 | pire |
| Lignes pires que baseline | 1 648 | 1 704 | +56 | pire |
| Gain économique pondéré | -2 360.77 EUR | -724.44 EUR | +1 636.33 EUR | moins mauvais |
| Loss modèle | 24 117.29 EUR | 22 480.96 EUR | -1 636.33 EUR | mieux selon coût |

Lecture :

- La nouvelle run ne bat toujours pas la baseline.
- En précision pure, elle est plus mauvaise que l'ancienne TimesFM.
- Le bias devient fortement positif : la calibration pousse vers la sur-prévision.
- Le gain économique pondéré est moins négatif, mais ce n'est pas un vrai gain prédictif : le modèle fait plus d'erreur globale mais ces erreurs coûtent moins selon la fonction économique actuelle.

Point critique observé dans les logs :

- `Daily evaluation refit starting` voit bien `base_train_rows=2915575`.
- Mais `Daily prediction calibration initialization starting` tombe à `base_train_rows=2900000`.
- Donc les 15 575 lignes bakery train sont retirées par le filtre d'éligibilité avant l'initialisation de calibration.
- La calibration validation utilise `eligible_rows=1900000` sur `valid_rows=1903325`, donc les 3 325 lignes bakery validation ne calibrent pas non plus.

Conclusion technique :

La calibration locale actuelle est dominée par FreshRetail et ne calibre pas vraiment sur bakery. Pour TimesFM zero-shot, comme il n'y a pas de fine-tuning des poids et pas de covariables utilisées (`feature_count=0`), FreshRetail n'apporte pas vraiment de transfert utile au modèle foundation. Il sert surtout à apprendre une calibration globale qui se transfère mal sur bakery.

Prochaine correction à tester :

- calibrer séparément par `dataset_source`, ou au minimum donner priorité à bakery quand bakery existe ;
- ne pas laisser FreshRetail écraser la calibration bakery ;
- comparer 3 variantes TimesFM :
  1. bakery only sans calibration FreshRetail ;
  2. bakery + calibration bakery uniquement ;
  3. FreshRetail + bakery mais calibration pondérée/fallback bakery.

## Correction protocole foundation - 2026-04-27 19:10

Le protocole visé est maintenant explicite pour TimesFM, Moirai et Chronos-2 :

- contexte initial modèle : FreshRetail train + FreshRetail validation uniquement ;
- exclusion des lignes bakery train/validation du contexte modèle ;
- refit quotidien : ajout uniquement de l'historique bakery déjà observé dans le split test ;
- au fold 1, comme `bakery_history_rows=0`, les séries bakery n'existent pas encore dans le contexte des modèles foundation.

Erreur observée sur TimesFM avant correction :

```text
ValueError: TimesFM prediction frame has no series present in model context.
```

Cause :

- TimesFM/Moirai/Chronos-2 prédisent à partir d'un contexte de série.
- Le protocole FreshRetail-only est correct, mais le premier jour du test bakery est un vrai cold-start : aucune série bakery n'a encore d'historique test.

Correction implémentée :

- ajout d'une série technique `__praedixa_foundation_cold_start__` construite depuis FreshRetail train+val ;
- quand une série demandée n'existe pas encore dans le contexte, le backend duplique ce contexte cold-start sous l'identifiant de la série demandée ;
- dès le fold 2, les 35 séries bakery disposent normalement de l'historique du fold 1 et utilisent leur contexte bakery test réel.

Important : ce fallback ne réintroduit pas bakery train/validation dans l'entraînement. Il permet seulement au premier jour test d'être scoré sans fuite.
