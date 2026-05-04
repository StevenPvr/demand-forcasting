# Documentation Praedixa Research

Cette documentation decrit l'etat actuel du repo `research_praedixa` pour le
wedge Praedixa de prevision de demande sur operations alimentaires perissables.

Le repo doit se lire dans cet ordre:

1. [Architecture demand forecast](demand_forecast/architecture.md)
2. [Protocole d'evaluation](demand_forecast/evaluation_protocol.md)
3. [Resultats XGBoost du 2026-04-30](demand_forecast/results_xgboost_2026_04_30.md)
4. [Runbook local](demand_forecast/runbook.md)
5. [Architecture medaillon](data_engineering/medaillon/architecture_actuelle.md)
6. [Couche Gold](data_engineering/medaillon/gold.md)
7. [Simulateur restaurant / foodservice](data_engineering/restaurant_simulator/)

## Etat Court

- Source de verite data: `gold.gold_training_matrix_d1` dans DuckDB.
- Backend courant le plus exploitable: `xgboost`.
- Backend cible long terme: TFT, encore en durcissement.
- Cible actuelle: `target_demand_qty_d_plus_1`, avec apprentissage possible sur
  residu additive quand le contrat le demande.
- Protocole de reference courant: bakery holdout comparable ARIMA, avec test
  sur 35 produits et 97 jours.

## Snapshot Resultat Courant

Le run XGBoost du 2026-04-30 donne:

- test: 3 395 lignes, 35 produits, du 2022-06-26 au 2022-09-30;
- MAE modele: `6.191286637136248`;
- MAE baseline statistique robuste: `5.954050073637703`;
- economie estimee vs baseline sous loss asymetrique: `3 274.27 EUR`.

Lecture correcte: le modele ne bat pas la baseline en MAE globale, mais il
ameliore la decision economique sous la fonction de cout actuelle.

## Sources De Verite

Les chiffres de resultats documentes viennent des artefacts locaux:

- `var/experiments/demand_forecast/evaluation/xgboost_test_metrics.json`
- `var/experiments/demand_forecast/evaluation/xgboost_economic_gain_vs_best_baseline.json`
- `var/experiments/demand_forecast/evaluation/xgboost_target_contract.json`

Pour les euros, la source de verite est
`xgboost_economic_gain_vs_best_baseline.json`; `xgboost_test_metrics.json`
porte surtout les metriques globales et la comparaison MAE.

Les docs historiques restent utiles pour le contexte, mais les anciens chemins
parquet ne doivent pas etre traites comme la source de verite d'entrainement.
