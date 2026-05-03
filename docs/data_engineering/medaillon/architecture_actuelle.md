# Architecture Médaillon Actuelle

Le médaillon actif est local, DuckDB/dbt-first, et orienté prévision D+1. La
source de vérité aval est `gold.gold_model_training_panel_d1`, pas les anciens
parquets intermédiaires.

```mermaid
flowchart TB
  Sources["var/sources/*<br/>Bakery, supplemental corpus, synthetic foodservice"] --> BronzeCore
  BronzeCore["load_core_bronze<br/>bronze_bakery_order_lines<br/>bronze_supplemental_corpus_daily<br/>bronze_synthetic_foodservice_*<br/>bronze_source_manifest"] --> SilverCore
  SilverCore["run_silver<br/>silver_daily_product_demand<br/>source registry<br/>feature registry"] --> OpenFetch
  OpenFetch["refresh_open_exogenous<br/>bornes depuis silver<br/>CSV open data"] --> BronzeOpen
  BronzeOpen["bronze_open_*"] --> SilverExog
  SilverExog["silver_open_*<br/>providers allowlist uniquement"] --> Gold
  SilverCore --> Gold
  Gold["run_gold<br/>gold_base_panel_d1<br/>gold_feature_bakery_d1<br/>gold_feature_synthetic_foodservice_d1<br/>gold_model_training_panel_d1"] --> Bundle
  Bundle["training_bundle<br/>train/tuning filtrés<br/>contrat cible + feature manifest"] --> Evaluation
  Evaluation["evaluation bakery reference<br/>daily refit walk-forward<br/>metrics + economic gain"]
```

## Responsabilités

- `raw_landing`: cible de production append-only, désormais représentée par le
  manifest bronze local. Une vraie ingestion POS/ERP/WFM reste une évolution.
- `bronze`: projection locale typée des sources, avec `source_name`,
  `source_policy_id`, hash, row count et manifest persistant.
- `staging`: cast et normalisation minimale, sans règle métier lourde.
- `silver`: contrat opérationnel non agrégé au grain
  `dataset_source x dt x location_id x product_id`, plus tables de gouvernance
  et exogènes autorisées. Si deux flux autorisés alimentent la même clé métier,
  Silver garde une seule ligne via une priorité déterministe: bronze direct puis
  corpus supplemental.
- `gold`: dataset ML D+1, avec labels alignés dans le futur, disponibilité
  temporelle explicite et contrat consommable par le bundle. Le synthetic
  foodservice actif suit `pilot_ready_chrono_60_20_20`; `bakery` suit
  `bakery_test_only` sur les 3 derniers mois. Les runs de référence peuvent
  matérialiser des cibles résiduelles autour d'une baseline terrain, mais
  l'évaluation se lit sur la cible absolue reconstruite.
- `training_bundle`: filtre les lignes train/tuning non éligibles et conserve un
  rapport d'exclusion. Le `val` gold devient `tuning.parquet`; le `test` gold
  devient `valid.parquet`, utilisé comme holdout final d'évaluation.
- `evaluation`: compare le modèle à une baseline statistique robuste, avec une
  lecture prédictive et une lecture économique séparées.

## Points Critiques

- Les providers open-data doivent être présents dans l'allowlist
  `silver_allowed_provider_sources`; quarantaine et sources inconnues sont
  exclues avant silver exogène.
- `decision_timestamp` représente la fin du jour D; `target_dt = dt + 1`.
- `target_semantics = observed_sales` tant que la demande latente n'est pas
  reconstruite explicitement.
- Les zéros densifiés et jours fermés/missing ne sont pas utilisables pour
  l'entraînement.
- Le refit quotidien d'évaluation ne doit ajouter que l'historique déjà observé
  avant la date prédite.
- `gold_feature_supplemental_corpus_d1` existe comme surface désactivée; elle
  ne fait pas partie de l'union active `gold_feature_panel_d1` dans l'état
  courant.
