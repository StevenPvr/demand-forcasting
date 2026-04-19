# Architecture Medaillon Actuelle

Ce document decrit l'architecture en medaillon reellement implemente aujourd'hui dans le repo.

Points importants :

- `raw_landing` JSON POS / ERP / WFM reste une cible d'architecture, pas une brique active du repo
- la `bronze` actuelle reste un schema DuckDB local de substitution alimente depuis `data/`
- la branche exogene ouverte fait maintenant partie du flux reel de la `gold`
- la `gold` n'est plus seulement `gold_base_panel_d1 -> gold_feature_panel_d1 -> gold_daily_product_forecast_panel_d1`
- elle passe maintenant par des slices `gold_feature_*` puis `gold_source_weights_d1`

## Schema visuel detaille

```mermaid
flowchart TB
  classDef storage fill:#f6f8fa,stroke:#57606a,color:#1f2328;
  classDef script fill:#ddf4ff,stroke:#0969da,color:#0a3069;
  classDef current fill:#dafbe1,stroke:#1a7f37,color:#116329;
  classDef future fill:#fff8c5,stroke:#9a6700,color:#7d4e00,stroke-dasharray: 6 4;
  classDef quality fill:#ffebe9,stroke:#cf222e,color:#82071e;
  classDef sidecar fill:#fbefff,stroke:#8250df,color:#552794;

  Raw["Cible future uniquement<br/>raw_landing JSON POS / ERP / WFM<br/>non implemente dans le repo actuel"]:::future

  subgraph Inputs["Inputs coeur reels utilises aujourd'hui"]
    FRTrain["data/bakery_sales/data_train.parquet<br/>FreshRetail train"]:::storage
    FRVal["data/bakery_sales/data_val.parquet<br/>FreshRetail val"]:::storage
    BakeryCSV["data/bakery_sales/Bakery sales.csv<br/>Bakery order lines"]:::storage
    M5Cal["data/m5/calendar.csv"]:::storage
    M5Prices["data/m5/sell_prices.csv"]:::storage
    M5Sales["data/m5/sales_train_validation.csv"]:::storage
  end

  OpenAPIs["Sources ouvertes consultees par la branche exogene<br/>Nager.Date<br/>Open-Meteo<br/>World Bank<br/>FRED"]:::storage
  OpenFetch["scripts/fetch_open_exogenous.py<br/>lit silver.silver_daily_product_demand<br/>calcule les bornes pays / location<br/>telecharge et persiste les fichiers exogenes"]:::script
  OpenFiles["data/external_open/*.csv<br/>location_metadata<br/>public_holidays<br/>school_holidays<br/>weather_daily<br/>macro_annual<br/>macro_timeseries<br/>+ open_exogenous_manifest.json"]:::storage
  OpenBronzeLoad["run_local_gold.py -> load_selected_bronze_specs(...)<br/>charge uniquement les tables bronze_open_*"]:::script

  Config["Configuration runtime<br/>warehouse/profiles.yml<br/>PRAEDIXA_DUCKDB_*<br/>PRAEDIXA_GOLD_BAKERY_TEST_MONTHS<br/>PRAEDIXA_OPEN_EXOGENOUS_DIR"]:::script
  BronzeLoader["scripts/load_bronze_duckdb.py<br/>backup local + chargement DuckDB bronze"]:::script
  SilverRunner["scripts/run_local_silver.py<br/>load bronze + dbt deps + dbt run/test +tag:silver"]:::script
  GoldRunner["scripts/run_local_gold.py<br/>refresh open exogenous par defaut<br/>puis dbt run/test tag:gold"]:::script
  LocalBackup["data/local_backup/*<br/>copie locale des sources + bronze_backup_manifest.json"]:::storage
  CloudDuckDB["DuckDB cloud / MotherDuck<br/>supporte mais desactive par defaut"]:::future

  FRTrain --> BronzeLoader
  FRVal --> BronzeLoader
  BakeryCSV --> BronzeLoader
  M5Cal --> BronzeLoader
  M5Prices --> BronzeLoader
  M5Sales --> BronzeLoader
  BronzeLoader --> LocalBackup
  Config --> SilverRunner
  Config --> GoldRunner
  SilverRunner --> BronzeLoader
  GoldRunner --> OpenFetch
  OpenAPIs --> OpenFetch --> OpenFiles --> OpenBronzeLoad
  GoldRunner --> OpenBronzeLoad
  BronzeLoader -. sync optionnelle .-> CloudDuckDB
  SilverRunner -. target path optionnel .-> CloudDuckDB
  GoldRunner -. target path optionnel .-> CloudDuckDB

  subgraph Warehouse["Warehouse actuel<br/>data/warehouse/praedixa.duckdb<br/>schemas bronze / silver / gold"]

    subgraph Bronze["Schema bronze actuel<br/>substitut local, pas encore une vraie ingestion POS"]
      BronzeCore["bronze_freshretail_daily<br/>bronze_bakery_order_lines<br/>bronze_m5_calendar<br/>bronze_m5_sell_prices<br/>bronze_m5_sales_long"]:::storage
      BronzeOpen["bronze_open_location_metadata<br/>bronze_open_public_holidays<br/>bronze_open_school_holidays<br/>bronze_open_weather_daily<br/>bronze_open_macro_annual<br/>bronze_open_macro_timeseries"]:::storage
    end

    subgraph Staging["dbt staging<br/>views proches source, typage / nettoyage leger"]
      StgCore["stg_freshretail_daily<br/>stg_bakery_order_lines<br/>stg_m5_calendar<br/>stg_m5_sell_prices<br/>stg_m5_sales_long"]:::current
      StgOpen["stg_open_location_metadata<br/>stg_open_public_holidays<br/>stg_open_school_holidays<br/>stg_open_weather_daily<br/>stg_open_macro_annual<br/>stg_open_macro_timeseries"]:::current
    end

    subgraph Silver["dbt silver<br/>verite operationnelle standardisee + exogenes prepares"]
      SilverCore["silver_freshretail_native<br/>silver_freshretail_daily_product_demand<br/>silver_bakery_order_lines<br/>silver_bakery_daily_product_demand<br/>silver_m5_native_daily<br/>silver_m5_daily_product_demand"]:::current
      SilverUnion["silver_daily_product_demand<br/>union canonique FreshRetail + Bakery + M5<br/>grain: dt x location_id x product_id"]:::current
      SilverOpen["silver_open_location_metadata<br/>silver_open_public_holiday_calendar_daily<br/>silver_open_school_holidays_daily<br/>silver_open_weather_daily<br/>silver_open_macro_country_daily"]:::current
      SilverQuality["silver_quality_incidents<br/>vue de controle qualite"]:::quality
      SilverManifest["silver_run_manifest<br/>vue de manifest par dataset"]:::quality
    end

    subgraph Gold["dbt gold<br/>dataset model-ready D+1"]
      GoldBase["gold_base_panel_d1<br/>densification journaliere<br/>join demande + metadata + holidays + school + weather + macro<br/>target_dt = dt + 1"]:::current
      GoldSlices["gold_feature_bakery_d1<br/>gold_feature_freshretail_d1<br/>gold_feature_m5_ca_d1<br/>gold_feature_m5_tx_d1<br/>gold_feature_m5_wi_d1<br/>via macro praedixa_gold_feature_slice"]:::current
      GoldFeaturePanel["gold_feature_panel_d1<br/>union all des slices"]:::current
      GoldWeights["gold_source_weights_d1<br/>sample_weight_source par split et dataset"]:::current
      GoldPanel["gold_daily_product_forecast_panel_d1<br/>panel final forecast D+1<br/>features courantes + futures<br/>weather + macro + sample weights"]:::current
      GoldManifest["gold_run_manifest<br/>vue de manifest par split et dataset"]:::quality
    end
  end

  BronzeLoader --> BronzeCore
  OpenBronzeLoad --> BronzeOpen

  BronzeCore --> StgCore --> SilverCore --> SilverUnion
  BronzeOpen --> StgOpen --> SilverOpen

  SilverUnion --> SilverQuality
  SilverUnion --> SilverManifest
  SilverUnion --> OpenFetch
  SilverUnion --> GoldBase
  SilverOpen --> GoldBase
  GoldBase --> GoldSlices --> GoldFeaturePanel
  GoldFeaturePanel --> GoldPanel
  GoldWeights --> GoldPanel
  GoldPanel --> GoldManifest

  SilverSchema["dbt tests silver/schema.yml<br/>not_null<br/>accepted_values<br/>unique_combination_of_columns"]:::quality
  GoldSchema["dbt tests gold/schema.yml<br/>split_bucket / target / not_null / uniqueness"]:::quality
  GoldCustom["warehouse/tests/*.sql<br/>bakery uniquement en test<br/>bakery hors train/val<br/>test bakery sur 3 mois<br/>target_dt = jour suivant"]:::quality

  SilverRunner --> SilverSchema
  GoldRunner --> GoldSchema
  GoldRunner --> GoldCustom
  SilverRunner --> SilverUnion
  GoldRunner --> GoldPanel

  subgraph Sidecar["Pipelines paralleles presents dans le repo<br/>mais hors colonne vertebrale dbt du medaillon"]
    GlobalDataset["src/research_praedixa/global_dataset/*<br/>standardisation Parquet canonique"]:::sidecar
    GlobalArtifacts["data/global_dataset/*.parquet<br/>global_daily_demand_manifest.json"]:::sidecar
    ResearchPipelines["feature_engineering / prediction / evaluation / optimisation<br/>experimentation ML Python"]:::sidecar
  end

  FRTrain -. reutilise aussi .-> GlobalDataset
  FRVal -. reutilise aussi .-> GlobalDataset
  BakeryCSV -. reutilise aussi .-> GlobalDataset
  M5Cal -. reutilise aussi .-> GlobalDataset
  M5Prices -. reutilise aussi .-> GlobalDataset
  M5Sales -. reutilise aussi .-> GlobalDataset
  GlobalDataset --> GlobalArtifacts --> ResearchPipelines
```

## Lecture rapide du schema

- Le flux medaillon actuel commence toujours dans `data/`, pas dans une vraie couche `raw_landing`.
- `scripts/load_bronze_duckdb.py` alimente le coeur `bronze` DuckDB de substitution et cree aussi un backup local.
- `scripts/run_local_gold.py` ajoute maintenant une vraie branche exogene : refresh des open data, ecriture dans `data/external_open/`, puis chargement des `bronze_open_*`.
- La couche `silver` ne contient plus seulement la demande canonique ; elle prepare aussi les tables exogenes consolidees par location, pays et date.
- `gold_base_panel_d1` joint desormais la demande dense avec metadata location, calendrier ferie, vacances scolaires, meteo et macro.
- La couche features `gold` est maintenant decoupee en slices par scope de split :
  - `bakery`
  - `freshretail`
  - `m5_ca`
  - `m5_tx`
  - `m5_wi`
- `gold_source_weights_d1` calcule les poids de source avant la publication du panel final `gold_daily_product_forecast_panel_d1`.
- Le dossier `src/research_praedixa/global_dataset/` existe toujours en parallele, mais ce n'est pas la colonne vertebrale du warehouse `dbt + DuckDB`.

## Frontieres importantes

- `bronze` actuelle :
  zone technique locale de substitution, pas encore une ingestion POS de production
- `silver` :
  verite operationnelle standardisee + preparation des exogenes exploitables
- `gold` :
  panel model-ready pour la prevision de demande `D+1`, enrichi par signaux ouverts et decoupe par slices de split
- `raw_landing` :
  cible future explicite mais non implementee
