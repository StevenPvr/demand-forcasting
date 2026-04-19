# Couche Gold Praedixa

## Role

La couche `gold` de Praedixa a pour objectif de produire le dataset final exploitable par le modele de prevision.

Elle doit transformer la `silver` en un panel journalier :

- unifie
- dense
- temporalement safe
- exploitable par un modele tabulaire type `XGBoost`
- structure pour l'entrainement, l'optimisation et l'evaluation

La `gold` n'est pas une couche de reporting.
Elle est la couche `model-ready` du wedge actuel Praedixa :

- prevision de la demande
- horizon `D+1`

## Objectif V1

L'objectif de la `gold` V1 est de tester si un modele entraine hors `bakery` peut generaliser sur `bakery`.

Le cadre d'evaluation est donc :

- `train` sur `M5 + FreshRetail`
- `val` sur `M5 + FreshRetail`
- `test` sur `bakery` uniquement

Autrement dit :

- `bakery` ne doit apparaitre ni dans `train`, ni dans `val`
- `test` doit servir de vrai test de generalisation zero-shot sur le wedge `bakery`

## Position dans l'architecture medaillon

- `raw_landing`
  - JSON bruts et immutables
- `bronze`
  - projection tabulaire proche source
- `silver`
  - verite operationnelle nettoyee et standardisee
  - demande observee au grain journalier
- `gold`
  - panel final `D+1`
  - split `train / val / test`
  - features temporelles et exogenes leak-safe
  - poids d'entrainement

## Sources exogenes V1

La `gold` V1 n'utilise que des sources exogenes open-source publiques.

Sources retenues :

- jours feries publics : `Nager.Date`
- vacances scolaires France : calendrier ICS open data ministeriel
- vacances scolaires `M5` : proxy district/city historical calendars pour `Sacramento`, `Austin`, `Madison`
- meteo historique : `Open-Meteo Archive API`
- macro mensuelle / trimestrielle `as-of` : `FRED` public CSV graph endpoint
- macro annuelle lente : `World Bank Indicators API`

Implications :

- aucune cle API payante n'est necessaire en V1
- le cloud reste supporte pour le warehouse, mais les sources exogenes restent ouvertes
- les hypotheses de geolocalisation doivent etre explicites et auditables

Important :

- les vacances scolaires `M5` sont un signal proxy de niveau district, pas une verite magasin par magasin
- `FreshRetail` reste sans vacances scolaires tant que la doc publique ne donne pas de mapping fiable `city_id -> city_name`

## Table publiee

La V1 publie une table gold canonique unique :

- `gold.gold_daily_product_forecast_panel_d1`

Le suffixe `d1` rappelle que :

- l'horizon predit est `D+1`
- la target de la ligne datee `t` est la demande observee a `t+1`

Implementation possible :

- des etapes intermediaires type `gold_base_panel_d1` ou `gold_feature_panel_d1` peuvent exister dans le pipeline
- mais le contrat externe V1 reste une seule table gold canonique

## Grain canonique

Le grain canonique est :

- `dt x location_id x product_id`

Chaque ligne represente l'etat de connaissance a la date `dt` pour predire la demande du lendemain.

Le dataset doit rester :

- sans multi-index cache
- avec IDs visibles comme colonnes
- avec unicite verifiee sur :
  - `dataset_source`
  - `dt`
  - `location_id`
  - `product_id`

## Cadence et densification

La `gold` V1 est :

- `daily`
- `dense`

Cela signifie :

- pour chaque serie `location_id x product_id`, on construit un calendrier journalier complet sur la fenetre retenue
- les jours absents ne sont pas ignores
- ils sont representes explicitement

Mais la densification ne doit jamais inventer de faux zeros.

Regles :

- `0` uniquement si la semantique metier du jour est suffisamment fiable
- `NULL` si la target ou une feature n'est pas legitime a renseigner
- les flags doivent distinguer :
  - `true_zero_demand`
  - `location_closed`
  - `missing_source_data`
  - `partial_day_data`
  - `observed_stockout`
  - `product_inactive`

Pour un modele type `XGBoost`, cette densification est plus propre qu'un panel sparse, car elle rend les :

- lags
- rollings
- saisonnalites hebdomadaires

beaucoup plus coherents.

## Regle de target

La target gold de reference est :

- `target_demand_qty_d_plus_1`

Definition :

- pour une ligne datee `t`
- `target_demand_qty_d_plus_1 = observed_demand_qty(t+1)`

Regles obligatoires :

- la target doit etre construite par un `shift(-1)` logique a l'interieur de chaque serie
- le shift doit respecter le calendrier journalier dense de la serie
- la target doit etre `NULL` si `t+1` n'est pas exploitable
- aucune forward-fill de la target
- aucune fuite d'information depuis `t+1` dans les features calculees a `t`

## Split V1

### Principe general

Le split est :

- chronologique
- par dataset
- sans melange aleatoire

### Bakery

`bakery` est reserve au test uniquement.

Regle :

- `test` = les `3 derniers mois` du dataset `bakery`
- aucun enregistrement `bakery` dans `train`
- aucun enregistrement `bakery` dans `val`

### M5

Regle :

- `train` = premiers `60%` chronologiques
- `val` = derniers `40%` chronologiques
- pas de `test`

### FreshRetail

Regle :

- `train` = premiers `60%` chronologiques
- `val` = derniers `40%` chronologiques
- pas de `test`

### Colonne de split

La table gold doit contenir :

- `split_bucket`

Valeurs autorisees :

- `train`
- `val`
- `test`

## Client hypothesis and scope columns

Toutes les lignes doivent partager un schema commun.

Quand une information n'existe pas pour un dataset :

- on conserve la colonne
- on met `NULL`
- on ajoute un flag `*_available` si cela aide le modele ou l'audit

Colonnes de contexte obligatoires :

- `dataset_source`
- `client_id`
- `vertical_level_1`
- `vertical_level_2`
- `country_code`
- `region_code`
- `city_name`
- `series_id`
- `location_id`
- `product_id`
- `category_level_1`
- `category_level_2`
- `category_level_3`

Remarques :

- `client_id` peut etre synthetique en V1 pour les datasets publics
- `vertical_level_1` doit etre harmonise sur un vocabulaire court et stable
- exemples :
  - `retail`
  - `restaurant`
  - `fast_food`
  - `bakery`
  - `coffee_shop`

## Feature families autorisees en gold

La `gold` peut contenir des features modeles.
Contrairement a la `silver`, c'est son role.

Mais ces features doivent rester :

- temporalement disponibles a la date de prediction
- compatibles avec un horizon `D+1`

### 1. Statut et qualite de ligne

- `location_open_flag`
- `product_active_flag`
- `day_complete_flag`
- `missing_sales_flag`
- `observed_stockout_flag`
- `observed_stockout_available`
- `anomaly_flag`
- `freshretail_rescaled_flag`
- `target_scale_assumption`

Note :

- `freshretail_rescaled_flag` est maintenant `false` dans la pipeline SQL `dbt`
- `target_scale_assumption` vaut `native_observed`

### 2. Calendrier

- `dt`
- `day_of_week`
- `day_of_month`
- `week_of_year`
- `month`
- `quarter`
- `year`
- `weekend_flag`
- `holiday_flag`
- `holiday_name`
- `school_holiday_flag`
- `bridge_day_flag`
- `pre_holiday_flag`
- `post_holiday_flag`
- `month_end_flag`

### 3. Saisonnalite encodee

- `sin_day_of_week`
- `cos_day_of_week`
- `sin_week_of_year`
- `cos_week_of_year`
- `sin_month`
- `cos_month`

### 4. Prix et promo

- `avg_selling_price`
- `observed_discount_amount`
- `discount_pct`
- `promo_flag`
- `price_available`

### 5. Meteo

Meteo autorisee uniquement si elle est disponible a la date de decision ou si une version previsionnelle existe reellement.

Colonnes prioritaires :

- `temperature_mean`
- `temperature_min`
- `temperature_max`
- `precipitation_mm`
- `humidity_mean`
- `wind_speed_mean`
- `weather_available`

### 6. Macro et signaux economiques

La `gold` peut embarquer des signaux macro, mais seulement ceux qui ont une plausibilite operationnelle et une disponibilite realiste.

Priorites V1 :

- `inflation_cpi`
- `food_cpi`
- `policy_rate`
- `unemployment_rate`
- `consumer_confidence`
- `retail_sales_index`
- `macro_available`

Points importants :

- ces variables vivent souvent a une frequence mensuelle ou hebdomadaire
- elles doivent etre jointes avec une logique de disponibilite reelle
- il faut utiliser la derniere valeur connue a la date `t`
- jamais la valeur revisee ou publiee plus tard

Variables a faible priorite en V1 :

- `gdp`
- `public_debt`
- autres variables macro trop lentes, trop agregees ou trop lointaines par rapport a la demande quotidienne

### 7. Historique leak-safe

La `gold` doit contenir des features temporelles une fois la densification faite.

Priorite V1 :

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`
- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_mean_28`
- `rolling_std_28`
- `ewm_mean_7`
- `ewm_mean_28`
- `same_dow_mean_4w`

Regles :

- les rollings sont strictement one-sided
- aucune fenetre ne doit inclure `t+1`
- si une feature utilise `t`, cela doit etre explicite et legitime pour predire `t+1`

### 8. Baselines statistiques en features

La `gold` peut embarquer des signaux issus de baselines statistiques simples.

Autorise en V1 :

- `naive_last_value`
- `seasonal_naive_d7`
- `moving_average_7`
- `moving_average_28`
- `ewm_forecast_like`

Non prioritaire en V1 :

- `ARIMA`
- `ETS`
- `Holt-Winters`
- autres modeles serie-par-serie couteux

Motif :

- trop lourds
- plus fragiles
- peu utiles pour un premier test de generalisation tabulaire `D+1`

## Features non autorisees en V1

Ne doivent pas entrer dans la gold V1 :

- toute feature necessitant une information indisponible a `t`
- toute meteo future non reellement forecastable dans le pipeline
- toute variable publiee apres la date de prediction mais jointe comme si elle etait connue
- toute imputation silencieuse de la target
- toute target normalisee pour certains datasets et brute pour d'autres

## Ponderation d'entrainement

Le dataset `M5` est beaucoup plus volumineux que `FreshRetail`, et surtout que `bakery`.

La `gold` doit donc porter des colonnes de ponderation afin d'eviter qu'un modele global n'apprenne quasi exclusivement `M5`.

Colonnes recommandees :

- `sample_weight_source`
- `sample_weight_business`
- `is_primary_eval_dataset`

Usage recommande :

- `sample_weight_source`
  - corrige l'asymetrie de volume entre datasets
- `sample_weight_business`
  - permet d'augmenter le poids des observations les plus proches du wedge ou du cout business
- `is_primary_eval_dataset`
  - identifie les observations `bakery` du test final

## Regles anti-leakage obligatoires

### Prediction timestamp

Le timestamp de prediction doit etre pense comme :

- fin de journee `t`
- prediction de la demande a `t+1`

### Regles obligatoires

- split chronologique uniquement
- aucun `bakery` dans `train` ni `val`
- aucun fit de scaler ou encodeur sur `val` ou `test`
- aucune fenetre de rolling incluant le present futur illegitime
- aucune knowledge promo future non reellement disponible
- aucune meteo future observee si on ne dispose pas d'une vraie prevision meteo
- aucun stockout interprete comme faible demande sans flag dedie

### Donnees macro

Les series macro doivent etre jointes avec :

- date de publication
- date de disponibilite effective

si l'on veut etre rigoureux.

Si cette information n'est pas encore disponible en V1 :

- il faut le documenter comme hypothese
- et rester conservateur sur les signaux macro embarques

Etat actuel :

- la `gold` V1 utilise une logique `as-of` avec lag de publication conservateur
- ce n'est pas encore une vraie base `vintage` avec revisions historisees

## Difference silver vs gold

### Silver

- verite operationnelle standardisee
- pas de features modeles finales
- demande observee journaliere

### Gold

- panel dense
- split explicite
- target `D+1`
- exogenes jointees
- lags et rollings
- baselines statistiques simples comme features
- poids d'entrainement

## Contrat de colonnes minimales

La table `gold.gold_daily_product_forecast_panel_d1` doit contenir au minimum :

- `dataset_source`
- `client_id`
- `vertical_level_1`
- `vertical_level_2`
- `country_code`
- `region_code`
- `city_name`
- `series_id`
- `location_id`
- `product_id`
- `category_level_1`
- `category_level_2`
- `category_level_3`
- `dt`
- `split_bucket`
- `target_demand_qty_d_plus_1`
- `location_open_flag`
- `product_active_flag`
- `day_complete_flag`
- `missing_sales_flag`
- `observed_stockout_flag`
- `observed_stockout_available`
- `anomaly_flag`
- `holiday_flag`
- `school_holiday_flag`
- `avg_selling_price`
- `promo_flag`
- `price_available`
- `weather_available`
- `macro_available`
- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`
- `rolling_mean_7`
- `rolling_mean_28`
- `same_dow_mean_4w`
- `seasonal_naive_d7`
- `sample_weight_source`
- `sample_weight_business`
- `is_primary_eval_dataset`
- `gold_run_id`

## Checks qualite obligatoires

### Checks de contrat

- unicite sur le grain canonique
- presence de `split_bucket`
- absence de `bakery` dans `train`
- absence de `bakery` dans `val`
- presence exclusive de `bakery` dans `test`
- target `D+1` bien construite

### Checks de densification

- calendrier journalier complet par serie
- pas de trou non explique
- flags distinguent bien :
  - zero reel
  - fermeture
  - manque de donnees
  - jour partiel
  - stockout

### Checks anti-leakage

- lags et rollings alignes
- aucune feature calculee avec `t+1`
- aucune jointure macro ou meteo futuriste

## Strategie de publication

Format recommande :

- table principale dans `DuckDB`
- export `Parquet` pour experimentation et training
- manifest `JSON` ou table dediee pour la tracabilite

Artifacts minimums :

- `gold.gold_daily_product_forecast_panel_d1`
- `gold.gold_run_manifest`

## Resume de la philosophie gold

La `gold` V1 doit etre pensee comme :

- un panel de forecasting dense
- unifie
- D+1
- zero-shot `bakery`
- structure pour `XGBoost`
- riche en signaux exogenes plausibles
- stricte sur le leakage

L'obsession n'est pas d'empiler un maximum de variables.
L'obsession est de construire un dataset :

- comparable entre sources
- economiquement plausible
- modelisable proprement
- defendable scientifiquement
