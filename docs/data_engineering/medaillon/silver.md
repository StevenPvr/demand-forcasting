# Couche Silver Praedixa

## Role

La couche `silver` de Praedixa a pour objectif de transformer des donnees operationnelles heterogenes en tables tabulaires standardisees, auditables et fiables, sans encore construire les features finales du modele.

La `silver` n'est pas la couche "dataset ML pret a entrainer".
Elle est la couche de verite operationnelle nettoyee sur laquelle la `gold` pourra ensuite :

- joindre les dimensions et signaux utiles
- construire les covariables modeles
- produire les datasets de training et de scoring

## Position dans l'architecture medaillon

- `raw_landing`
  - payloads JSON bruts des POS, ERP, WFM et autres sources
  - conservation immutable pour audit, replay et debug
  - aucun nettoyage metier
- `bronze`
  - projection tabulaire proche source a partir des JSON bruts
  - flatten des structures imbriquees
  - standardisation minimale des types techniques
  - fidelite maximale a la source
- `silver`
  - lecture des tables bronze tabulaires
  - normalisation tabulaire
  - standardisation des schemas
  - nettoyage non destructif
  - qualification des anomalies
  - consolidation au grain metier et au grain journalier
- `gold`
  - jeux de donnees orientes use case
  - panel de forecasting
  - jointures de dimensions et signaux
  - feature engineering leak-safe
  - datasets de training, validation, backtest et inference

## Principe directeur

La `silver` doit rendre les donnees exploitables sans inventer de l'information.

Elle part de tables `bronze` deja tabulaires.
Elle ne doit pas supporter le cout principal du parsing JSON brut.

Elle doit :

- standardiser
- typer
- dedoublonner
- tracer
- signaler les problemes

Elle ne doit pas :

- masquer les donnees manquantes
- forward-fill la target
- confondre fermeture, rupture, manque de donnees et vrai zero
- supprimer silencieusement les outliers
- melanger nettoyage metier et feature engineering modele

## Canonical grain

La couche `silver` doit supporter deux niveaux de grain.

Ces grains sont calcules a partir de la `bronze` tabulaire et non a partir des payloads JSON bruts.

### 1. Grain transactionnel normalise

Pour les sources transactionnelles deja projetees en bronze tabulaire :

- `order_id`
- `order_line_id`
- `location_id`
- `product_id`
- `sold_at`

Cette couche permet de :

- corriger les problemes de timezone
- gerer refunds, voids, annulations
- dedoublonner des tickets
- qualifier les problemes avant agregation

### 2. Grain journalier de demande observee

Table centrale pour la prevision :

- `dt x location_id x product_id`

C'est le grain de reference pour construire ensuite la `gold demand forecasting`.

## Frontiere stricte entre silver et gold

### Ce qui vit en silver

- faits metier standardises
- dimensions conformees
- dimensions calendrier / meteo / promo standardisees
- flags de qualite et d'incident
- demande observee journaliere

### Ce qui vit en gold

- jointure finale du panel de modelling
- encodages et transformations modele
- lags et rollings
- interactions
- features derivees calendrier leak-safe
- features derivees meteo leak-safe
- features de segmentation et d'historique

Regle simple :

- `silver` contient les donnees calendrier et meteo comme dimensions conformees
- `gold` les transforme en covariables exploitables par le modele

Donc on ne "rajoute pas le calendrier pour la premiere fois en gold".
On le standardise en `silver`, puis on l'exploite en `gold`.

## Tables silver cibles

Le minimum serieux en production est le suivant.

### `silver_order_lines`

Table transactionnelle normalisee.

Colonnes minimales :

- `source_system`
- `source_file_id`
- `source_record_hash`
- `silver_run_id`
- `location_id`
- `location_timezone`
- `order_id`
- `order_line_id`
- `product_id`
- `product_name_raw`
- `category_level_1`
- `category_level_2`
- `category_level_3`
- `sold_at_local_ts`
- `sold_at_utc_ts`
- `business_date`
- `quantity`
- `gross_amount`
- `discount_amount`
- `net_amount`
- `tax_amount`
- `currency`
- `void_flag`
- `refund_flag`
- `cancel_flag`
- `line_status`
- `data_quality_status`

Objectif :

- conserver le fait metier normalise avant agregation

### `silver_products`

Dimension produit conformee.

Colonnes minimales :

- `source_system`
- `product_id`
- `product_name`
- `category_level_1`
- `category_level_2`
- `category_level_3`
- `active_flag`
- `valid_from_dt`
- `valid_to_dt`

### `silver_locations`

Dimension site conformee.

Colonnes minimales :

- `source_system`
- `location_id`
- `location_name`
- `timezone`
- `region_id`
- `brand_id`
- `management_group_id`
- `opened_flag`
- `valid_from_dt`
- `valid_to_dt`

### `silver_calendar`

Dimension calendrier standardisee.

Colonnes minimales :

- `dt`
- `weekday_name`
- `day_of_week`
- `week_of_year`
- `month`
- `quarter`
- `year`
- `holiday_flag`
- `holiday_name`
- `school_holiday_flag`
- `bridge_day_flag`
- `month_end_flag`
- `payday_proximity`

### `silver_weather_daily`

Dimension meteo standardisee au grain journalier.

Colonnes minimales :

- `dt`
- `region_id`
- `precipitation_mm`
- `avg_temperature_c`
- `min_temperature_c`
- `max_temperature_c`
- `humidity_avg`
- `wind_level_avg`
- `weather_data_available_flag`

### `silver_promotions`

Table promo standardisee si disponible.

Colonnes minimales :

- `dt`
- `location_id`
- `product_id`
- `promo_flag`
- `discount_pct`
- `discount_amount`
- `promo_type`

### `silver_daily_product_demand`

Table centrale de la silver forecasting.

Grain :

- `dt x location_id x product_id`

Colonnes minimales :

- `silver_run_id`
- `source_system`
- `source_partition_date`
- `series_id`
- `dt`
- `location_id`
- `product_id`
- `category_level_1`
- `category_level_2`
- `category_level_3`
- `observed_demand_qty`
- `observed_revenue_net`
- `observed_discount_amount`
- `tickets_count`
- `avg_selling_price`
- `promo_flag`
- `location_open_flag`
- `day_complete_flag`
- `missing_sales_flag`
- `observed_stockout_flag`
- `observed_stockout_available`
- `refund_qty`
- `void_qty`
- `cancel_qty`
- `anomaly_flag`
- `anomaly_score`

### `silver_quality_incidents`

Journal de qualite obligatoire.

Colonnes minimales :

- `incident_id`
- `silver_run_id`
- `source_system`
- `entity_type`
- `entity_key`
- `dt`
- `severity`
- `rule_name`
- `incident_type`
- `observed_value`
- `expected_range_or_rule`
- `status`
- `created_at`

### `silver_run_manifest`

Manifest de run pour auditabilite.

Colonnes minimales :

- `silver_run_id`
- `pipeline_version`
- `source_system`
- `started_at`
- `finished_at`
- `input_row_count`
- `output_row_count`
- `duplicate_count`
- `null_count_summary`
- `incident_count`
- `status`

## Regles metier obligatoires

### Regles sur la target

- la target silver de reference est la `demande observee`, pas la demande latente
- un refund ne doit pas etre fusionne silencieusement avec une vente positive
- un void ou une annulation ne doit pas etre compte comme demande
- un jour ferme n'est pas un zero de demande
- un jour partiellement charge n'est pas un zero de demande
- un stockout observe n'est pas une faible demande
- aucune forward-fill de la target

### Regles sur le temps

- toute timestamp source doit etre tracee
- toute date locale doit etre calculee dans la timezone du site
- l'agrégation journaliere doit se faire apres normalisation des timezones
- l'ordre temporel doit etre verifie avant toute agrégation ou derivation

### Regles sur les cles

- IDs comme colonnes explicites, jamais caches dans l'index
- unicite verifiee au grain canonique
- aucun merge de grains differents sans agrégation explicite

### Regles sur les valeurs

- les prix negatifs ou incoherents doivent etre flagges
- les quantites impossibles doivent etre routees en incident qualite
- les nulls sur colonnes critiques doivent etre traces avant toute decision de traitement

## Donnees manquantes

La silver doit qualifier les absences de donnees, pas les camoufler.

Cas a distinguer explicitement :

- `true_zero_demand`
- `location_closed`
- `missing_source_data`
- `partial_day_data`
- `observed_stockout`

Ces etats doivent vivre dans des colonnes ou flags explicites.
Ils ne doivent pas etre recondenses dans un seul indicateur opaque.

## Outliers

La silver ne supprime pas les outliers par defaut.
Elle les classe.

Trois familles :

- `data_error`
  - doublon ticket
  - timestamp hors plage
  - prix impossible
  - quantite impossible
- `business_outlier_explainable`
  - promo
  - jour ferie
  - evenement local
  - lancement produit
- `unexplained_outlier`
  - spike ou drop non explique

Le traitement downstream se decide en `gold` ou lors de l'analyse, pas dans la silver par suppression implicite.

## Checks qualite obligatoires

La silver doit publier des checks automatises a chaque run.

### Checks de schema

- presence des colonnes requises
- types conformes
- dates typées
- timezone disponible quand necessaire

### Checks de grain

- unicite sur `order_line_id` quand attendu
- unicite sur `dt x location_id x product_id` pour `silver_daily_product_demand`
- detection des doublons source

### Checks metier

- quantites negatives
- prix negatifs
- revenu incoherent avec prix et quantite
- refunds sans ligne source identifiable si la source le permet
- site inconnu
- produit inconnu

### Checks temporels

- tri temporel
- detection de jours manquants
- detection de jours partiels
- detection de trous anormaux par serie

### Checks de missingness

- taux de null par colonne
- taux de null par site
- taux de null par produit
- comparaison avant et apres transformation

### Checks d'anomalies

- volumes quotidiens hors bandes historiques
- nombre de tickets anormal
- prix moyens anormaux
- drop ou spike soudain non explique

## Strategie de publication

Recommandation :

- format primaire : `Parquet`
- metadata de run : `JSON` ou table SQL dediee
- table silver publiee dans le warehouse si disponible

Le `CSV` ne doit servir qu'au debug ou a l'export ponctuel.

## SQL ou Python / Polars ?

### Reponse courte

Dans l'architecture cible :

- `raw_landing` contient les JSON bruts
- `bronze` les projette en tabulaire proche source
- `silver` lit ce tabulaire bronze

Dans ce scenario, la `silver` doit etre `SQL-first`, avec `Python / Polars` en support pour certains cas specifiques.

### Ce qui doit aller en Python / Polars en amont ou en support

- parsing heterogene des fichiers source
- adaptateurs POS differents
- flatten de JSON
- corrections de schema et cast complexes
- regles metier proceduralement riches
- detection d'anomalies et validations de contrat
- ecriture de manifests et d'incidents qualite

En pratique, ces traitements vivent surtout :

- dans le `raw_landing -> bronze`
- ou dans des jobs de support a la silver

Ils ne doivent pas etre le moteur principal de la silver si la bronze est deja tabulaire.

`Polars` est tres bon pour :

- traitements colonnes performants
- fichiers `Parquet` et `CSV`
- transformations locales ou batch tres rapides
- pipeline tabulaire en memoire / streaming

### Ce qui doit aller en SQL

- tables silver publiees et requetables
- jointures de dimensions
- aggregations declaratives
- controles d'unicite et checks declaratifs
- exposition analytique
- consommation par les autres couches du systeme

SQL est tres bon pour :

- lisibilite des transformations metier stabilisees
- exploitation dans un warehouse
- monitoring et audit SQL natifs
- joins lourds et orchestration data centralisee

### Big data : SQL ou Polars ?

Sur la vraie big data :

- si les donnees vivent deja dans un warehouse distribue et que la bronze est tabulaire, `SQL` gagne en simplicite operationnelle
- si les donnees arrivent en fichiers, exports, batches ou objets, `Polars` est souvent plus rapide et plus agreable que pandas pour les preparer
- `Polars` seul n'est pas un remplaçant d'un moteur distribue complet
- pour une volumetrie vraiment massive multi-tenant et multi-clients, le bon cran d'echelle devient plutot :
  - warehouse SQL distribue
  - DuckDB local pour prototypage
  - ou moteur distribue type Spark si la volumetrie l'exige un jour

### Recommandation Praedixa

Pour Praedixa aujourd'hui, la recommandation la plus efficace est :

- `raw_landing` en JSON brut immutable
- `Python + Polars` ou fonctions JSON natives du warehouse pour construire la `bronze` tabulaire
- `SQL` comme colonne vertebrale de la `silver`
- `SQL` pour la publication et l'exploitation des tables silver stabilisees
- `Parquet` comme format d'echange intermediaire

Autrement dit :

- `Polars` pour fabriquer proprement la `bronze` quand necessaire
- `SQL` pour fabriquer et operer la `silver`

C'est le meilleur compromis entre :

- vitesse d'iteration early-stage
- robustesse de production
- auditabilite
- performance
- maintenabilite

## Decision de design retenue

Praedixa doit considerer la couche `silver` comme une couche de standardisation et de qualite, pas comme une couche de feature engineering.

Le noyau minimal de production est :

- `silver_order_lines`
- `silver_products`
- `silver_locations`
- `silver_calendar`
- `silver_weather_daily`
- `silver_promotions`
- `silver_daily_product_demand`
- `silver_quality_incidents`
- `silver_run_manifest`

## Suite logique

Une fois la silver specifiee, l'etape suivante est de definir :

- la spec de la `gold demand forecasting`
- le contrat de jointure `silver -> gold`
- les checks anti-leakage a imposer a la gold
- le scaffolding de pipeline Python / SQL correspondant
