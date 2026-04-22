# Praedixa Research

Repo de recherche et d’architecture data pour Praedixa.

L’objectif du projet est de construire un pipeline sérieux de prévision pour des opérations alimentaires périssables, avec une cible analytique prioritaire :

- **idéalement** : prévision de **demande latente**
- **sinon** : prévision de **ventes observées**

Le repo est aujourd’hui surtout un socle de **data engineering**, de **standardisation multi-sources**, de **contrats de données**, de **backtesting temporel** et de **préparation du futur backend modèle unique TFT**.

Le point important à garder en tête :

- le pipeline data est réel et exploitable
- le backend modèle final **TFT** est la direction retenue
- il n’est **pas encore branché complètement** dans ce repo

## Vision du repo

Praedixa ne vise pas un simple forecasting générique.

Le repo sert à préparer une couche de prévision opérationnelle capable de :

- unifier des sources hétérogènes
- respecter la temporalité réelle de l’information
- distinguer, quand c’est possible, **demande latente** et **vente observée**
- produire un panel canonique compatible avec une modélisation globale
- garder une lecture business claire : gaspillage, ruptures, coût matière, niveau de service

## Ce que le repo fait aujourd’hui

### Actif

- ingestion locale de données bronze dans DuckDB
- transformation `bronze -> silver -> gold` via `dbt`
- standardisation de plusieurs datasets vers un schéma quotidien canonique
- gestion d’un registre de conformité des sources utilisables commercialement
- enrichissements exogènes open-source et dérivés
- génération de données `first-party` et synthétiques pour le cold start
- préparation locale de jeux de travail pour la suite du pipeline
- baselines, contrats de cible et structure d’évaluation

### Transition

- `features_selection_lag` reste disponible comme couche de préparation et de diagnostic locale
- `optimisation` et `evaluation` gardent leur structure, mais sont déjà réalignées vers un backend TFT unique

### Placeholder

- backend d’entraînement **TFT**
- tuning TFT
- évaluation finale TFT
- artefacts de modèle entraîné en production-ready

Le repo a été nettoyé pour sortir l’ancienne logique boosting. Il n’y a pas de faux remplacement “tabulaire” à considérer comme la direction cible.

## Architecture

### Vue d’ensemble

Le repo est organisé autour de 5 couches principales :

1. `apps/`
   Expose les points d’entrée exécutables du monorepo.
2. `platform/warehouse/`
   Contient le projet `dbt + DuckDB` pour les couches `bronze`, `silver` et `gold`.
3. `platform/python/src/praedixa/platform/`
   Contient la logique Python plateforme : runtime, standardisation, signaux exogènes, gouvernance et runners warehouse.
4. `products/demand_forecast/src/praedixa/demand_forecast/`
   Contient la logique produit wedge : feature screening, training, evaluation, contrats et backend TFT.
5. `tests/`
   Couvre les pipelines et les invariants du repo.

### Flux canonique

Le flux de référence aujourd’hui est :

1. chargement bronze local
2. `silver` SQL-first
3. `gold` SQL-first
4. standardisation globale quotidienne
5. préparation du backend modèle

En pratique :

```text
sources locales / open data
-> bronze
-> silver
-> gold
-> panel canonique quotidien
-> backend TFT (à brancher)
```

La séparation entre backbone `dbt` et stages Python est détaillée dans
[docs/data_engineering/python_architecture.md](/Users/steven/Programmation/research_praedixa/docs/data_engineering/python_architecture.md).

## Structure du repo

### Dossiers principaux

- `platform/python/src/praedixa/platform/`
  Logique Python plateforme.
- `platform/python/src/praedixa/platform/datasets/standardization/`
  Standardisation multi-sources au format quotidien canonique.
- `products/demand_forecast/src/praedixa/demand_forecast/training/`
  Surface de tuning/optimisation. Structure conservée, backend TFT encore à brancher.
- `products/demand_forecast/src/praedixa/demand_forecast/evaluation/`
  Surface d’évaluation et d’artefacts métier. Backend TFT encore à brancher.
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/backend.py`
  Point d’arrêt explicite qui rappelle que le backend modèle cible est TFT.
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`
  Surface minimale prévue pour le futur backend TFT.
- `apps/warehouse/main.py`
  Entry point unique du medaillon local `bronze -> silver -> gold`.
- `apps/warehouse/run_silver/main.py`
  Runner local specialise `bronze -> silver`.
- `apps/warehouse/run_gold/main.py`
  Runner local specialise `silver -> gold`.
- `apps/platform/prepare_first_party_onboarding/main.py`
  Prépare un feed first-party minimal au bon format.
- `apps/platform/generate_synthetic_cold_start/main.py`
  Génère des données synthétiques de cold start.
- `platform/warehouse/`
  Projet `dbt`.
- `tests/`
  Tests unitaires et d’intégration locale.

## Philosophie cible : demande latente d’abord

Le repo doit être lu avec la distinction suivante :

- **demande latente** :
  ce que le point de vente aurait vendu/servi sans rupture, fermeture partielle ou contrainte opérationnelle
- **vente observée** :
  ce qui a effectivement été enregistré

Cette distinction est centrale pour Praedixa :

- quand les signaux de censure existent, la bonne cible est la demande latente
- quand ils n’existent pas, on tombe explicitement en mode ventes observées

Le repo ne doit jamais faire croire qu’une prévision de ventes observées est automatiquement une prévision de demande latente.

## Politique données et conformité commerciale

Le repo intègre une politique explicite sur les sources utilisables commercialement.

### Registre de sources

La référence est :

- `platform/warehouse/seeds/source_registry.csv`
- `platform/python/src/praedixa/platform/governance/source_registry.py`

Ce registre porte notamment :

- le type de source
- la licence
- l’autorisation d’usage commercial
- l’autorisation d’entraînement ML
- le statut de revue
- l’inclusion ou non dans le corpus d’entraînement

### Principe

Une source n’entre pas dans le corpus canonique d’entraînement si elle n’est pas explicitement compatible avec :

- l’usage commercial
- l’entraînement ML
- la policy du repo

### Exemples actuels

Sources dataset autorisées dans l’état actuel :

- `freshretail`
- `freshretail_lt`
- `bakery`
- `uci_online_retail`
- `uci_online_retail_ii`
- `mendeley_ecommerce`
- `mendeley_pharmacy_id`
- `mendeley_bangladesh_retail`
- `first_party_daily`
- `synthetic_v1`

Source explicitement exclue :

- `m5`

Providers à statut particulier :

- `world_bank_indicators` : autorisé
- `french_school_calendar_ics` : autorisé
- `openstreetmap_odbl` : autorisé pour features dérivées
- `open_meteo_api` : quarantaine contractuelle
- `nager_date_api` : quarantaine contractuelle
- `fred_public_api` : bloqué

## Schéma canonique

Le cœur du repo est un panel quotidien unifié, au grain :

```text
dt x location_id x product_id
```

Colonnes structurantes fréquentes :

- `dt`
- `series_id`
- `location_id`
- `product_id`
- `target_demand_qty_d_plus_1`
- `target_delta_log_wow_d_plus_1`
- `lag_1`
- `lag_7`
- `rolling_mean_7`
- `holiday_flag`
- `weather_temperature`
- `observed_stockout_flag`

Le pipeline conserve aussi la logique de contrat cible via `products/demand_forecast/src/praedixa/demand_forecast/contracts/targets.py`.

## Datasets et standardisation

### Pipeline global

Point d’entrée :

```bash
.venv/bin/python -m apps.platform.build_global_dataset.main
```

Ce pipeline construit :

- `var/datasets/global_dataset/freshretail_daily.parquet`
- `var/datasets/global_dataset/commercial_external_daily.parquet`
- `var/datasets/global_dataset/bakery_daily.parquet` si disponible
- `var/datasets/global_dataset/global_daily_demand.parquet`
- `var/datasets/global_dataset/global_daily_demand_manifest.json`

### Ce que fait la standardisation

- harmonisation des colonnes
- alignement sur le schéma canonique
- validation qualité
- émission d’un manifest de synthèse

## Warehouse local

### Medaillon

Point d’entrée de référence :

```bash
.venv/bin/python -m apps.warehouse.main
```

Ce runner :

- lance la step `silver` avec chargement bronze intégré
- exécute `dbt seed`, `dbt run` et `dbt test` sur `silver`
- enchaîne ensuite sur la step `gold`
- peut rafraîchir les enrichissements open-source avant `gold`
- exécute `dbt seed`, `dbt run` et `dbt test` sur `gold`
- reste un `main.py` sans parsing CLI, prévu pour un lancement direct depuis l'IDE ou `python -m`

Les runners spécialisés `apps.warehouse.run_silver.main` et `apps.warehouse.run_gold.main` restent supportés pour les cas ciblés, mais ne sont plus le point d’entrée recommandé.

### Fichier DuckDB local

Par défaut :

```text
var/warehouse/praedixa.duckdb
```

Pour plus de détails warehouse :

- voir `platform/warehouse/README.md`

## Feature prep locale

Point d’entrée :

```bash
.venv/bin/python -m apps.demand_forecast.run_feature_screening.main
```

Cette étape est **transitionnelle**. Elle n’est **pas** la cible finale du repo. Elle reste utile pour :

- explorer les signaux temporels
- produire des jeux de travail locaux
- garder un outillage de diagnostic pendant la transition vers le backend TFT

Sorties typiques :

- `var/experiments/demand_forecast/feature_screening/train_selection_70_selected.parquet`
- `var/experiments/demand_forecast/feature_screening/train_tuning_30_selected.parquet`
- `var/experiments/demand_forecast/feature_screening/selected_lag_features.json`

## Backend modèle : état actuel

### Direction retenue

Le backend modèle visé est :

- **un pipeline unique TFT**

### État actuel

Les surfaces suivantes existent :

- `products/demand_forecast/src/praedixa/demand_forecast/training/`
- `products/demand_forecast/src/praedixa/demand_forecast/evaluation/`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/backend.py`

Mais le vrai backend TFT n’est pas encore branché.

Concrètement :

- `optimisation` et `evaluation` sont des placeholders structurels explicites
- elles échouent explicitement tant que le backend TFT n’est pas implémenté
- cela évite de laisser croire qu’un ancien backend est encore valide

## Cold start, first-party et synthétique

Le repo prépare aussi le futur mode cold start.

### First-party onboarding

Script :

```bash
.venv/bin/python -m apps.platform.prepare_first_party_onboarding.main
```

But :

- préparer un feed client minimal
- converger vers le schéma canonique

### Données synthétiques

Script :

```bash
.venv/bin/python -m apps.platform.generate_synthetic_cold_start.main
```

But :

- générer des séries synthétiques de cold start
- renforcer les cas d’ouverture / faible historique / manque de données

## Installation et environnement

### Prérequis

- Python `>= 3.13`
- environnement virtuel `.venv`

### Dépendances

Le projet utilise notamment :

- `duckdb`
- `dbt-duckdb`
- `pandas`
- `polars`
- `numpy`
- `optuna`
- `matplotlib`
- `openpyxl`

Le backend TFT n’est pas encore ajouté aux dépendances runtime dans ce repo.

## Commandes utiles

### Construire le dataset global

```bash
.venv/bin/python -m apps.platform.build_global_dataset.main
```

### Charger la bronze locale

```bash
.venv/bin/python -m apps.warehouse.load_bronze.main
```

### Lancer silver local

```bash
.venv/bin/python -m apps.warehouse.run_silver.main
```

### Lancer gold local

```bash
.venv/bin/python -m apps.warehouse.run_gold.main
```

### Préparer les jeux locaux de travail

```bash
.venv/bin/python -m apps.demand_forecast.run_feature_screening.main
```

### Préparer un feed first-party

```bash
.venv/bin/python -m apps.platform.prepare_first_party_onboarding.main
```

### Générer des données synthétiques de cold start

```bash
.venv/bin/python -m apps.platform.generate_synthetic_cold_start.main
```

### Tests ciblés

```bash
.venv/bin/python -m unittest \
  tests.platform.datasets.standardization.test_pipeline \
  tests.apps.warehouse.test_run_silver \
  tests.apps.warehouse.test_run_gold \
  tests.products.demand_forecast.feature_screening.test_pipeline \
  tests.products.demand_forecast.training.test_pipeline \
  tests.products.demand_forecast.evaluation.test_pipeline \
  tests.products.demand_forecast.backends.tft.test_model_utils
```

## Tests et vérification

Le repo contient des tests de :

- schéma canonique
- qualité du dataset global
- runners silver/gold
- enrichissements open-source
- surfaces optimisation/évaluation
- transition vers le backend TFT

Commande de vérification canonique :

```bash
.venv/bin/python -m apps.dev.verify_repo.main
```

Les checks exécutés restent :

```bash
.venv/bin/python -m compileall praedixa platform/python/src products/demand_forecast/src apps
.venv/bin/python -m unittest discover -s tests
.venv/bin/dbt parse --project-dir platform/warehouse --profiles-dir platform/warehouse
```

## Limites actuelles

- le backend TFT n’est pas encore implémenté de bout en bout
- toutes les sources n’exposent pas des signaux suffisants pour reconstruire la demande latente
- certaines sources externes utiles restent volontairement en quarantaine tant que la conformité commerciale n’est pas verrouillée
- la couche `features_selection_lag` reste un outil de transition, pas la destination finale

## Roadmap technique immédiate

Les priorités cohérentes avec l’état actuel du repo sont :

1. figer le socle repo / doc / architecture
2. conserver `gold` comme contrat canonique d’entrée
3. brancher le vrai backend **TFT**
4. reconnecter `optimisation` et `evaluation`
5. mesurer proprement la différence entre :
   - demande latente quand elle est observable/estimable
   - ventes observées sinon

## Fichiers à connaître en priorité

- `AGENTS.md`
- `platform/warehouse/README.md`
- `platform/python/src/praedixa/platform/datasets/standardization/pipeline.py`
- `platform/python/src/praedixa/platform/governance/source_registry.py`
- `platform/warehouse/seeds/source_registry.csv`
- `apps/warehouse/run_silver/main.py`
- `apps/warehouse/run_gold/main.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/backend.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`

## Résumé

Ce repo n’est plus un terrain d’essai pour multiplier les backends modèles.

Le chargeur bronze canonique est `apps/warehouse/load_bronze/main.py`. Il n’existe plus de variante ClickHouse distincte dans le repo.

C’est maintenant un socle propre pour :

- la standardisation multi-sources
- la gouvernance de la donnée
- le panel canonique de prévision
- la préparation du futur pipeline unique **TFT**
- la lecture rigoureuse de la prévision en **demande latente** quand c’est possible, et en **ventes observées** sinon
