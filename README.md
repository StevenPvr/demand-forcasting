# Praedixa Research

Repo de recherche et d’architecture data pour Praedixa.

L’objectif du projet est de construire un pipeline sérieux de prévision pour des opérations alimentaires périssables, avec une cible analytique prioritaire :

- **idéalement** : prévision de **demande latente**
- **sinon** : prévision de **ventes observées**

Le repo est aujourd’hui surtout un socle de **data engineering**, de **standardisation multi-sources**, de **contrats de données**, de **backtesting temporel** et de transition vers un backend modèle de référence **TFT**.

Le point important à garder en tête :

- le pipeline data est réel et exploitable
- le backend modèle de référence visé est **TFT**
- `xgboost` reste le backend câblé par défaut tant que la bascule TFT n’est pas complète

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
- baselines, contrats de cible et structure d’évaluation

### En transition

- backend d’entraînement **TFT** présent mais pas encore chemin par défaut
- tuning TFT en cours de durcissement
- évaluation finale à aligner sur les artefacts TFT promouvables
- artefacts de modèle entraîné à rendre production-ready

La direction cible reste TFT, mais le repo conserve `xgboost` comme backend opérationnel par défaut pendant la transition.

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
   Contient la logique produit wedge : training, evaluation, contrats et backends modèles.
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
-> bundle d'entraînement
-> backend modèle (`xgboost` par défaut, TFT cible)
```

La séparation entre backbone `dbt` et stages Python est détaillée dans
[docs/data_engineering/python_architecture.md](docs/data_engineering/python_architecture.md).

## Structure du repo

### Dossiers principaux

- `platform/python/src/praedixa/platform/`
  Logique Python plateforme.
- `platform/python/src/praedixa/platform/datasets/standardization/`
  Standardisation multi-sources au format quotidien canonique.
- `products/demand_forecast/src/praedixa/demand_forecast/training/`
  Surface de tuning/optimisation. `xgboost` par défaut, TFT en transition.
- `products/demand_forecast/src/praedixa/demand_forecast/evaluation/`
  Surface d’évaluation et d’artefacts métier.
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/`
  Backend TFT présent, à finaliser avant promotion comme backend par défaut.
- `apps/warehouse/main.py`
  Entry point unique du medaillon local explicite `bronze core -> silver -> open exogenous -> gold`.
- `apps/warehouse/run_silver/main.py`
  Runner local specialise `bronze -> silver`.
- `apps/warehouse/run_gold/main.py`
  Runner local specialise `silver -> gold`.
- `apps/platform/prepare_first_party_onboarding/main.py`
  Prépare un feed first-party minimal au bon format.
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
- `first_party_daily`

Providers à statut particulier :

- `world_bank_indicators` : autorisé
- `insee_bdm` : autorisé
- `deterministic_public_holidays` : autorisé
- `praedixa_location_metadata` : autorisé
- `french_school_calendar_ics` : autorisé
- `openstreetmap_odbl` : autorisé pour features dérivées
- `open_meteo_api` : quarantaine contractuelle
- `nager_date_api` : quarantaine contractuelle

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

Ce point d’entrée sert aujourd’hui à valider et synthétiser l’assemblage global
en mémoire. La source aval de référence pour l’entraînement reste le panel
médaillon `gold.gold_model_training_panel_d1`.

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

- charge les sources coeur dans `bronze`
- matérialise `silver`
- rafraîchit les exogènes ouverts depuis les bornes silver
- charge les `bronze_open_*`
- matérialise `gold.gold_model_training_panel_d1`
- peut rafraîchir les enrichissements open-source avant `gold`
- exécute `dbt seed`, `dbt run` et `dbt test` sur `gold`
- reste un `main.py` sans parsing CLI, prévu pour un lancement direct depuis l'IDE ou `python -m`

Contrat de split actif:

- `freshretail` et `freshretail_lt`: `train`/`val` chronologique 60/40 pour
  l'optimisation;
- `bakery`: `test` uniquement sur les 3 derniers mois pour l'évaluation finale;
- dans le bundle, `val` est matérialisé en `tuning.parquet` et `test` en
  `valid.parquet`.

Les runners spécialisés `apps.warehouse.run_silver.main` et `apps.warehouse.run_gold.main` restent supportés pour les cas ciblés, mais ne sont plus le point d’entrée recommandé.

### Fichier DuckDB local

Par défaut :

```text
var/warehouse/praedixa.duckdb
```

Pour plus de détails warehouse :

- voir `platform/warehouse/README.md`

## Backend modèle : état actuel

### Direction retenue

Le backend modèle visé est :

- **un pipeline unique TFT**

### État actuel

Les surfaces suivantes existent :

- `products/demand_forecast/src/praedixa/demand_forecast/training/`
- `products/demand_forecast/src/praedixa/demand_forecast/evaluation/`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/`
- `apps/demand_forecast/run_tft_training/main.py`

Le backend TFT est présent mais la transition n’est pas encore complète.
Le backend par défaut reste `xgboost` dans `SUPPORTED_MODEL_BACKENDS = ("xgboost", "tft")`.

Concrètement :

- `xgboost` reste le chemin câblé par défaut pour les runs courants
- `xgboost` sélectionne automatiquement CUDA quand le runtime XGBoost le supporte,
  sinon il retombe sur CPU
- `tft` est la direction cible pour le backend modèle de référence
- les contrats `gold`, bundle, feature mapping et artefacts sont progressivement durcis pour cette bascule

### Runtime XGBoost

Le profil officiel XGBoost est `auto` au niveau du backend. Au démarrage, le
code exécute un mini preflight XGBoost CUDA réel, pas seulement un test Torch.

- preflight CUDA OK : `device="cuda"`, `tree_method="hist"`,
  `QuantileDMatrix`, folds Optuna sérialisés sur un GPU;
- preflight CUDA KO : `device="cpu"`, `tree_method="hist"`, folds Optuna
  parallélisés côté CPU;
- Metal/MPS n'est pas supporté par XGBoost et reste rejeté explicitement.

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
- `torch`
- `lightning`
- `pytorch-forecasting`

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

### Préparer un feed first-party

```bash
.venv/bin/python -m apps.platform.prepare_first_party_onboarding.main
```

### Tests ciblés

```bash
.venv/bin/python -m unittest \
  tests.platform.datasets.standardization.test_pipeline \
  tests.apps.warehouse.test_run_silver \
  tests.apps.warehouse.test_run_gold \
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

Commande de vérification canonique une fois les outils de typage installés dans
l'environnement :

```bash
.venv/bin/python -m apps.dev.verify_repo.main
```

Dans ce checkout, les binaires `pyright`, `mypy` et `ruff` ne sont pas installés
dans `.venv/bin`. La validation ciblée reproductible passe donc par `uvx` pour
le typage tant que ces outils ne sont pas ajoutés au groupe `dev`.

Checks à exécuter avant de promouvoir un changement médaillon :

```bash
.venv/bin/python -m compileall praedixa platform/python/src products/demand_forecast/src apps
uvx pyright --project pyrightconfig.json platform/python/src/praedixa/platform/warehouse platform/python/src/praedixa/platform/governance products/demand_forecast/src/praedixa/demand_forecast/training products/demand_forecast/src/praedixa/demand_forecast/training_bundle products/demand_forecast/src/praedixa/demand_forecast/backends/tft
.venv/bin/python -m unittest discover -s tests
.venv/bin/dbt parse --project-dir platform/warehouse --profiles-dir platform/warehouse
```

## Limites actuelles

- le backend TFT n’est pas encore le chemin par défaut de bout en bout
- toutes les sources n’exposent pas des signaux suffisants pour reconstruire la demande latente
- certaines sources externes utiles restent volontairement en quarantaine tant que la conformité commerciale n’est pas verrouillée

## Roadmap technique immédiate

Les priorités cohérentes avec l’état actuel du repo sont :

1. figer le socle repo / doc / architecture
2. conserver `gold` comme contrat canonique d’entrée
3. promouvoir **TFT** comme backend de référence après parité contrôlée
4. aligner `optimisation` et `evaluation` sur les artefacts TFT promouvables
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
