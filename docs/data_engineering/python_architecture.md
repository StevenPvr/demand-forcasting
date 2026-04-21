# Architecture Python Praedixa

## Objectif

Cette note fixe la frontiere entre le warehouse canonique SQL-first et les stages Python du repo.

Le principe directeur est simple :

- `platform/warehouse/` reste la source canonique pour le medaillon `bronze -> silver -> gold`
- `platform/python/src/praedixa/platform/` et `products/demand_forecast/src/praedixa/demand_forecast/` orchestrent, standardisent et preparent les artefacts Python autour de ce backbone
- le backend modele unique TFT reste une direction explicite, mais pas encore un backend actif dans ce repo

## Couche canonique

Le flux de reference reste :

```text
sources locales / open data
-> bronze DuckDB
-> silver dbt
-> gold dbt
-> stages Python
```

Les modules Python ne doivent pas reimplementer la logique metier deja canonisee dans `dbt`.
Ils consomment le warehouse, appliquent des contrats de cible, produisent des artefacts locaux et preparent l'integration du backend modele.

## Regles de structure

Chaque package Python actif suit la meme logique :

- `main.py` : point d'entree executable directement
- `pipeline.py` : facade publique stable, orientee orchestration
- sous-modules internes : logique par responsabilite

Exemples de responsabilites a isoler :

- `splits`
- `sampling`
- `correlation`
- `baselines`
- `reference_data`
- `artifacts`
- `backend_adapter`

Les points d'entree sous `apps/` doivent rester des wrappers CLI minces au-dessus de modules Python importables et testables.

## Contrats partages

Les briques partagees vivent sous `platform/python/src/praedixa/platform/` et `products/demand_forecast/src/praedixa/demand_forecast/` :

- `platform.runtime.paths` : chemins de reference
- `platform.runtime.constants` : constantes runtime stables
- `platform.runtime.warehouse` : configuration warehouse / dbt / DuckDB partagee par les runners
- `demand_forecast.contracts.targets` : contrat de cible et reconstruction business-safe
- `platform.governance.source_registry` : politique de sources commercialement exploitables
- `demand_forecast.backends.tft.backend` et `demand_forecast.backends.tft.model_utils` : facade explicite vers le backend modele unique

## Frontiere TFT

Tant que le backend TFT n'est pas branche :

- `optimisation` et `evaluation` doivent exposer clairement un statut placeholder
- les helpers reutilisables peuvent vivre dans ces packages
- aucun code mort critique ne doit rester melange a l'orchestrateur public

Autrement dit :

- on peut preparer les donnees, les folds, les baselines et les artefacts
- on ne doit pas faire croire que l'entrainement final est deja actif

## Critere d'une bonne extraction

Une extraction est valide si elle :

- reduit la taille d'un gros fichier sans changer l'API publique testee
- rend la logique plus testable independamment
- evite les cycles d'import
- preserve la lecture metier du pipeline
- facilite une iteration future par agent sans re-decouverte complete du repo
