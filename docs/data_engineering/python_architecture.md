# Architecture Python Praedixa

## Objectif

Cette note fixe la frontiere entre le warehouse canonique SQL-first et les stages Python du repo.

Le principe directeur est simple :

- `platform/warehouse/` reste la source canonique pour le medaillon `bronze -> silver -> gold`
- `platform/python/src/praedixa/platform/` et `products/demand_forecast/src/praedixa/demand_forecast/` orchestrent, standardisent et preparent les artefacts Python autour de ce backbone
- `xgboost` reste le backend actif par défaut pendant que TFT devient le backend de référence cible

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
- `demand_forecast.backends.tft` : backend TFT présent, encore en transition avant promotion par défaut

## Frontiere Modele

Tant que TFT n'est pas le backend par défaut :

- `xgboost` reste utilisable comme backend courant;
- `xgboost` résout son runtime en `auto`: CUDA si le preflight XGBoost passe,
  CPU sinon;
- les contrats `gold`, bundle et feature mapping doivent rester compatibles TFT;
- `optimisation` et `evaluation` doivent expliciter le backend réellement utilisé;
- aucun code mort critique ne doit rester melange a l'orchestrateur public.

Autrement dit :

- on peut préparer les données, les folds, les baselines et les artefacts;
- on ne doit pas présenter TFT comme chemin promu tant que la parité, les tests et les artefacts ne sont pas verrouillés.

## Critere d'une bonne extraction

Une extraction est valide si elle :

- reduit la taille d'un gros fichier sans changer l'API publique testee
- rend la logique plus testable independamment
- evite les cycles d'import
- preserve la lecture metier du pipeline
- facilite une iteration future par agent sans re-decouverte complete du repo
