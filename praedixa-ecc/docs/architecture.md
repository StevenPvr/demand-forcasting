# Architecture V1

## Positionnement

Praedixa ECC n'est pas un clone generaliste d'ECC.
La V1 est un framework interne Codex-only, optimise pour des cas de data science et de forecasting rigoureux, avec une surcouche metier Praedixa.

## Principe de conception

Le framework suit une architecture en deux couches plus une surface d'execution :

- `core` : protocoles generiques de data contracts, anti-leakage, backtesting, baselines, evaluation et recherche experimentale
- `praedixa` : conventions metier demande/effectifs, signaux retail perissable, traduction business et ROI
- `surface codex` : `.codex/` et `.agents/skills/` installes dans le repo actif

## Boucles

### Forecasting core loop

`dataset contract -> features -> split temporel -> baseline -> modele -> walk-forward -> metriques -> diagnostic`

### ML research loop

`hypothese -> design experimental -> run -> comparaison -> validation -> decision`

### Praedixa operating loop

`ingestion -> qualite data -> prevision demande/effectifs -> evaluation business -> livrable exploitable`

## Source de verite

Le framework est maintenu dans `praedixa-ecc/`.

Le script `sync_codex_surface.py` projette la surface active dans :

- `.codex/`
- `.agents/skills/`

Cette separation permet d'ameliorer le framework sans bricoler directement la surface active du repo.

## Critere de reussite V1

Si Codex travaille sur un vrai probleme Praedixa de prevision, il doit naturellement :

- identifier la bonne boucle
- appliquer les garde-fous anti-erreurs classiques
- produire un artefact standardise
- conclure avec une lecture business exploitable
