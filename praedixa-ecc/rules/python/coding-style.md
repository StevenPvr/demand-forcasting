# Python Coding Style

## Principes

- Ecrire du code d'abord lisible, ensuite clever.
- Preferer l'explicite a la magie.
- Garder des fonctions petites, stables et testables.
- Eviter la logique metier dispersee dans les scripts d'entree.

## Structure

- `from __future__ import annotations` en premiere ligne des nouveaux modules.
- Les imports sont groupes par standard library, tiers, local.
- Les points d'entree ont un `main()` mince ; l'orchestration appelle des fonctions ou pipelines nommes.
- Une fonction doit rester focalisee et idealement sous 50 lignes hors docstring.
- Un module doit rester lisible et idealement sous 500 lignes.

## Typage

- Typage explicite pour fonctions publiques, structures de config et variables critiques.
- Utiliser les types modernes (`list[str]`, `dict[str, int]`, `Path | None`) plutot que les aliases legacy quand la version Python le permet.
- Preferer `Protocol` pour les contrats structurels et `dataclass` pour les objets de config ou DTO simples.
- Ne pas masquer une ambiguite metier avec `Any`.

## Nommage

- Fonctions : verbe + objet (`load_dataset`, `build_training_frame`, `compute_bias_metrics`).
- Booleens : prefixes explicites (`is_`, `has_`, `can_`, `should_`).
- Constantes critiques en majuscules nommees (`TARGET_COL`, `KEY_COLS`, `DEFAULT_HORIZON`).
- Eviter les abreviations opaques hors vocabulaire metier etabli.

## I/O et config

- Les chemins, seeds, horizons et colonnes critiques passent par config ou constantes nommees.
- Utiliser `pathlib.Path`, pas des concatenations de strings pour les chemins.
- Ne jamais hardcoder un secret, un token ou un chemin utilisateur dans le code source.
- Les fonctions qui lisent/ecrivent doivent l'indiquer clairement dans leur nom ou leur docstring.

## Logging et erreurs

- `logging` pour le code de pipeline ; pas de `print` dans le code de prod.
- Attraper des exceptions specifiques ; pas de `except Exception:` silencieux sans raison explicite.
- Chainer les exceptions quand on remonte une erreur metier.
- Echouer bruyamment sur les violations de contrat de donnees ou de protocole.

## Commentaires et docstrings

- Docstring obligatoire pour les fonctions publiques ou non triviales.
- Expliquer les hypotheses, les invariants, ou le pourquoi d'une transformation, pas reformuler le code.
- Preferer un nom de fonction clair a un commentaire compensatoire.
