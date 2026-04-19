# Python Patterns

## Objets simples

- Utiliser `dataclass` pour les objets de config, DTO ou resultats simples.
- Utiliser `frozen=True` quand l'immutabilite clarifie le contrat.
- Utiliser `Protocol` pour exprimer un contrat de comportement sans heritage inutile.

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    train_path: Path
    val_path: Path
    output_dir: Path
    random_seed: int
```

## Fonctions

- Preferer les fonctions pures pour la transformation de donnees.
- Separarer clairement :
  - lecture/ecriture
  - transformation
  - entrainement
  - evaluation
- Eviter les fonctions qui renvoient des tuples opaques si un dataclass ou dict nomme rend le contrat plus clair.

## Collections et iteration

- Preferer comprehensions et fonctions nommees aux boucles verbeuses quand la logique reste lisible.
- Eviter les side effects caches dans une comprehension.
- Utiliser des generateurs pour les flux paresseux ou les collections potentiellement lourdes.

## Dates, chemins, systeme

- Utiliser `Path` pour les chemins.
- Utiliser des `datetime` ou timestamps explicites pour le temps ; pas de strings implicites au coeur du code.
- Eviter les appels shell la ou Python standard est plus simple et plus portable.

## DataFrames

- Une transformation DataFrame doit laisser un contrat de colonnes clair.
- Eviter les mutations surprises chainees et les assignments opaques.
- Preferer des pipelines par etapes nommees plutot qu'un bloc monolithique de transformations.
- Lorsqu'une logique devient difficile a verifier dans pandas, extraire une fonction intermediaire testable.

## Ressources et contexte

- Utiliser des context managers pour les fichiers, connexions et ressources temporaires.
- Les fichiers temporaires et dossiers temporaires doivent vivre dans un contexte borne.

## Concurrence et parallelisme

- N'ajouter du parallelisme qu'avec une justification mesuree.
- Les fonctions executees en workers doivent etre deterministes et limitees en side effects.
- Toujours garder un mode simple `num_workers=1` facile a deboguer.
