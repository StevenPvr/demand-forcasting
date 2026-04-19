# Python Testing

## Cadre du repo

- Rester coherent avec le repo : aujourd'hui les tests sont ecrits en `unittest`.
- Ne pas introduire un second style de test sans benefice clair et accord explicite.
- Les tests doivent tourner sans reseau et sans dependance a un environnement externe fragile.

## Principes

- Un test doit verifier un comportement, pas reproduire toute l'implementation.
- Un test doit etre deterministe, petit et lisible.
- Les jeux de donnees de test doivent etre minimaux mais plausibles metier.
- Preferer plusieurs tests cibles a un test geant qui couvre tout mal.

## Priorites pour ce repo

- prevention du leakage
- generation correcte des lags et rolling windows
- splits temporels explicites
- validation de schema et unicite au grain
- metriques erreur + bias
- cas metier : stockout, fermeture, zero demande, intermittent demand, promo

## Fixtures

- Construire des DataFrames miniatures mais realistes.
- Inclure des dates explicites pour rendre les erreurs temporelles visibles.
- Tester les cas limites : debut de serie, trous de donnees, valeurs manquantes, colonnes absentes, doublons.

## Style

- Un nom de test doit exprimer le comportement attendu.
- Un test doit laisser clair :
  - le setup
  - l'action
  - l'assertion
- Eviter les asserts trop vagues ou purement structurelles si le comportement important peut etre teste directement.

## Regression tests

- Tout bug corrige dans le pipeline doit idealement laisser un test de regression.
- Quand une erreur venait d'un leakage ou d'un mauvais alignement temporel, le test doit proteger explicitement contre cette rechute.
