# Ajouter Une Source

## Étapes

1. Ajouter une entrée dans `platform/warehouse/seeds/source_registry.csv`.
2. Ajouter une `BronzeTableSpec` avec `source_policy_id` et `expected_columns`.
3. Ajouter ou adapter le modèle staging: cast minimal, pas de logique métier.
4. Ajouter un adaptateur silver qui sort le contrat de demande canonique.
5. Joindre la source via `silver_allowed_training_dataset_sources` si elle est
   autorisée pour l'entraînement.
6. Ajouter les tests dbt de grain, non-null, valeurs acceptées et qualité label.
7. Mettre à jour cette documentation si la source change le flux médaillon.

## Règles

- Ne jamais inventer une demande latente si la source fournit seulement des ventes.
- Ne pas masquer une source sous un autre `dataset_source`.
- Ne pas ajouter de logique feature dans staging.
- Toute source open-data doit être bloquée si son provider est en quarantaine.
