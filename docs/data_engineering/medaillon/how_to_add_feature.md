# Ajouter Une Feature Gold

## Étapes

1. Vérifier la disponibilité temporelle: la feature est-elle connue à
   `decision_timestamp`?
2. Ajouter la transformation dans le modèle gold le plus proche.
3. Ajouter la ligne correspondante dans `platform/warehouse/seeds/feature_registry.csv`.
4. Définir le rôle TFT: static, known future, observed/history, support ou target.
5. Définir la politique de missing/default.
6. Ajouter un test dbt si la feature peut introduire une fuite temporelle.
7. Vérifier le bundle et le manifest de features.

## Règles

- Une observation publiée après coup ne peut pas devenir une feature connue.
- Une feature constante/all-zero n'est gardée que si `allow_constant=true`.
- Les features météo/macro doivent porter une logique `available_at <= decision_timestamp`.
- Les colonnes support (`location_id`, `product_id`, `lag_1`) ne sont pas
  nécessairement des inputs modèle.
