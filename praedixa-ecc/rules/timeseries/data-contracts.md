# Time Series Data Contracts

- Le grain canonique doit etre explicite et stable.
- Toute table importante doit declarer sa cle de serie, sa granularite et son horizon d'usage.
- Valider presence des colonnes, types attendus, tri temporel et unicite au grain.
- Distinguer demande observee, demande censuree, fermeture, stockout et donnee manquante.
- Ne jamais cacher la target dans l'index.
- Ne jamais merger des tables de grains differents sans agregation explicite.
