# Experiment Integrity

- Commencer par une hypothese falsifiable, pas par un modele favori.
- Definir avant la run : splits, horizon, baselines, metriques, segments critiques et critere de succes.
- Garder le test final intouchable jusqu'a la comparaison finale.
- Ne jamais re-interpreter retroactivement le protocole pour sauver une run.
- Rapporter au minimum une mesure d'erreur, une mesure de biais et une lecture par segment.
- La decision finale doit etre explicite : `keep`, `reject`, `investigate`.
