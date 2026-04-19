# Leakage

- Interdiction du split aleatoire pour un probleme de forecasting.
- Toute feature basee sur la target doit etre strictement issue du passe.
- Toujours `shift(1)` ou equivalent avant une rolling statistic sur la target.
- Les scalers, encoders et imputers apprennent sur train uniquement.
- Les variables exogenes futures ne sont autorisees que si elles sont reellement connues au moment de la decision.
- Un stockout ne doit pas etre appris comme une faible demande.
