# Backtesting

- Le protocole par defaut est `rolling-origin` ou `walk-forward`.
- Un backtest doit simuler la production : meme horizon, meme information disponible, meme cadence de reentrainement ou de refresh.
- Toujours comparer a au moins une baseline naive ou seasonal naive pertinente.
- Evaluer sur plusieurs folds, pas une seule fenetre.
- Les resultats doivent etre rapportes par horizon et par segment business important.
- Un gain offline ne vaut que s'il est stable sur plusieurs folds et coherent avec le cout business.
