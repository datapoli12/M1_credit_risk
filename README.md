# Memoire M1 - Risque de credit

Ce projet sert de base de travail pour le memoire M1 (Data Science & IA) sur le risque de credit. La cible est binaire (defaut oui/non) et la modelisation econometrique (logit) reste le point de reference, avec des modeles ML en comparaison.

Structure rapide :
- `data/` : raw immuable, interim pour etapes intermediaires, processed pour les features finales
- `src/` : logique principale (data, models, evaluation, utils, config)
- `artifacts/` : modeles entraines (joblib)
- `notebooks/` : exploration et presentation, pas de logique centrale

Arborescence (MVP) :
```
data/
  raw/
  external/
  interim/
  processed/

src/
  data/
  models/
  evaluation/
  utils/
  config/

artifacts/
notebooks/
reports/
figures/
literature/
environment/
```

Environnement : `M1_credit_risk` et `environment/requirements.txt`.

Le pipeline attendu reste simple : data -> features -> models -> evaluation -> artifacts.
