# AGENTS — Mémoire M1 | Risque de crédit

## Contexte académique
Mémoire de Master 1 en DATA SCIENCE & IA.
Thème : Risque de crédit, avec un focus sur le défaut de crédit (variable binaire).

La base de données contient une variable cible indiquant
si un individu a fait défaut ou non (0/1), construite en amont
à partir d’une probabilité de défaut et d’un seuil métier.

## Objectifs du mémoire
- Expliquer les déterminants du défaut de crédit.
- Identifier les variables qui augmentent ou réduisent
  la probabilité de tomber en défaut.
- Prédire le défaut (oui/non) pour de nouvelles observations
  à l’aide de modèles de scoring.

## Périmètre méthodologique (IMPORTANT)
- Le mémoire ne vise PAS la modélisation réglementaire de la PD.
- Pas de calibration IFRS 9, pas de LGD/EAD implémentées.
- La PD continue peut être discutée en littérature,
  mais la modélisation porte exclusivement sur une cible binaire.

## Modélisation autorisée
- Modèles économétriques explicatifs (logit de référence).
- Modèles de machine learning de classification
  (ex. XGBoost, LightGBM) à des fins comparatives.
- Toute approche ML doit être comparée au modèle logit.

## Interprétabilité et justification
- L’interprétation économique est obligatoire.
- Les modèles ML doivent être expliqués (ex. SHAP).
- Chaque choix de variable et de modèle doit être justifié
  théoriquement ou empiriquement.

## Organisation du code 
- src/data : nettoyage des données et construction des variables.
- src/models : estimation des modèles de classification.
- src/evaluation : évaluation des performances (ROC, AUC, KS).
- src/utils : fonctions réutilisables (chargement, métriques).
Les notebooks servent uniquement à l’exploration et à la présentation
des résultats, pas à contenir la logique principale.

## Contraintes techniques
- Python uniquement.
- Utiliser l’environnement M1_credit_risk.
- Ne jamais modifier les données dans data/raw.
- Les résultats doivent être reproductibles.

## Bonnes pratiques
- Un modèle logit de référence est obligatoire.
- Les modèles ML sont utilisés comme compléments comparatifs.
- Les modèles doivent être sauvegardés (joblib).
- Séparer clairement estimation et évaluation.

## Sécurité et discipline
- Demander confirmation avant toute suppression de fichier.
- Ne jamais agir en dehors du dossier du projet.
