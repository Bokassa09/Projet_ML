# Projet_ML

## Description

Ce projet vise à prédire si un patient est atteint de diabète ou non à partir d'un ensemble de données médicales. Il utilise des techniques de Machine Learning, telles que **KNN** et **l’arbre de décision**, pour effectuer des prédictions.

## Objectif

L'objectif de ce projet est de **prédire le diabète** à partir d’un ensemble de données médicales en utilisant des algorithmes d’apprentissage supervisé. Il s’agit d’un problème de **classification binaire** (`0 = Pas de diabète`, `1 = Diabète`), basé sur des attributs comme :
- Le taux de glucose
- L’indice de masse corporelle (IMC)
- L'âge du patient
- Le nombre de grossesses, etc.

## Prérequis

Avant de commencer, assurez-vous d’avoir installé les outils suivants :

- **Python 3.x**
- **Un environnement virtuel (`venv`)**
- **Bibliothèques Python** : `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Installation des dépendances :

1️⃣ **Créer et activer un environnement virtuel (`venv`)** :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
venv\Scripts\activate     # Sur Windows

