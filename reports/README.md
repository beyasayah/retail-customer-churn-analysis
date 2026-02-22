# Atelier Machine Learning – Analyse Comportementale Clientèle Retail

## Titre et description du projet
Ce projet a été réalisé dans le cadre de l’atelier Machine Learning du module GI2.  
L’objectif est d’analyser le comportement des clients d’un site e‑commerce de cadeaux afin de **prédire le churn** (départ client) et de proposer des recommandations marketing.  
La chaîne complète de traitement est mise en œuvre : exploration, préparation, modélisation, évaluation et déploiement.

## Instructions d’installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/beyasayah/project_ml_retail.git
cd project_ml_retail
# Création
python -m venv venv

# Activation sous Windows
venv\Scripts\activate

# Activation sous Linux / Mac
source venv/bin/activate
pip install -r requirements.txt
project_ml_retail/
│
├── data/                     # Toutes les données
│   ├── raw/                  # Données brutes (ne pas modifier)
│   ├── processed/            # Données après nettoyage léger (avant split)
│   └── train_test/           # Données splittées (train/test) prêtes pour la modélisation
│
├── notebooks/                # Notebooks Jupyter
│   ├── 01_exploration_preparation.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelisation.ipynb
│
├── src/                      # Scripts Python réutilisables
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/                   # Modèles et objets de preprocessing sauvegardés
│   ├── preprocessor.pkl
│   └── best_model.pkl
│
├── app/                      # Application web Flask
│   └── app.py
│
├── reports/                  # Visualisations et rapports d’exploration
│   └── exploration_report.xlsx
│
├── requirements.txt          # Dépendances Python
├── README.md                 # Documentation
└── .gitignore                # Fichiers ignorés par Git



Exploration : notebooks/01_exploration_preparation.ipynb
Analyse des données, détection des valeurs manquantes, aberrantes, visualisations.

Préprocessing : notebooks/02_preprocessing.ipynb
Nettoyage avancé, feature engineering, encodage, normalisation, split train/test.
Les données préparées sont sauvegardées dans data/train_test/.