# Projet Immobilier France

## Résumé du projet
Ce projet, réalisé en groupe de 4 personnes dans le cadre de la formation **Data Science & Engineering - Fullstack chez Jedha**, propose une solution complète pour l'estimation du prix immobilier pour la France métropolitaine. Notre approche couvre tout le cycle de vie du projet, depuis l'exploration des données jusqu'à l'application web interactive, en utilisant le Machine Learning pour l'estimation des prix.

## Source des données
Nous utilisons le dataset **Immobilier France** disponible sur Kaggle :  
🔗 [https://www.kaggle.com/datasets/benoitfavier/immobilier-france](https://www.kaggle.com/datasets/benoitfavier/immobilier-france)
Ce dataset regroupe les transactions immobilières de **2014 à 2024** (prix, surface, type de bien, localisation, date de transaction, etc.).

## Architecture du projet
Le projet se compose de 3 parties principales 
### 1. Entraînement des modèles
Deux modèles distincts sont entraînés pour les appartements et les maisons:`train_model_immo_1_Appartement.py` et `train_model_immo_2_Maison.py`
(LightGBM a été choisi pour sa rapidité et sa capacité à gérer efficacement de grands volumes de données tabulaires.)

Chaque script :
- Nettoie les données (suppression des outliers, normalisation des codes postaux).
- Calcule un prix de référence au m² basé sur la médiane des 6 derniers mois par commune.
- Crée des clusters géographiques avec KMeans.
- Optimise les hyperparamètres via Optuna (avec intégration de la méthode EARLY STOPPING pour atteindre les meilleures performances).
- Suit les performances (MAE, R²) sur MLflow.
### 2. API
Une API REST: `API_app.py`
- Charge 2 modèles entrainés depuis AWS S3 et expose API via un endpoint pour les données de la prédiction.
- Géocode les adresses via Nominatim (OpenStreetMap).
- Enrichit chaque estimation avec des données contextuelles (transports, parcs, supermarchés) grâce à l'API Overpass.
- Retourne une prédiction pour les prix détaillés.
### 3. Interface utilisateur
Une application interactive basée sur Streamlit: `STREAMLIT_app.py`
Ceci permet aux utilisateurs de :
- Saisir les caractéristiques d'un bien (adresse, surface, pièces, etc.).
- Afficher l'estimation avec ses métriques détaillées.
- Visualiser la localisation et les points d'intérêt sur une carte interactive.
- Générer automatiquement une annonce immobilière professionnelle via Mistral AI API, accompagnée d'une vue Street View via Google Maps API.



