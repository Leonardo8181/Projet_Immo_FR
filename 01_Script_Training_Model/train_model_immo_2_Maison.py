import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import geopandas as gpd
import mlflow
import os
from datetime import datetime
from dotenv import load_dotenv

#   MLflow Configuration 
load_dotenv()
os.getenv("AWS_ACCESS_KEY_ID")
os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION')
mlflow.set_tracking_uri("https://luleifrance-serveur-mlflow.hf.space")  # URL directe
EXPERIMENT_NAME = "projet-immobilier-fr-Maison"

# Créer ou récupérer l'expérience
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id
mlflow.set_experiment(EXPERIMENT_NAME)

# Chargement du fichier de transactions
df = pd.read_csv('src/transactions_complet.csv')

#  1. Suppression des colonnes inutiles (sauf les surfaces brutes) 
cols_surfaces_brutes = [
    'surface_dependances',
    'surface_locaux_industriels',
    'surface_terrains_agricoles',
    'surface_terrains_sols',
    'surface_terrains_nature'
]

colonnes_a_supprimer = [
    'id_ville',
    'vefa',
    'adresse',
    'id_parcelle_cadastre'
]
df.drop(columns=colonnes_a_supprimer, inplace=True, errors='ignore')

#  2. Conversion de la date et création de annee_mois 
df['date_transaction'] = pd.to_datetime(df['date_transaction'])
df['annee_mois'] = df['date_transaction'].dt.to_period('M').astype(str)

#  3. Filtrage : 5 dernières années + uniquement Maison 
max_date = df['date_transaction'].max()
cutoff_date = max_date - pd.DateOffset(years=5)
df = df[(df['date_transaction'] >= cutoff_date) & (df['type_batiment'] == 'Maison')]
print(f"Après filtrage (5 ans + Maison) : {len(df)} lignes.")

#  4. Normalisation du code postal et du département 
df['code_postal'] = pd.to_numeric(df['code_postal'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(5)
df['departement'] = df['departement'].astype(str).str[:2].str.pad(2, side='left', fillchar='0')
df = df[df['departement'] != '97']   # pas d'outre-mer


# 5. Calcul de la surface du terrain (uniquement à partir de surface_terrains_sols)

def parse_surface_set(val):
    """Extrait la somme des entiers d'une chaîne du type '{123,60}' ou '{}'."""
    if pd.isna(val) or str(val).strip() in ('{}', ''):
        return 0.0
    nums = re.findall(r'\d+', str(val))
    return sum(map(float, nums)) if nums else 0.0

# On ne parse que la colonne qui nous intéresse
if 'surface_terrains_sols' in df.columns:
    df['surface_terrains_sols'] = df['surface_terrains_sols'].apply(parse_surface_set)

# surface_terrain = surface_terrains_sols
df['surface_terrain'] = df['surface_terrains_sols']

# Suppression de toutes les colonnes de surfaces brutes (elles ne sont plus nécessaires)
df.drop(columns=cols_surfaces_brutes, inplace=True, errors='ignore')


# 6. Création du metres carrés de référence – MÉDIANE RÉCENTE PAR VILLE

print("Calcul du metres carrés de référence (médiane des 6 derniers mois, par ville)...")

if 'prix_m2' not in df.columns:
    df['prix_m2'] = df['prix'] / df['surface_habitable']

# Date de référence : la transaction la plus récente du dataset filtré
date_max = df['date_transaction'].max()
date_limite = date_max - pd.DateOffset(months=6)

# Sous‑ensemble des 6 derniers mois
df_recent = df[df['date_transaction'] >= date_limite].copy()

# Médiane du metres carrés par ville sur cette période
ref_ville = df_recent.groupby('code_postal')['prix_m2'].median().reset_index(name='prix_m2_ref_ville')

# Médiane départementale (fallback)
ref_dep = df_recent.groupby('departement')['prix_m2'].median().reset_index(name='prix_m2_ref_dep')

# Fusion avec le DataFrame principal
df = df.merge(ref_ville, on='code_postal', how='left')
df = df.merge(ref_dep, on='departement', how='left')

# Coalescence : ville → département → médiane nationale
df['prix_m2_ref'] = df['prix_m2_ref_ville'].fillna(df['prix_m2_ref_dep'])
df['prix_m2_ref'] = df['prix_m2_ref'].fillna(df['prix_m2'].median())

# Nettoyage des colonnes intermédiaires
df.drop(columns=['prix_m2_ref_ville', 'prix_m2_ref_dep'], inplace=True, errors='ignore')
print("prix_m2_ref (médiane récente par ville) ajouté.")

#  7. Clustering géospatial 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

coords = df[['latitude', 'longitude']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=200, random_state=42, n_init='auto')
df['cluster_geo'] = kmeans.fit_predict(coords_scaled)

centroids = kmeans.cluster_centers_
df['dist_centre_cluster'] = np.sqrt(
    ((coords_scaled - centroids[df['cluster_geo']])**2).sum(axis=1)
)

print("Nombre de transactions par cluster (extrait) :")
print(df['cluster_geo'].value_counts().head(10))

#  8. Sélection des colonnes finales 
colonnes_finales = [
    'prix',
    'n_pieces',
    'surface_habitable',
    'latitude',
    'longitude',
    'departement',
    'prix_m2',
    'prix_m2_ref',
    'cluster_geo',
    'dist_centre_cluster',
    'surface_terrain'
]
df = df[colonnes_finales]

#  9. Nettoyage des valeurs infinies et manquantes 
num_cols = df.select_dtypes(include='number').columns
print("Colonnes avec des infinis :")
for col in num_cols:
    if np.isinf(df[col]).any():
        print(f"{col} : {np.isinf(df[col]).sum()} valeurs infinies")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=num_cols, inplace=True)

print(f"DataFrame final prêt : {df.shape[0]} lignes, {df.shape[1]} colonnes.")


# 8. ENTRAÎNEMENT DU MODÈLE (LightGBM) – MAISON OPTIMISÉE 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import optuna
import joblib

#  GPU si disponible 
try:
    model_test = lgb.LGBMRegressor(device='gpu')
    lgb_device = 'gpu'
    print("LightGBM : GPU activé.")
    max_bin_upper = 200        # accélère sans perte notable
except Exception:
    lgb_device = 'cpu'
    max_bin_upper = 255
    print("LightGBM : CPU.")


# 1. FILTRAGE RENFORCÉ

print("Filtrage quantiles 0.5% – 99.5%")
low_price, high_price = df['prix'].quantile([0.005, 0.995])
low_surf, high_surf = df['surface_habitable'].quantile([0.005, 0.995])
low_ter, high_ter = df['surface_terrain'].quantile([0.005, 0.995])

df = df[
    (df['prix'] >= low_price) & (df['prix'] <= high_price) &
    (df['surface_habitable'] >= low_surf) & (df['surface_habitable'] <= high_surf) &
    (df['surface_terrain'] >= low_ter) & (df['surface_terrain'] <= high_ter)
].copy()

def filter_outliers_iqr(df, group_col, target_col, factor=4.0):
    def _filter(group):
        Q1 = group[target_col].quantile(0.25)
        Q3 = group[target_col].quantile(0.75)
        IQR = Q3 - Q1
        return group[(group[target_col] >= Q1 - factor * IQR) &
                     (group[target_col] <= Q3 + factor * IQR)]
    return df.groupby(group_col, group_keys=False).apply(_filter)

df = filter_outliers_iqr(df, 'departement', 'prix', factor=4.0)
df = filter_outliers_iqr(df, 'departement', 'surface_terrain', factor=4.0)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Lignes après filtrage : {len(df)}")


# 2. FEATURE ENGINEERING (inchangé)

df.drop(columns=['prix_m2'], inplace=True, errors='ignore')

df['surface_par_piece'] = df['surface_habitable'] / df['n_pieces'].replace(0, 1)
df['log_surface'] = np.log1p(df['surface_habitable'])
df['log_surface_terrain'] = np.log1p(df['surface_terrain'])
df['ratio_terrain_habitable'] = df['surface_terrain'] / df['surface_habitable'].replace(0, np.nan)
df['ratio_terrain_habitable'] = df['ratio_terrain_habitable'].fillna(0)
df['log_ratio_terrain_habitable'] = np.log1p(df['ratio_terrain_habitable'])
df['lat_lon_interact'] = df['latitude'] * df['longitude']
df['log_prix_m2_ref'] = np.log1p(df['prix_m2_ref'])

cat_features = ['cluster_geo', 'departement']
for col in cat_features:
    df[col] = df[col].astype(str)

num_features = [
    'n_pieces', 'surface_habitable', 'latitude', 'longitude',
    'prix_m2_ref', 'dist_centre_cluster', 'surface_par_piece',
    'log_surface', 'surface_terrain', 'log_surface_terrain',
    'ratio_terrain_habitable', 'log_ratio_terrain_habitable',
    'lat_lon_interact', 'log_prix_m2_ref'
]

X = df[num_features + cat_features]
y = df['prix']
y_log = np.log1p(y)


# 3. SPLIT TRAIN/TEST

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
print(f"Train : {X_train.shape}, Test : {X_test.shape}")


# 4. PRÉPROCESSEUR RAPIDE POUR OPTUNA (TargetEncoder cv=5)

preprocessor_optuna = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('te', TargetEncoder(target_type='continuous', cv=5, smooth='auto', random_state=42),
         ['departement', 'cluster_geo'])
    ])
preprocessor_optuna.set_output(transform='pandas')


# 5. RECHERCHE OPTUNA (50 essais, accélérée)

n_search = min(150_000, len(X_train))          # sous-échantillon réduit
frac_echantillon = n_search / len(X_train)
X_search, _, y_search, _ = train_test_split(
    X_train, y_train_log, train_size=n_search, random_state=42
)
groups_search = X_search['departement'].values
gkf = GroupKFold(n_splits=3)                   # 3 plis (plus rapide)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 500),   
        'num_leaves': trial.suggest_int('num_leaves', 50, 255),        
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 128, max_bin_upper),
        'random_state': 42, 'n_jobs': -1,
        'device': lgb_device, 'verbosity': -1,
    }
    pipe = Pipeline([
        ('prep', preprocessor_optuna),
        ('model', lgb.LGBMRegressor(**params))
    ])
    scores = cross_val_score(pipe, X_search, y_search,
                             cv=gkf, scoring='r2', groups=groups_search,
                             n_jobs=1)
    return np.mean(scores)

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print("Meilleurs hyperparamètres :", best_params)


# 6. ENTRAÎNEMENT FINAL (robuste, avec cv=10 et early stopping 50)

final_params = best_params.copy()
final_params['n_estimators'] = 2000
final_params.update({'device': lgb_device, 'n_jobs': -1,
                     'random_state': 42, 'verbosity': -1})

preprocessor_final = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('te', TargetEncoder(target_type='continuous', cv=10, smooth='auto', random_state=42),
         ['departement', 'cluster_geo'])
    ])
preprocessor_final.set_output(transform='pandas')

X_tr_final, X_val_final, y_tr_final_log, y_val_final_log = train_test_split(
    X_train, y_train_log, test_size=0.1, random_state=42
)

X_tr_prep = preprocessor_final.fit_transform(X_tr_final, y_tr_final_log)
X_val_prep = preprocessor_final.transform(X_val_final)

model = lgb.LGBMRegressor(**final_params)
model.fit(X_tr_prep, y_tr_final_log,
          eval_set=[(X_val_prep, y_val_final_log)],
          callbacks=[lgb.early_stopping(stopping_rounds=50)])

best_model = Pipeline([('prep', preprocessor_final), ('model', model)])


# 7. ÉVALUATION

y_pred_train_log = best_model.predict(X_train)
y_pred_test_log  = best_model.predict(X_test)
y_pred_train = np.expm1(y_pred_train_log)
y_pred_test  = np.expm1(y_pred_test_log)
y_train_orig = np.expm1(y_train_log)
y_test_orig  = np.expm1(y_test_log)

mae_train = mean_absolute_error(y_train_orig, y_pred_train)
r2_train  = r2_score(y_train_orig, y_pred_train)
mae_test  = mean_absolute_error(y_test_orig, y_pred_test)
r2_test   = r2_score(y_test_orig, y_pred_test)

print("\n=== Performances du modèle LightGBM optimisé (Maisons) ===")
print(f"Train : MAE = {mae_train:,.0f} € | R² = {r2_train:.4f}")
print(f"Test  : MAE = {mae_test:,.0f} € | R² = {r2_test:.4f}")


# 9. SAUVEGARDE DU MODÈLE FINAL COMPLET (Maisons)

import joblib

# ----- Récupération des objets de l'étape 6 (clustering) déjà ajustés -----
# Ces variables sont encore dans la mémoire : scaler, kmeans
# (elles ont été créées sur le dataframe AVANT filtrage des outliers)
scaler_coords = scaler          # StandardScaler sur [latitude, longitude]
kmeans_all = kmeans             # KMeans 200 clusters

# ----- Artefacts pour le calcul de prix_m2_ref -----
# ref_ville et ref_dep proviennent de l'étape 5 (version Maison), toujours disponibles.
# La médiane nationale récente est calculée sur df_recent (étape 5).
mediane_prix_m2 = df_recent['prix_m2'].median()
print(f"Médiane nationale récente (6 derniers mois) : {mediane_prix_m2:.2f} €/m²")

# ----- Regroupement dans un dictionnaire unique -----
modele_final = {
    'model': best_model,                     # Pipeline (preprocessor + LGBM)
    'kmeans': kmeans_all,                    # KMeans 200 clusters
    'scaler_coords': scaler_coords,          # StandardScaler pour coordonnées
    'ref_ville': ref_ville,                  # DataFrame [code_postal, prix_m2_ref_ville]
    'ref_dep': ref_dep,                      # DataFrame [departement, prix_m2_ref_dep]
    'mediane_prix_m2': mediane_prix_m2,      # float
}

# Sauvegarde d’un seul fichier
joblib.dump(modele_final, 'final_model_maisons.pkl')
print("\n Modèle final complet (Maisons) sauvegardé dans 'final_model_maisons.pkl'")



#  MLflow Tracking Partie FIN 
from datetime import datetime  # déjà importé en haut, mais s'assurer qu'il est là

# Nom du run dynamique
run_name = f"LGBM_immo_{datetime.now().strftime('%b%d_%H-%M')}"

with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
    # Log des meilleurs paramètres
    mlflow.log_params({k.replace('model__', ''): v for k, v in best_params.items()})
    mlflow.log_param("frac_echantillon", frac_echantillon)
    mlflow.log_param("nb_clusters", 200)

    # Log des métriques
    mlflow.log_metrics({
        "mae_train": mae_train,
        "r2_train": r2_train,
        "mae_test": mae_test,
        "r2_test": r2_test
    })

    # Log de l'artefact final (modèle complet)
    mlflow.log_artifact('final_model_maisons.pkl', artifact_path="modele_immo")

    print(" Modèle uploadé sur le serveur MLflow.")


print("\n Modèle final complet sauvegardé dans 'final_model_maisons.pkl'")