# API FASTAPI, Modèles chargés depuis AWS S3 (version simplifiée)
import pandas as pd
import numpy as np
import requests
import joblib
import os
import boto3
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
import uvicorn


# 1. Configuration S3 et chargement des modèles
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
S3_KEY_APPARTEMENT = os.environ["S3_KEY_APPARTEMENT"]
S3_KEY_MAISON = os.environ["S3_KEY_MAISON"]
LOCAL_APPARTEMENT_PATH = "final_model_appartements.pkl"
LOCAL_MAISON_PATH = "final_model_maisons.pkl"

MODEL_FILES = {
    "Appartement": (S3_KEY_APPARTEMENT, LOCAL_APPARTEMENT_PATH),
    "Maison": (S3_KEY_MAISON, LOCAL_MAISON_PATH),
}

MODEL_DATA = {}
s3_client = boto3.client("s3")

for type_bien, (s3_key, local_path) in MODEL_FILES.items():
    if not os.path.exists(local_path):
        print(f"Téléchargement du modèle {type_bien} depuis s3://{S3_BUCKET}/{s3_key} ...")
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
    pack = joblib.load(local_path)
    MODEL_DATA[type_bien] = {
        "model": pack["model"],
        "kmeans": pack["kmeans"],
        "scaler_coords": pack["scaler_coords"],
        "ref_ville": pack["ref_ville"],
        "ref_dep": pack["ref_dep"],
        "mediane_prix_m2": pack["mediane_prix_m2"],
    }


# 2. Dictionnaire des départements
NOM_DEPARTEMENT = {
    '01': 'Ain', '02': 'Aisne', '03': 'Allier', '04': 'Alpes-de-Haute-Provence',
    '05': 'Hautes-Alpes', '06': 'Alpes-Maritimes', '07': 'Ardèche', '08': 'Ardennes',
    '09': 'Ariège', '10': 'Aube', '11': 'Aude', '12': 'Aveyron', '13': 'Bouches-du-Rhône',
    '14': 'Calvados', '15': 'Cantal', '16': 'Charente', '17': 'Charente-Maritime',
    '18': 'Cher', '19': 'Corrèze', '2A': 'Corse-du-Sud', '2B': 'Haute-Corse',
    '21': "Côte-d'Or", '22': "Côtes-d'Armor", '23': 'Creuse', '24': 'Dordogne',
    '25': 'Doubs', '26': 'Drôme', '27': 'Eure', '28': 'Eure-et-Loir', '29': 'Finistère',
    '30': 'Gard', '31': 'Haute-Garonne', '32': 'Gers', '33': 'Gironde', '34': 'Hérault',
    '35': 'Ille-et-Vilaine', '36': 'Indre', '37': 'Indre-et-Loire', '38': 'Isère',
    '39': 'Jura', '40': 'Landes', '41': 'Loir-et-Cher', '42': 'Loire', '43': 'Haute-Loire',
    '44': 'Loire-Atlantique', '45': 'Loiret', '46': 'Lot', '47': 'Lot-et-Garonne',
    '48': 'Lozère', '49': 'Maine-et-Loire', '50': 'Manche', '51': 'Marne', '52': 'Haute-Marne',
    '53': 'Mayenne', '54': 'Meurthe-et-Moselle', '55': 'Meuse', '56': 'Morbihan',
    '57': 'Moselle', '58': 'Nièvre', '59': 'Nord', '60': 'Oise', '61': 'Orne',
    '62': 'Pas-de-Calais', '63': 'Puy-de-Dôme', '64': 'Pyrénées-Atlantiques',
    '65': 'Hautes-Pyrénées', '66': 'Pyrénées-Orientales', '67': 'Bas-Rhin', '68': 'Haut-Rhin',
    '69': 'Rhône', '70': 'Haute-Saône', '71': 'Saône-et-Loire', '72': 'Sarthe',
    '73': 'Savoie', '74': 'Haute-Savoie', '75': 'Paris', '76': 'Seine-Maritime',
    '77': 'Seine-et-Marne', '78': 'Yvelines', '79': 'Deux-Sèvres', '80': 'Somme',
    '81': 'Tarn', '82': 'Tarn-et-Garonne', '83': 'Var', '84': 'Vaucluse',
    '85': 'Vendée', '86': 'Vienne', '87': 'Haute-Vienne', '88': 'Vosges',
    '89': 'Yonne', '90': 'Territoire de Belfort', '91': 'Essonne', '92': 'Hauts-de-Seine',
    '93': 'Seine-Saint-Denis', '94': 'Val-de-Marne', '95': "Val-d'Oise",
    '971': 'Guadeloupe', '972': 'Martinique', '973': 'Guyane', '974': 'La Réunion',
    '976': 'Mayotte'
}


# 3. Fonctions de contexte (centre-ville, Overpass)
_cache_centre = {}

def get_centre_ville(code_postal: str):
    """Retourne le nom de la mairie et les coordonnées du centre pour un code postal."""
    cp = code_postal.strip().zfill(5)
    if cp in _cache_centre:
        return _cache_centre[cp]

    url = f"https://geo.api.gouv.fr/communes?codePostal={cp}&fields=nom,centre&format=json&limit=1"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                commune = data[0]
                centre = commune.get("centre")
                if centre:
                    nom_mairie = f"Mairie de {commune['nom']}"
                    lat_c, lon_c = centre["coordinates"][1], centre["coordinates"][0]
                    _cache_centre[cp] = (nom_mairie, (lat_c, lon_c))
                    return _cache_centre[cp]
    except Exception:
        pass
    return "Mairie inconnue", (None, None)

def get_infos_proches(lat, lon, rayon_m=5000):
    query = f"""
    [out:json][timeout:30];
    (
      node["highway"="bus_stop"](around:{rayon_m},{lat},{lon});
      node["railway"="tram_stop"](around:{rayon_m},{lat},{lon});
      node["railway"="station"](around:{rayon_m},{lat},{lon});
      node["station"="subway"](around:{rayon_m},{lat},{lon});
      node["public_transport"="stop_position"](around:{rayon_m},{lat},{lon});
      way["leisure"="park"](around:{rayon_m},{lat},{lon});
      way["leisure"="garden"](around:{rayon_m},{lat},{lon});
      way["landuse"="recreation_ground"](around:{rayon_m},{lat},{lon});
      relation["leisure"="park"](around:{rayon_m},{lat},{lon});
      node["shop"="supermarket"](around:{rayon_m},{lat},{lon});
    );
    out center tags;
    """
    transports, parcs, supermarches = [], [], []
    try:
        resp = requests.post("https://overpass-api.de/api/interpreter",
                             data={"data": query},
                             headers={"User-Agent": "ImmoPrediction/1.0",
                                      "Accept": "application/json",
                                      "Content-Type": "text/plain"},
                             timeout=30)
        if resp.status_code != 200:
            return [], [], []
        data = resp.json()
        for el in data.get("elements", []):
            if "center" in el:
                pt = (el["center"]["lat"], el["center"]["lon"])
            elif "lat" in el and "lon" in el:
                pt = (el["lat"], el["lon"])
            else:
                continue
            dist = geodesic((lat, lon), pt).meters
            tags = el.get("tags", {})
            nom = tags.get("name", "").strip()
            if not nom:
                continue

            if any(k in tags for k in ("highway", "railway", "station", "public_transport")):
                if tags.get("station") == "subway" or tags.get("subway") == "yes":
                    typ = "Métro"
                elif tags.get("railway") == "station" or tags.get("train") == "yes":
                    typ = "Train / RER"
                elif tags.get("railway") == "tram_stop" or tags.get("tram") == "yes":
                    typ = "Tramway"
                elif tags.get("highway") == "bus_stop" or tags.get("bus") == "yes":
                    typ = "Bus"
                elif "train" in str(tags.get("public_transport", "")):
                    typ = "Train / RER"
                elif "tram" in str(tags.get("public_transport", "")):
                    typ = "Tramway"
                elif "bus" in str(tags.get("public_transport", "")):
                    typ = "Bus"
                else:
                    typ = "Autre"
                transports.append({"nom": nom, "distance_m": round(dist), "type": typ,
                                   "lat": pt[0], "lon": pt[1]})
            elif tags.get("leisure") in ("park", "garden") or tags.get("landuse") == "recreation_ground":
                if nom.lower().startswith("square") or tags.get("leisure") == "square":
                    continue
                nom_lower = nom.lower()
                if "jardin" in nom_lower:
                    typ_p = "Jardin"
                elif "parc" in nom_lower:
                    typ_p = "Parc"
                elif tags.get("leisure") == "park":
                    typ_p = "Parc"
                elif tags.get("leisure") == "garden":
                    typ_p = "Jardin"
                elif tags.get("landuse") == "recreation_ground":
                    typ_p = "Espace vert"
                else:
                    typ_p = "Autre"
                parcs.append({"nom": nom, "distance_m": round(dist), "type": typ_p,
                              "lat": pt[0], "lon": pt[1]})
            elif tags.get("shop") in ("supermarket", "convenience"):
                supermarches.append({"nom": nom, "distance_m": round(dist),
                                     "lat": pt[0], "lon": pt[1]})

        transports.sort(key=lambda x: x["distance_m"])
        parcs.sort(key=lambda x: x["distance_m"])
        supermarches.sort(key=lambda x: x["distance_m"])
    except Exception:
        pass
    return transports, parcs, supermarches


# 4. Fonctions métier unifiées
geolocator = Nominatim(user_agent="immo_pred")

def geocode(adresse: str, code_postal_user: Optional[str] = None):
    """Géocode une adresse et retourne latitude, longitude, code postal, département."""
    location = geolocator.geocode(adresse)
    if not location:
        raise ValueError("Adresse introuvable.")
    lat, lon = location.latitude, location.longitude
    if code_postal_user:
        cp_str = str(code_postal_user).strip().zfill(5)
    else:
        cp_str = str(location.raw.get("address", {}).get("postcode", "")).zfill(5)
        if cp_str == "00000":
            raise ValueError("Code postal introuvable, utilisez le paramètre code_postal.")
    dept = cp_str[:2]
    return lat, lon, cp_str, dept


def estimer_bien(adresse: str, surface: float, n_pieces: int,
                 code_postal: Optional[str], type_bien: str,
                 surface_terrain: float = 0.0):
    """Estimation commune aux appartements et maisons."""
    lat, lon, cp_str, dept = geocode(adresse, code_postal)
    md = MODEL_DATA[type_bien]
    scaler, kmeans, model = md["scaler_coords"], md["kmeans"], md["model"]
    ref_ville, ref_dep, mediane = md["ref_ville"], md["ref_dep"], md["mediane_prix_m2"]

    # Cluster géographique et distance au centre du cluster
    coord = scaler.transform([[lat, lon]])
    cluster = kmeans.predict(coord)[0]
    dist_cluster = np.linalg.norm(coord - kmeans.cluster_centers_[cluster])

    # Prix de référence
    ligne_ville = ref_ville[ref_ville["code_postal"] == cp_str]
    if not ligne_ville.empty:
        prix_ref = ligne_ville["prix_m2_ref_ville"].iloc[0]
    else:
        ligne_dep = ref_dep[ref_dep["departement"] == dept.zfill(2)]
        prix_ref = ligne_dep["prix_m2_ref_dep"].iloc[0] if not ligne_dep.empty else mediane

    # Construction des features communes
    row = {
        "n_pieces": n_pieces,
        "surface_habitable": surface,
        "latitude": lat,
        "longitude": lon,
        "prix_m2_ref": prix_ref,
        "dist_centre_cluster": dist_cluster,
        "surface_par_piece": surface / max(n_pieces, 1),
        "log_surface": np.log1p(surface),
        "cluster_geo": str(cluster),
        "departement": dept.zfill(2),
    }

    # Features supplémentaires pour les maisons
    if type_bien == "Maison":
        row.update({
            "surface_terrain": surface_terrain,
            "log_surface_terrain": np.log1p(surface_terrain),
            "ratio_terrain_habitable": surface_terrain / max(surface, 1),
            "log_ratio_terrain_habitable": np.log1p(surface_terrain / max(surface, 1)),
            "lat_lon_interact": lat * lon,
            "log_prix_m2_ref": np.log1p(prix_ref),
        })

    X_new = pd.DataFrame([row])
    log_prix = model.predict(X_new)[0]
    prix_estime = np.expm1(log_prix)

    # Contexte enrichi
    nom_mairie, (lat_cv, lon_cv) = get_centre_ville(cp_str)
    d_cv = geodesic((lat, lon), (lat_cv, lon_cv)).meters if lat_cv else None
    transports, parcs, supermarches = get_infos_proches(lat, lon)

    # Meilleur par type de transport / parc
    best_t = {}
    for t in transports:
        typ = t["type"]
        if typ not in best_t or t["distance_m"] < best_t[typ]["distance_m"]:
            best_t[typ] = t
    best_p = {}
    for p in parcs:
        typ = p["type"]
        if typ not in best_p or p["distance_m"] < best_p[typ]["distance_m"]:
            best_p[typ] = p

    return {
        "adresse": adresse,
        "departement": NOM_DEPARTEMENT.get(dept.zfill(2), f"Département {dept}"),
        "code_postal": cp_str,
        "type_bien": type_bien,
        "surface": surface,
        "n_pieces": n_pieces,
        "surface_terrain": surface_terrain if type_bien == "Maison" else 0,
        "prix_estime": prix_estime,
        "prix_ref": prix_ref,
        "nom_mairie": nom_mairie,
        "dist_centre_ville": d_cv,
        "transports": best_t,
        "parcs": best_p,
        "supermarche": supermarches[0] if supermarches else None,
        "lat": lat,
        "lon": lon,
    }

# 5. Modèles de données pour l'API
class BienRequest(BaseModel):
    adresse: str = Field(..., description="Adresse du bien")
    type_bien: Literal["Appartement", "Maison"] = Field(..., description="Type de bien")
    surface: float = Field(..., gt=0, description="Surface habitable en m²")
    n_pieces: int = Field(..., gt=0, description="Nombre de pièces")
    code_postal: Optional[str] = Field(None, description="Code postal (ex: 75008)")
    surface_terrain: Optional[float] = Field(0.0, ge=0, description="Surface du terrain en m² (obligatoire pour une maison)")

    @validator("surface_terrain")
    def verifier_surface_terrain(cls, v, values):
        if values.get("type_bien") == "Maison" and (v is None or v <= 0):
            raise ValueError("Pour une Maison, vous devez fournir une surface_terrain > 0")
        return v

class TransportOut(BaseModel):
    nom: str
    distance_m: int
    type: str
    lat: float
    lon: float

class ParcOut(BaseModel):
    nom: str
    distance_m: int
    type: str
    lat: float
    lon: float

class SupermarcheOut(BaseModel):
    nom: str
    distance_m: int
    lat: float
    lon: float

class ContexteOut(BaseModel):
    distance_centre_ville: Optional[float]
    nom_mairie: str
    transports: List[TransportOut]
    parcs: List[ParcOut]
    supermarche: Optional[SupermarcheOut]

class EstimationOut(BaseModel):
    estimation: float
    prix_m2_estime: float
    prix_m2_ref: float
    ecart: float
    surface_terrain: Optional[float] = 0.0
    contexte: ContexteOut
    lat: float
    lon: float


# 6. Application FastAPI
app = FastAPI(title="Estimation immobilière combinée (S3)")

@app.post("/predict", response_model=EstimationOut)
async def estimer(bien: BienRequest):
    try:
        info = estimer_bien(
            adresse=bien.adresse,
            surface=bien.surface,
            n_pieces=bien.n_pieces,
            code_postal=bien.code_postal,
            type_bien=bien.type_bien,
            surface_terrain=bien.surface_terrain if bien.type_bien == "Maison" else 0.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne")

    prix_m2_estime = info["prix_estime"] / info["surface"]
    ecart = ((prix_m2_estime - info["prix_ref"]) / info["prix_ref"]) * 100

    transports_out = [TransportOut(**t) for t in info["transports"].values()] if info["transports"] else []
    parcs_out = [ParcOut(**p) for p in info["parcs"].values()] if info["parcs"] else []
    supermarche_out = SupermarcheOut(**info["supermarche"]) if info["supermarche"] else None

    contexte = ContexteOut(
        distance_centre_ville=info["dist_centre_ville"],
        nom_mairie=info["nom_mairie"],
        transports=transports_out,
        parcs=parcs_out,
        supermarche=supermarche_out,
    )

    return EstimationOut(
        estimation=round(info["prix_estime"], 2),
        prix_m2_estime=round(prix_m2_estime, 2),
        prix_m2_ref=round(info["prix_ref"], 2),
        ecart=round(ecart, 1),
        surface_terrain=info["surface_terrain"],
        contexte=contexte,
        lat=info["lat"],
        lon=info["lon"],
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)