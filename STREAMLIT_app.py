import os
import textwrap
from typing import Dict, Any, Optional, Tuple

import folium
import requests
import streamlit as st
from streamlit_folium import folium_static
from langchain_mistralai import ChatMistralAI


# Configuration & secrets (variables d'environnement)

API_URL = "https://luleifrance-api-immobilier-france.hf.space/predict"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY", "")

DEPARTEMENTS: Dict[str, str] = {
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
    '93': 'Seine-Saint-Denis', '94': 'Val-de-Marne', '95': "Val-d'Oise"
}

PREFIX_TO_COORDS = {
    "75": (48.856614, 2.352222),
    "69": (45.767299, 4.834329),
    "13": (43.296950, 5.369890),
    "31": (43.604462, 1.444247),
    "59": (50.631189, 3.069479),
}


# Fonctions utilitaires

def get_street_view_iframe(lat: float, lon: float, width: int = 1100, height: int = 500,
                           heading: int = 0, pitch: int = 25, fov: int = 100) -> str:
    if not GOOGLE_API_KEY:
        return ""
    src = (
        f"https://www.google.com/maps/embed/v1/streetview"
        f"?key={GOOGLE_API_KEY}"
        f"&location={lat},{lon}"
        f"&heading={heading}"
        f"&pitch={pitch}"
        f"&fov={fov}"
    )
    return (
        f'<iframe width="{width}" height="{height}" '
        f'style="border:0; border-radius:8px;" '
        f'loading="lazy" allowfullscreen '
        f'referrerpolicy="no-referrer-when-downgrade" '
        f'src="{src}"></iframe>'
    )

def geocode_mairie(nom_mairie: str, code_postal: str) -> Optional[Tuple[float, float]]:
    if not code_postal:
        return None
    prefix = code_postal[:2]
    if prefix in PREFIX_TO_COORDS:
        return PREFIX_TO_COORDS[prefix]
    query = f"Mairie de {nom_mairie}, {code_postal}, France" if nom_mairie.strip() else f"mairie {code_postal}, France"
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "EstimationApp/1.0"},
            timeout=5,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None

def format_distance(m: float) -> str:
    return f"{m:.0f} m" if m < 1000 else f"{m/1000:.1f} km"

def build_annonce_prompt(payload: dict, data: dict) -> str:
    c = data['contexte']
    d = data
    transports = "\n".join(f"- {t['nom']} [{t['type']}] (à proximité)" for t in c.get("transports", [])) or "Aucun"
    parcs = "\n".join(f"- {p['nom']} [{p['type']}] (à proximité)" for p in c.get("parcs", [])) or "Aucun"
    superm = f"- {c['supermarche']['nom']} (à proximité)" if c.get("supermarche") else "Aucun"
    dist_str = format_distance(c.get('distance_centre_ville')) if c.get('distance_centre_ville') else "Non disponible"
    return textwrap.dedent(f"""
    Rédige une annonce immobilière attrayante en français de **strictement 300 à 350 mots maximum**.
    Adresse : {payload['adresse']}
    Type : {payload['type_bien']} | Surface : {payload['surface']} m² | Pièces : {payload['n_pieces']}
    Terrain : {d.get('surface_terrain',0)} m²
    Prix estimé : {d['estimation']:,.0f} € | Prix/m² : {d['prix_m2_estime']:.2f} €/m²
    Réf. local : {d['prix_m2_ref']:.2f} €/m² | Écart : {d['ecart']:+.1f}%
    Distance centre-ville : {dist_str} | Mairie : {c.get('nom_mairie','')}
    Transports : {transports}
    Parcs : {parcs}
    Supermarché : {superm}
    Annonce engageante et professionnelle.
    """).strip()

def create_map(lat: float, lon: float, adresse: str, estimation: float,
               contexte: dict, mairie_coords=None, nom_mairie="", dist_centre=None) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker(
        [lat, lon], popup=f"{adresse}<br>Estimation : {estimation:,.0f} €",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)
    if mairie_coords:
        popup_text = f"<b>Mairie :</b> {nom_mairie}<br>"
        if dist_centre:
            popup_text += f"<b>Distance centre‑ville :</b> {format_distance(dist_centre)}<br>"
        # La ligne "Vol d'oiseau" a été supprimée
        folium.Marker(
            mairie_coords,
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color="darkblue", icon="building", prefix="fa")
        ).add_to(m)
    for t in contexte.get("transports", []):
        folium.CircleMarker(
            [t["lat"], t["lon"]], radius=5, color="blue", fill=True,
            popup=f"{t['nom']} ({t['distance_m']} m)"
        ).add_to(m)
    for p in contexte.get("parcs", []):
        folium.CircleMarker(
            [p["lat"], p["lon"]], radius=5, color="green", fill=True,
            popup=f"{p['nom']} ({p['distance_m']} m)"
        ).add_to(m)
    if contexte.get("supermarche"):
        s = contexte["supermarche"]
        folium.CircleMarker(
            [s["lat"], s["lon"]], radius=5, color="orange", fill=True,
            popup=f"{s['nom']} ({s['distance_m']} m)"
        ).add_to(m)
    return m


# État persistant

st.set_page_config(page_title="Estimation Immobilière France – Prix & Annonce", page_icon="🏠", layout="wide")
st.markdown("<style>.section-title{font-size:1.3rem;font-weight:600;margin-top:1.5rem;border-bottom:2px solid #f0f0f0;padding-bottom:0.2rem;}</style>", unsafe_allow_html=True)

# Initialisations
st.session_state.setdefault("estimation_data", None)
st.session_state.setdefault("annonce", None)
st.session_state.setdefault("last_payload", None)
st.session_state.setdefault("generating", False)
st.session_state.setdefault("surface_terrain_input", 0.0)

st.title("🏠 Outil d'Estimation Immobilière France")
st.markdown("Estimez le prix d'un bien immobilier en France métropolitaine, découvrez le contexte local et générez une annonce avec l'IA.")

if not MISTRAL_API_KEY:
    st.sidebar.warning("Clé API Mistral non configurée. Ajoutez-la dans les variables d'environnement.")

# --- Formulaire (sidebar) ---
with st.sidebar:
    st.header("🔍 Caractéristiques du bien")
    with st.form("form_estimation", clear_on_submit=False):
        adresse = st.text_input("Adresse*", key="adresse_input", placeholder="ex: 01 Rue de Rivoli, Paris")
        col1, col2 = st.columns([1, 1])
        with col1:
            code_postal = st.text_input("Code postal*", key="cp_input", placeholder="75004")
        with col2:
            type_bien = st.selectbox("Type de bien*", ["Appartement", "Maison"], key="type_input")
        n_pieces = st.number_input("Nombre de pièces*", min_value=1, value=2, step=1, key="pieces_input")
        surface = st.number_input("Surface habitable (m²)*", min_value=1.0, value=50.0, step=1.0, key="surface_input")
        terrain_label = "Surface de terrain (m²)" + ("*" if type_bien == "Maison" else "")
        surface_terrain = st.number_input(
            terrain_label, min_value=0.0, step=1.0,
            key="surface_terrain_input",
            help="Surface totale de la parcelle (m²) incluant jardin, cour, garage, piscine, dépendances, etc. Obligatoire (> 0) pour une maison. Laisser 0 pour un appartement."
        )
        submitted = st.form_submit_button("💰 Estimer le prix", use_container_width=True)

# --- Traitement de la soumission ---
if submitted:
    adresse_val = st.session_state.adresse_input.strip()
    cp_val = st.session_state.cp_input.strip()
    type_val = st.session_state.type_input
    terrain_val = st.session_state.surface_terrain_input

    errors = []
    if not adresse_val:
        errors.append("L'adresse est obligatoire.")
    if not cp_val:
        errors.append("Le code postal est obligatoire.")
    if type_val == "Maison" and terrain_val <= 0:
        errors.append("Pour une maison, la surface de terrain doit être > 0 m².")
    if type_val == "Appartement" and terrain_val > 0:
        errors.append("Pour un appartement, la surface de terrain doit être = 0 m².")

    if errors:
        for e in errors:
            st.sidebar.error(e)
        st.stop()
    else:
        payload = {
            "adresse": adresse_val,
            "type_bien": type_val,
            "surface": st.session_state.surface_input,
            "n_pieces": st.session_state.pieces_input,
            "surface_terrain": terrain_val,
        }
        if cp_val:
            payload["code_postal"] = cp_val

        with st.spinner("Calcul de l'estimation en cours..."):
            try:
                resp = requests.post(API_URL, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                elif resp.status_code == 422:
                    detail = resp.json().get("detail", "Erreur de validation.")
                    st.error(detail)
                    st.stop()
                elif resp.status_code == 404:
                    st.error("Adresse introuvable. Vérifiez l'adresse et le code postal.")
                    st.stop()
                else:
                    st.error(f"Erreur API (code {resp.status_code}) : {resp.text}")
                    st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"Impossible de joindre l'API : {e}")
                st.stop()

        st.session_state.estimation_data = data
        st.session_state.last_payload = payload
        st.session_state.annonce = None
        st.rerun()


# Affichage des résultats

if st.session_state.estimation_data:
    data = st.session_state.estimation_data
    payload = st.session_state.last_payload
    lat, lon = data["lat"], data["lon"]
    contexte = data["contexte"]
    terrain = data.get("surface_terrain", 0)

    cp = payload.get("code_postal", "").strip() or "00000"
    departement = DEPARTEMENTS.get(cp[:2], f"Département {cp[:2]}")
    mairie_nom = contexte.get("nom_mairie", "")
    mairie_coords = geocode_mairie(mairie_nom, cp)
    dist_centre = contexte.get("distance_centre_ville")

    st.markdown("---")
    st.markdown("### 📋 Récapitulatif du bien")
    cols = st.columns(4)
    cols[0].metric(" Estimation", f"{data['estimation']:,.0f} €")
    cols[1].metric(" Prix/m² estimé", f"{data['prix_m2_estime']:.2f} €")
    cols[2].metric(" Prix/m² local", f"{data['prix_m2_ref']:.2f} €")
    cols[3].metric(" Écart marché", f"{data['ecart']:+.1f} %",
                   delta=f"{data['ecart']:+.1f} %" if data['ecart'] != 0 else None,
                   delta_color="normal" if data['ecart'] >= 0 else "inverse")

    with st.expander("📄 Détails du bien", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Adresse :** {payload['adresse']}")
            st.write(f"**Département :** {departement}")
            st.write(f"**Code Postal :** {cp}")
            st.write(f"**Type :** {payload['type_bien']}")
        with c2:
            st.write(f"**Surface habitable :** {payload['surface']} m²")
            st.write(f"**Pièces :** {payload['n_pieces']}")
            if payload['type_bien'] == "Maison":
                st.write(f"**Terrain :** {terrain} m²")

    st.markdown('<div class="section-title">🏙️ Contexte local</div>', unsafe_allow_html=True)
    if dist_centre:
        st.info(f"📍 Distance à {mairie_nom} (centre‑ville) : **{format_distance(dist_centre)}**")
    else:
        st.info("📍 Centre-ville : information non disponible")

    ctx = st.columns(3)

    with ctx[0]:
        st.markdown("🚌 **Transports**")
        if contexte.get("transports"):
            transports_filtres = [t for t in contexte["transports"] if t["type"] != "Autre"]
            if transports_filtres:
                for t in transports_filtres:
                    st.markdown(f"- {t['nom']} ({t['distance_m']} m) [{t['type']}]")
            else:
                st.markdown("Aucun arrêt trouvé.")
        else:
            st.markdown("Aucun arrêt trouvé.")

    with ctx[1]:
        st.markdown("🌳 **Parcs / Jardins**")
        if contexte.get("parcs"):
            # Exclure les espaces verts
            parcs_filtres = [p for p in contexte["parcs"] if p["type"] != "Espace vert"]
            if parcs_filtres:
                for p in parcs_filtres:
                    st.markdown(f"- {p['nom']} ({p['distance_m']} m) [{p['type']}]")
            else:
                st.markdown("Aucun parc trouvé.")
        else:
            st.markdown("Aucun parc trouvé.")

    with ctx[2]:
        st.markdown("🛒 **Supermarché à proximité**")
        supermarche_data = contexte.get("supermarche")
        if supermarche_data:
            # Si l'API renvoie une liste, on prend les 2 premiers
            if isinstance(supermarche_data, list):
                top_supers = sorted(supermarche_data, key=lambda x: x["distance_m"])[:2]
                for s in top_supers:
                    st.markdown(f"- {s['nom']} ({s['distance_m']} m)")
            elif isinstance(supermarche_data, dict):
                # Si un seul supermarché est retourné (objet unique)
                st.markdown(f"- {supermarche_data['nom']} ({supermarche_data['distance_m']} m)")
            else:
                st.markdown("Aucun supermarché trouvé.")
        else:
            st.markdown("Aucun supermarché trouvé.")

    st.markdown('<div class="section-title">🗺️ Localisation et points d\'intérêt</div>', unsafe_allow_html=True)
    m = create_map(lat, lon, payload['adresse'], data['estimation'], contexte, mairie_coords, mairie_nom, dist_centre)
    folium_static(m, width=1100, height=500)

    # --- Annonce IA ---
    st.markdown('<div class="section-title">📝 Annonce immobilière générée par IA</div>', unsafe_allow_html=True)
    if not MISTRAL_API_KEY:
        st.info("Clé Mistral non configurée.")
    else:
        if st.session_state.annonce and not st.session_state.generating:
            st.markdown(st.session_state.annonce)
            if GOOGLE_API_KEY:
                st.components.v1.html(get_street_view_iframe(lat, lon, 1100, 500, pitch=25, fov=100), height=520, scrolling=False)
                st.caption("Vue Street View interactive – vous pouvez tourner, incliner et zoomer.")
            else:
                st.warning("Clé Google API absente – activez l'API Maps Embed pour la vue interactive.")
        if st.button("✨ Générer une annonce", disabled=st.session_state.generating):
            st.session_state.generating = True
            st.session_state.annonce = None
            st.rerun()
        if st.session_state.generating:
            prompt = build_annonce_prompt(payload, data)
            llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY, temperature=0.7, max_tokens=700, streaming=True)
            placeholder = st.empty()
            with placeholder.container():
                st.markdown("**Génération en cours...**")
                msg = st.empty()
                full = ""
                for chunk in llm.stream(prompt):
                    full += chunk.content
                    msg.markdown(full + "▌")
                msg.markdown(full)
            st.session_state.annonce = full
            st.session_state.generating = False
            st.rerun()