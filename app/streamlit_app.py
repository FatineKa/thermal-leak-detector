# la page web : un formulaire, un bouton, une réponse
# en local, depuis le dossier du projet :  streamlit run app/streamlit_app.py

import requests
import streamlit as st

st.set_page_config(page_title="Estimation DPE", page_icon="🏠", layout="centered")

# couleurs officielles de l'étiquette dpe
couleurs = {"A": "#009036", "B": "#53AF31", "C": "#C8D200", "D": "#FFED00",
            "E": "#FBBA00", "F": "#EB6909", "G": "#E2001A"}
# seuils officiels de consommation (kWh/m²/an) pour chaque lettre
seuils = {"A": "≤ 70", "B": "71 à 110", "C": "111 à 180", "D": "181 à 250",
          "E": "251 à 330", "F": "331 à 420", "G": "> 420"}


def etiquette(choisie=None, probas=None):
    # dessine l'échelle a à g comme sur un vrai dpe ; la lettre choisie est mise en avant
    lignes = []
    for i, (lettre, couleur) in enumerate(couleurs.items()):
        largeur = 35 + i * 9
        texte = f"{round(probas[lettre] * 100)} %" if probas else seuils[lettre] + " kWh/m²/an"
        epaisseur = "3px solid #000" if lettre == choisie else "none"
        fleche = " ◀" if lettre == choisie else ""
        lignes.append(
            f"<div style='display:flex;align-items:center;gap:10px;margin:3px 0'>"
            f"<div style='background:{couleur};width:{largeur}%;padding:4px 10px;font-weight:700;"
            f"color:#000;border:{epaisseur};border-radius:0 20px 20px 0'>{lettre}</div>"
            f"<div style='font-size:14px'>{texte}<b>{fleche}</b></div></div>")
    return "".join(lignes)


# --- en-tête ---
st.title("Estimation de la classe DPE")
st.write(
    "Le **diagnostic de performance énergétique (DPE)** classe un logement de **A** (très économe) "
    "à **G** (très énergivore). Il est obligatoire pour vendre ou louer, et demande la visite d'un "
    "diagnostiqueur. Cet outil donne une **estimation** à partir d'informations que vous connaissez "
    "déjà : surface, année de construction, chauffage…")

with st.expander("Comment ça marche ?", expanded=True):
    st.markdown(etiquette(), unsafe_allow_html=True)
    st.markdown(
        "- **Passoire thermique** : un logement classé **F** ou **G**. Depuis 2025, un logement **G** "
        "ne peut plus être mis en location ; ce sera le cas des **F** en 2028, et des **E** en 2034.\n"
        "- **Le modèle** a appris sur **1,5 million de DPE réels** (base de l'ADEME, toute la France), "
        "avec un algorithme de machine learning (XGBoost).\n"
        "- **Sa fiabilité** : la lettre exacte est juste environ **1 fois sur 2**, et à une lettre près "
        "**9 fois sur 10**. Il ne connaît ni l'isolation, ni les fenêtres, ni les travaux : deux "
        "logements identiques sur le papier peuvent avoir des DPE différents.")

# --- le formulaire ---
# les valeurs doivent être écrites exactement comme dans les données d'entraînement
energies = ["Électricité", "Gaz naturel", "Réseau de Chauffage urbain", "Fioul domestique",
            "Bois – Bûches", "Bois – Granulés (pellets) ou briquettes", "GPL", "Propane"]

st.subheader("Le logement")
col1, col2 = st.columns(2)
type_batiment = col1.selectbox("Maison ou appartement", ["appartement", "maison"])
surface = col2.number_input("Surface (m²)", min_value=9, max_value=500, value=50)
periode = col1.selectbox("Année de construction", [
    "avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988",
    "1989-2000", "2001-2005", "2006-2012", "2013-2021", "après 2021"], index=1)
departement = col2.text_input("Département", value="75")

st.subheader("Le chauffage")
col1, col2 = st.columns(2)
energie = col1.selectbox("Énergie de chauffage", energies)
energie_eau_chaude = col2.selectbox("Énergie de l'eau chaude", energies)
# chauffage individuel ou collectif : seulement pour les appartements (vide pour les maisons)
if type_batiment == "appartement":
    installation = col1.selectbox("Chauffage", ["individuel", "collectif", "mixte (collectif-individuel)"])
else:
    installation = "inconnu"

# questions plus difficiles : facultatives, avec des valeurs habituelles par défaut
with st.expander("Plus de détails (facultatif, pour une estimation plus précise)"):
    col1, col2 = st.columns(2)
    equipement = col1.selectbox("Équipement de chauffage", [
        "je ne sais pas", "gaz", "électrique", "pompe à chaleur", "réseau de chaleur", "fioul", "bois"])
    altitude = col2.selectbox("Altitude", ["inférieur à 400m", "400-800m", "supérieur à 800m"])
    hauteur = col1.number_input("Hauteur sous plafond (m)", min_value=2.2, max_value=5.0, value=2.5, step=0.1)
    niveaux = col2.number_input("Nombre d'étages du logement", min_value=1, max_value=5, value=1)

# --- la réponse ---
if st.button("Estimer la classe", type="primary", use_container_width=True):
    question = {
        "type_batiment": type_batiment, "surface": surface, "periode": periode,
        "departement": departement.strip().upper().zfill(2), "energie": energie,
        "installation": installation, "energie_eau_chaude": energie_eau_chaude,
        "equipement": "inconnu" if equipement == "je ne sais pas" else equipement,
        "hauteur": hauteur, "niveaux": niveaux, "altitude": altitude,
    }
    with st.spinner("Réveil du modèle… (jusqu'à une minute la première fois)"):
        try:
            reponse = requests.post(st.secrets["API_URL"], json=question, timeout=90)
            reponse.raise_for_status()
            resultat = reponse.json()[0]
        except requests.RequestException:
            st.error("Le modèle n'a pas répondu. Réessayez dans un instant.")
            st.stop()

    classe = resultat["classe"]
    probas = resultat["probabilites"]
    risque = round((probas["F"] + probas["G"]) * 100)

    st.divider()
    st.subheader(f"Classe estimée : {classe}")
    st.markdown(etiquette(classe, probas), unsafe_allow_html=True)
    st.write("")
    if risque >= 50:
        st.error(f"**Risque de passoire thermique (F ou G) : {risque} %**")
    elif risque >= 20:
        st.warning(f"**Risque de passoire thermique (F ou G) : {risque} %**")
    else:
        st.success(f"**Risque de passoire thermique (F ou G) : {risque} %**")
    st.caption("Le pourcentage à côté de chaque lettre est la probabilité donnée par le modèle. "
               "Estimation indicative : elle ne remplace pas un vrai DPE.")
