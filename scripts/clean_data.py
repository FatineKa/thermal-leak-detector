# nettoie l'échantillon et le coupe en deux : entraînement et test
# à lancer depuis le dossier du projet :  python scripts/clean_data.py
# entrée : data/dpe_sample.parquet
# résultats : data/train/train.parquet et data/test/test.parquet

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# réglages qu'on peut changer au lancement (arguments)
parser = argparse.ArgumentParser()
parser.add_argument("--entree", default="data/dpe_sample.parquet")  # fichier brut
parser.add_argument("--sortie", default="data")  # dossier qui recevra train/ et test/
args = parser.parse_args()

brut = pd.read_parquet(args.entree)
print(f"{len(brut):,} logements au départ")


# regroupe les 130 types d'équipement de chauffage en 7 familles simples
def famille_chauffage(texte):
    if pd.isna(texte):
        return "inconnu"
    t = texte.lower()
    if "pac" in t or "pompe" in t:
        return "pompe à chaleur"
    if "fioul" in t:
        return "fioul"
    if "gaz" in t or "gpl" in t:
        return "gaz"
    if any(mot in t for mot in ["bois", "poêle", "insert", "foyer", "cuisinière"]):
        return "bois"
    if any(mot in t for mot in ["électrique", "joule", "convecteur"]):
        return "électrique"
    if "réseau" in t:
        return "réseau de chaleur"
    return "autre"


# garde les colonnes utiles, avec des noms simples
df = pd.DataFrame({
    "classe": brut["etiquette_dpe"],
    "surface": brut["surface_habitable_logement"],
    "periode": brut["periode_construction"],
    "type_batiment": brut["type_batiment"],
    "energie": brut["type_energie_principale_chauffage"],
    "installation": brut["type_installation_chauffage"].fillna("inconnu"),
    "equipement": brut["type_generateur_chauffage_principal"].map(famille_chauffage),
    "energie_eau_chaude": brut["type_energie_principale_ecs"].replace("Non affecté", "inconnu"),
    "departement": brut["code_departement_ban"],
    "altitude": brut["classe_altitude"],
    "niveaux": brut["nombre_niveau_logement"],
    "hauteur": brut["hauteur_sous_plafond"],
    # changement de règle du 1er janvier 2026 : 1 = après, 0 = avant
    "apres_2026": (pd.to_datetime(brut["date_etablissement_dpe"]) >= "2026-01-01").astype(int),
    # date du dpe : pas utilisée par le modèle, sert aux tests dans le temps
    "date_dpe": pd.to_datetime(brut["date_etablissement_dpe"]),
})

# enlève les immeubles entiers : on prédit pour un seul logement
df = df[df["type_batiment"] != "immeuble"]

# enlève les valeurs impossibles (9 m² et 2,20 m : minimums légaux pour louer)
df = df[df["surface"].between(9, 500)]
df = df[df["hauteur"].between(2.2, 5)]
df = df[df["niveaux"].between(1, 5)]
df = df[df["altitude"] != "Non affecté"]

# enlève les lignes où il manque encore quelque chose (missing values)
df = df.dropna()
print(f"{len(df):,} logements après nettoyage")

# coupe : 80 % entraînement (train), 20 % test, avec la même part de chaque classe
train, test = train_test_split(df, test_size=0.2, stratify=df["classe"], random_state=42)

os.makedirs(f"{args.sortie}/train", exist_ok=True)
os.makedirs(f"{args.sortie}/test", exist_ok=True)
train.to_parquet(f"{args.sortie}/train/train.parquet", index=False)
test.to_parquet(f"{args.sortie}/test/test.parquet", index=False)
print(f"entraînement : {len(train):,} logements, test : {len(test):,} logements")
