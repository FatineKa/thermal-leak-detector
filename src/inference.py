# le modèle en ligne : sagemaker appelle ces 4 fonctions, toujours dans cet ordre
# 1. charger le modèle  2. lire la question  3. prédire  4. formuler la réponse

import json
import os

import joblib
import pandas as pd

classes = list("ABCDEFG")


def model_fn(model_dir):
    # appelée une seule fois, au démarrage
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(corps, content_type):
    # transforme le json reçu en tableau
    if content_type != "application/json":
        raise ValueError(f"type de contenu non géré : {content_type}")
    donnees = json.loads(corps)
    if isinstance(donnees, dict):  # un seul logement
        donnees = [donnees]
    df = pd.DataFrame(donnees)
    # le site note les logements avec la règle d'aujourd'hui
    df["apres_2026"] = 1
    # si l'utilisateur ne sait pas, on met « inconnu », comme à l'entraînement
    for colonne in ["installation", "equipement"]:
        if colonne not in df:
            df[colonne] = "inconnu"
        df[colonne] = df[colonne].fillna("inconnu")
    return df


def predict_fn(df, modele):
    # garde seulement les colonnes que le modèle a apprises, dans le même ordre
    probas = modele.predict_proba(df[modele.feature_names_in_])
    reponses = []
    for p in probas:
        classe = classes[int(p.argmax())]
        reponses.append({
            "classe": classe,
            "probabilites": {c: round(float(x), 3) for c, x in zip(classes, p)},
            "passoire": classe in ("F", "G"),
        })
    return reponses


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
