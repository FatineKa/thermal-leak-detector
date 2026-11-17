# vérifie automatiquement que src/inference.py marche
# à lancer depuis le dossier du projet :  python -m pytest tests

import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, "src")
import inference  # noqa: E402
from train import nombres, textes  # noqa: E402

logement = {
    "surface": 120, "periode": "1948-1974", "type_batiment": "maison",
    "energie": "Fioul domestique", "energie_eau_chaude": "Fioul domestique",
    "departement": "23", "altitude": "400-800m", "niveaux": 2, "hauteur": 2.5,
}


@pytest.fixture(scope="module")
def modele():
    # un tout petit modèle, entraîné en une seconde : le vrai modèle n'est pas dans le dépôt
    hasard = np.random.default_rng(0)
    n = 70
    df = pd.DataFrame({
        "surface": hasard.uniform(9, 500, n), "niveaux": hasard.integers(1, 6, n),
        "hauteur": hasard.uniform(2.2, 5, n), "apres_2026": hasard.integers(0, 2, n),
        "periode": "1948-1974", "type_batiment": "maison", "energie": "Fioul domestique",
        "installation": "inconnu", "equipement": "inconnu",
        "energie_eau_chaude": "Fioul domestique", "departement": "23", "altitude": "400-800m",
    })
    y = np.arange(n) % 7  # les 7 classes, de a à g
    m = Pipeline([
        ("prep", ColumnTransformer([("texte", OneHotEncoder(handle_unknown="ignore"), textes)],
                                   remainder="passthrough")),
        ("xgb", XGBClassifier(n_estimators=5)),
    ])
    return m.fit(df[nombres + textes], y)


def test_un_logement_donne_une_ligne():
    df = inference.input_fn(json.dumps(logement), "application/json")
    assert len(df) == 1


def test_plusieurs_logements():
    df = inference.input_fn(json.dumps([logement, logement, logement]), "application/json")
    assert len(df) == 3


def test_regle_d_aujourd_hui():
    df = inference.input_fn(json.dumps(logement), "application/json")
    assert df["apres_2026"].iloc[0] == 1


def test_reponses_manquantes_deviennent_inconnu():
    df = inference.input_fn(json.dumps(logement), "application/json")
    assert df["installation"].iloc[0] == "inconnu"
    assert df["equipement"].iloc[0] == "inconnu"


def test_refuse_un_autre_format():
    with pytest.raises(ValueError):
        inference.input_fn("surface=120", "text/csv")


def test_la_reponse_est_coherente(modele):
    df = inference.input_fn(json.dumps(logement), "application/json")
    reponse = inference.predict_fn(df, modele)[0]
    assert reponse["classe"] in "ABCDEFG"
    assert abs(sum(reponse["probabilites"].values()) - 1) < 0.01
    assert reponse["passoire"] == (reponse["classe"] in ("F", "G"))


def test_colonnes_en_trop_ignorees(modele):
    df = inference.input_fn(json.dumps({**logement, "couleur_volets": "bleu"}), "application/json")
    inference.predict_fn(df, modele)  # ne doit pas planter


def test_la_sortie_est_du_json(modele):
    df = inference.input_fn(json.dumps(logement), "application/json")
    texte, type_contenu = inference.output_fn(inference.predict_fn(df, modele), "application/json")
    assert type_contenu == "application/json"
    assert json.loads(texte)[0]["classe"] in "ABCDEFG"
