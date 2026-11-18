# entraîne le modèle : sagemaker lance ce script sur une machine louée
# test sur ton ordinateur :  python src/train.py

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

classes = list("ABCDEFG")
nombres = ["surface", "niveaux", "hauteur", "apres_2026"]
textes = ["periode", "type_batiment", "energie", "installation", "equipement",
          "energie_eau_chaude", "departement", "altitude"]


def preparer(df):
    # sépare les infos du logement (x) et la réponse en chiffre, a=0 ... g=6 (y)
    return df[nombres + textes], df["classe"].map({c: i for i, c in enumerate(classes)}).to_numpy()


if __name__ == "__main__":
    # réglages trouvés par le réglage automatique de sagemaker (scripts/launch_tuning.py)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--min-child-weight", type=float, default=1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=0.51)
    # part de l'entraînement gardée pour la validation (0 = tout pour apprendre).
    # le réglage automatique (tuning) met 0.2 : il compare sur la validation, jamais sur le test.
    parser.add_argument("--validation", type=float, default=0.0)
    # dossiers : sagemaker les donne dans des variables d'environnement (environment variables).
    # les valeurs après servent quand on lance le script sur son ordinateur.
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "data/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    parser.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "data/test"))
    args = parser.parse_args()

    x_train, y_train = preparer(pd.read_parquet(args.train))
    x_test, y_test = preparer(pd.read_parquet(args.test))
    if args.validation > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=args.validation, stratify=y_train, random_state=42)

    modele = Pipeline([
        ("prep", ColumnTransformer(
            [("texte", OneHotEncoder(handle_unknown="ignore"), textes)],
            remainder="passthrough",
        )),
        ("xgb", XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
        )),
    ])

    # poids adoucis : se tromper sur une classe rare coûte plus cher (class weights)
    poids = np.sqrt(compute_sample_weight("balanced", y_train))
    modele.fit(x_train, y_train, xgb__sample_weight=poids)

    # notes écrites dans les logs : sur la validation pendant le réglage, sinon sur le test
    if args.validation > 0:
        nom, x_note, y_note = "valid", x_valid, y_valid
    else:
        nom, x_note, y_note = "test", x_test, y_test
    pred = modele.predict(x_note)
    print(f"{nom}_exactitude={accuracy_score(y_note, pred):.4f}")
    print(f"{nom}_a_une_classe_pres={np.mean(np.abs(pred - y_note) <= 1):.4f}")
    print(f"{nom}_f1_macro={f1_score(y_note, pred, average='macro'):.4f}")
    print(f"{nom}_passoire_precision={precision_score(y_note >= 5, pred >= 5):.4f}")
    print(f"{nom}_passoire_rappel={recall_score(y_note >= 5, pred >= 5):.4f}")

    # enregistre le modèle : sagemaker l'emballe dans model.tar.gz et l'envoie dans s3
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(modele, os.path.join(args.model_dir, "model.joblib"))
