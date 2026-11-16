# entraîne le modèle : sagemaker lance ce script sur une machine louée
# test sur ton ordinateur :  python src/train.py

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    # réglages trouvés dans notebooks/03_reglages.ipynb (hyperparameters)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-weight", type=float, default=20)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    # dossiers : sagemaker les donne dans des variables d'environnement (environment variables).
    # les valeurs après servent quand on lance le script sur son ordinateur.
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "data/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    parser.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "data/test"))
    args = parser.parse_args()

    x_train, y_train = preparer(pd.read_parquet(args.train))
    x_test, y_test = preparer(pd.read_parquet(args.test))

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

    # notes sur le test, écrites dans les logs
    pred = modele.predict(x_test)
    print(f"test_exactitude={accuracy_score(y_test, pred):.4f}")
    print(f"test_a_une_classe_pres={np.mean(np.abs(pred - y_test) <= 1):.4f}")
    print(f"test_f1_macro={f1_score(y_test, pred, average='macro'):.4f}")
    print(f"test_passoire_precision={precision_score(y_test >= 5, pred >= 5):.4f}")
    print(f"test_passoire_rappel={recall_score(y_test >= 5, pred >= 5):.4f}")

    # enregistre le modèle : sagemaker l'emballe dans model.tar.gz et l'envoie dans s3
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(modele, os.path.join(args.model_dir, "model.joblib"))
