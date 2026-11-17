# sur la machine sagemaker (processing job) : télécharge puis nettoie, l'un après l'autre
# sagemaker récupère ensuite tout ce qui est dans /opt/ml/processing/ et l'envoie dans s3

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--nb-lignes", default="2000000")
args = parser.parse_args()

dossier = os.path.dirname(os.path.abspath(__file__))
brut = "/opt/ml/processing/brut/dpe_sample.parquet"
os.makedirs(os.path.dirname(brut), exist_ok=True)

subprocess.run([sys.executable, f"{dossier}/download_data.py",
                "--nb-lignes", args.nb_lignes, "--sortie", brut], check=True)
subprocess.run([sys.executable, f"{dossier}/clean_data.py",
                "--entree", brut, "--sortie", "/opt/ml/processing/propre"], check=True)
