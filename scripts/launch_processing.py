# envoie le téléchargement + nettoyage à sagemaker (processing job)
# à lancer depuis le dossier du projet, après aws login :  python scripts/launch_processing.py

import sagemaker
from sagemaker.processing import FrameworkProcessor, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn

bucket = "sagemaker-thermal-leak-3769"
role = "arn:aws:iam::376943093487:role/service-role/AmazonSageMaker-ExecutionRole-20261116T221626"
nb_lignes = 2_000_000
dossier_s3 = f"s3://{bucket}/{nb_lignes // 1_000_000}m"  # par exemple s3://.../2m

processeur = FrameworkProcessor(
    estimator_cls=SKLearn,          # même machine prête à l'emploi que pour l'entraînement
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.xlarge",   # 4 processeurs, 16 go de mémoire
    instance_count=1,
    sagemaker_session=sagemaker.Session(),
)

processeur.run(
    code="prepare_data.py",
    source_dir="scripts",           # envoie aussi download_data.py, clean_data.py et requirements.txt
    arguments=["--nb-lignes", str(nb_lignes)],
    # ce que la machine écrit dans ces dossiers part dans s3 à la fin
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/brut", destination=f"{dossier_s3}/raw"),
        ProcessingOutput(source="/opt/ml/processing/propre", destination=f"{dossier_s3}/clean"),
    ],
)
print("données prêtes ici :", f"{dossier_s3}/clean/")
