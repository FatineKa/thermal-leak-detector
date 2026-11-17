# envoie l'entraînement à sagemaker (training job)
# à lancer depuis le dossier du projet, après aws login :  python scripts/launch_training.py

import sagemaker
from sagemaker.sklearn.estimator import SKLearn

bucket = "sagemaker-thermal-leak-3769"
# le badge que sagemaker utilise pour lire et écrire dans s3 (role)
role = "arn:aws:iam::376943093487:role/service-role/AmazonSageMaker-ExecutionRole-20261116T221626"

estimateur = SKLearn(
    entry_point="train.py",
    source_dir="src",               # envoie tout le dossier src, avec requirements.txt
    role=role,
    instance_type="ml.m5.xlarge",   # la machine louée : 4 processeurs, 16 go de mémoire
    instance_count=1,
    framework_version="1.2-1",      # version de scikit-learn installée sur la machine
    py_version="py3",
    output_path=f"s3://{bucket}/models/",  # où ranger le modèle entraîné
    sagemaker_session=sagemaker.Session(),
)

# lance l'entraînement : chaque canal (channel) devient un dossier sur la machine
donnees = f"s3://{bucket}/2m/clean"  # les 2 millions de logements (processing job)
estimateur.fit({
    "train": f"{donnees}/train/",
    "test": f"{donnees}/test/",
})

print("modèle enregistré ici :", estimateur.model_data)
