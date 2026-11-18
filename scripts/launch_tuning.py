# réglage automatique des hyperparamètres par sagemaker (automatic model tuning)
# à lancer depuis le dossier du projet, après aws login :  python scripts/launch_tuning.py

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter

bucket = "sagemaker-thermal-leak-3769"
role = "arn:aws:iam::376943093487:role/service-role/AmazonSageMaker-ExecutionRole-20261116T221626"
donnees = f"s3://{bucket}/2m/clean"

estimateur = SKLearn(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    output_path=f"s3://{bucket}/tuning/",
    # 20 % de l'entraînement gardés pour comparer les réglages : le test n'est jamais regardé
    hyperparameters={"validation": 0.2},
    sagemaker_session=sagemaker.Session(),
)

tuner = HyperparameterTuner(
    estimateur,
    objective_metric_name="valid_f1_macro",  # la note à rendre la plus haute possible
    objective_type="Maximize",
    # où trouver cette note dans les logs de train.py
    metric_definitions=[{"Name": "valid_f1_macro", "Regex": "valid_f1_macro=([0-9.]+)"}],
    # les plages dans lesquelles sagemaker cherche
    hyperparameter_ranges={
        "n-estimators": IntegerParameter(200, 1000),
        "max-depth": IntegerParameter(4, 10),
        "learning-rate": ContinuousParameter(0.02, 0.2, scaling_type="Logarithmic"),
        "min-child-weight": IntegerParameter(1, 30),
        "subsample": ContinuousParameter(0.6, 1.0),
        "colsample-bytree": ContinuousParameter(0.5, 1.0),
    },
    strategy="Bayesian",       # chaque essai tient compte des précédents
    max_jobs=20,               # 20 essais au total
    max_parallel_jobs=4,       # 4 en même temps
)

tuner.fit({"train": f"{donnees}/train/", "test": f"{donnees}/test/"}, wait=False)
print("réglage lancé :", tuner.latest_tuning_job.name)
