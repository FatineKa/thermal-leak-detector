# met le modèle en ligne : un point de terminaison sans serveur (serverless endpoint)
# à lancer depuis le dossier du projet, après aws login :  python scripts/deploy.py

from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::376943093487:role/service-role/AmazonSageMaker-ExecutionRole-20261116T221626"
# le modèle entraîné sur 1,5 million de logements, réglages trouvés par sagemaker (training job)
modele_s3 = ("s3://sagemaker-thermal-leak-3769/models/"
             "sagemaker-scikit-learn-2026-11-18-00-19-40-557/output/model.tar.gz")

modele = SKLearnModel(
    model_data=modele_s3,
    role=role,
    entry_point="inference.py",  # les 4 fonctions qui répondent aux questions
    source_dir="src",            # envoie aussi requirements.txt (xgboost)
    framework_version="1.2-1",   # même version que pour l'entraînement
    py_version="py3",
)

predicteur = modele.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=2048,  # 2 go de mémoire
        max_concurrency=1,       # une seule question à la fois : protège la facture
    ),
    endpoint_name="thermal-leak-detector",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

# premier essai avec un logement inventé
print(predicteur.predict({
    "surface": 120, "periode": "1948-1974", "type_batiment": "maison",
    "energie": "Fioul domestique", "installation": "individuel", "equipement": "fioul",
    "energie_eau_chaude": "Fioul domestique", "departement": "23",
    "altitude": "400-800m", "niveaux": 2, "hauteur": 2.5,
}))
