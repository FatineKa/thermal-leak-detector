# supprime le modèle en ligne (endpoint) : plus rien ne peut coûter d'argent
# pour le remettre en ligne :  python scripts/deploy.py
# à lancer depuis le dossier du projet, après aws login :  python scripts/teardown.py

from sagemaker.predictor import Predictor

predicteur = Predictor(endpoint_name="thermal-leak-detector")
predicteur.delete_model()     # le modèle enregistré dans sagemaker
predicteur.delete_endpoint()  # l'endpoint et sa configuration
print("endpoint supprimé. les données et le modèle entraîné restent dans s3.")
