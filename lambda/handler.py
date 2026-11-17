# la réceptionniste : reçoit la question du site, la passe au modèle, renvoie la réponse
# le site n'a pas de badge aws ; la lambda, si (role)

import os

import boto3

runtime = boto3.client("sagemaker-runtime")


def lambda_handler(event, context):
    corps = event.get("body") or "{}"  # la question envoyée par le site
    reponse = runtime.invoke_endpoint(
        EndpointName=os.environ["ENDPOINT_NAME"],  # thermal-leak-detector
        ContentType="application/json",
        Body=corps,
    )
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": reponse["Body"].read().decode(),
    }
