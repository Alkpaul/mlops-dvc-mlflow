import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "https://dagshub.com/Alkpaul/mlops-dvc-mlflow.mlflow"
MODEL_NAME = "DigitsRandomForest"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient()
versions = client.search_model_versions(
    filter_string=f"name='{MODEL_NAME}'"
)
ready_versions = [v for v in versions if v.status == "READY"]

if not ready_versions:
    raise RuntimeError("Aucune version du modèle disponible.")

latest = max(ready_versions, key=lambda v: int(v.version))
model_uri = f"models:/{MODEL_NAME}/{latest.version}"
destination = Path(__file__).resolve().parent / "model"

with tempfile.TemporaryDirectory() as temporary_directory:
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=temporary_directory,
    )
    shutil.copytree(downloaded, destination, dirs_exist_ok=True)

print(f"Modèle téléchargé : {model_uri}")