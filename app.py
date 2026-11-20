import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(openapi_url=None)

# Le modèle sera téléchargé dans l'image pendant le build Docker.
model = mlflow.sklearn.load_model("model")


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=64, max_length=64)


@app.post("/predict")
def predict(data: PredictRequest):
    features = np.asarray([data.features], dtype=float)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}