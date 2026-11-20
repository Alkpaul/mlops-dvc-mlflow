# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .

RUN --mount=type=secret,id=dagshub_username,env=MLFLOW_TRACKING_USERNAME,required=true \
    --mount=type=secret,id=dagshub_token,env=MLFLOW_TRACKING_PASSWORD,required=true \
    python download_model.py

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]