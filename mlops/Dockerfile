FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow \
    boto3 \
    pandas \
    scikit-learn

COPY . /app

CMD ["python", "mlops/train_classifier.py"]
