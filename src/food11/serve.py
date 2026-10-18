"""Serve the registered food11 model behind a FastAPI endpoint."""

import io
import os

import mlflow
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

MODEL_URI = "models:/food11@champion"
IMAGE_SIZE = (128, 128)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(title="food11")
model = None


@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model = mlflow.pyfunc.load_model(MODEL_URI)


def preprocess(raw):
    image = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMAGE_SIZE)
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return array.transpose(2, 0, 1)[None, :]


def softmax(logits):
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logits = np.asarray(model.predict(preprocess(await file.read())))[0]
    probabilities = softmax(logits)
    index = int(probabilities.argmax())
    return {"category": CATEGORIES[index], "confidence": float(probabilities[index])}
