"""Classifier — bot detection microservice.
Loads ONNX model baked into the image at startup.
"""
import os
import numpy as np
import onnxruntime
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "answerdotai/ModernBERT-base")
ONNX_PATH = Path(os.getenv("ONNX_PATH", "classifier.onnx"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

_tokenizer = None
_session = None
_ready = False


def load_model():
    global _tokenizer, _session, _ready
    if _ready:
        return
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading ONNX: {ONNX_PATH}")
    _session = onnxruntime.InferenceSession(str(ONNX_PATH))
    _ready = True
    print("Classifier ready")


def classify(text: str) -> float:
    load_model()
    enc = _tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=MAX_LENGTH)
    logits = _session.run(
        ["logits"],
        {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)},
    )[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return max(0.0, min(1.0, float(probs[0, 1])))


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    is_bot_probability: float = Field(ge=0.0, le=1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="youarebot-classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ok" if _ready else "not ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return PredictResponse(is_bot_probability=classify(req.text))
