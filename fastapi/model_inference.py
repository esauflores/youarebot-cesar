# ponytail: lazy-loading singleton — model loaded on first request
import logging

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)
_classifier = None

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
CANDIDATE_LABELS = ["bot", "human"]
HYPOTHESIS_TEMPLATE = "This message was written by a {}."


def load_model():
    global _classifier
    if _classifier is None:
        logger.info(f"Loading zero-shot pipeline: {MODEL_NAME}")
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            device=-1,
        )
    return _classifier


def classify_text(text: str) -> float:
    classifier = load_model()
    result = classifier(text, candidate_labels=CANDIDATE_LABELS)
    bot_index = result["labels"].index("bot")
    return max(0.0, min(1.0, float(result["scores"][bot_index])))
