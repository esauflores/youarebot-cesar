import uvicorn
import psycopg2
import time
from fastapi import FastAPI
from uuid import uuid4

from config import logger, Config
from models_fast_api import IncomingMessage, Prediction
from model_inference import classify_text
from database import insert_prediction, init_db

app = FastAPI()


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage):
    probability = classify_text(msg.text)
    prediction_id = uuid4()

    try:
        insert_prediction(
            prediction_id=prediction_id,
            message_id=msg.id,
            dialog_id=msg.dialog_id,
            text=msg.text,
            participant_index=msg.participant_index,
            is_bot_probability=probability,
        )
    except Exception as e:
        logger.error(f"DB insert error: {e}")

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=probability,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
def on_startup():
    while True:
        try:
            conn = psycopg2.connect(
                database=Config().DB_NAME,
                user=Config().DB_USER,
                password=Config().DB_PASSWORD,
                host=Config().DB_HOST,
                port=Config().DB_PORT,
            )
            conn.close()
            break
        except psycopg2.OperationalError:
            logger.warning("Waiting for PostgreSQL...")
            time.sleep(2)

    init_db()
    logger.info("FastAPI ready")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config().PORT)
