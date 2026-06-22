import uvicorn
import psycopg2
import time
from fastapi import FastAPI
from uuid import uuid4, UUID

from config import logger, Config
from models_fast_api import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction
from create_answer import get_response_from_llm
from model_inference import classify_text
from database import insert_message, init_db

app = FastAPI()


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    dialog_id: UUID | None = body.dialog_id

    logger.info(f"Received message: dialog_id={body.dialog_id} text={body.last_msg_text}")

    try:
        insert_message(
            msg_id=body.last_message_id,
            dialog_id=body.dialog_id,
            text=body.last_msg_text,
            participant_index=0,
        )
    except Exception as e:
        logger.error(f"DB insert error (user msg): {e}")
        dialog_id = None

    response_from_llm: str = get_response_from_llm(dialog_id, body.last_msg_text)
    logger.info(f"LLM response: {response_from_llm}")

    try:
        insert_message(
            msg_id=uuid4(),
            dialog_id=body.dialog_id,
            text=response_from_llm,
            participant_index=1,
        )
    except Exception as e:
        logger.error(f"DB insert error (bot msg): {e}")

    return GetMessageResponseModel(
        new_msg_text=response_from_llm, dialog_id=body.dialog_id
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage):
    logger.info(f"Classify: dialog_id={msg.dialog_id} text={msg.text}")
    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=classify_text(msg.text),
    )


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