"""Orchestrator — public API gateway.
Routes requests to internal services, stores predictions in PostgreSQL.
"""
import os
from datetime import datetime, timezone
from uuid import uuid4
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import psycopg2

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
LLM_URL = os.getenv("LLM_URL", "http://llm:8080")
DB_URL = os.getenv("DATABASE_URL", "postgresql://student:student_pass@postgres:5432/chat_db")


def get_db():
    return psycopg2.connect(DB_URL)


def init_db():
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id UUID PRIMARY KEY,
                    message_id UUID NOT NULL,
                    dialog_id UUID NOT NULL,
                    text TEXT NOT NULL,
                    participant_index INT NOT NULL,
                    is_bot_probability FLOAT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception:
        pass
    yield

app = FastAPI(title="youarebot-orchestrator", lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    text = body.get("text", "")

    payload = {
        "text": text,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{CLASSIFIER_URL}/predict", json=payload, timeout=30)
    data = resp.json()

    prob = data.get("is_bot_probability", 0.5)
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (id, message_id, dialog_id, text, participant_index, is_bot_probability) VALUES (%s, %s, %s, %s, %s, %s)",
                (str(uuid4()), str(uuid4()), str(uuid4()), text, 0, prob),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return JSONResponse({
        "id": str(uuid4()),
        "message_id": str(uuid4()),
        "dialog_id": str(uuid4()),
        "participant_index": 0,
        "is_bot_probability": prob,
    })


@app.post("/get_message")
async def get_message(request: Request):
    body = await request.json()
    payload = {
        "model": "qwen3.5-0.8b",
        "messages": [
            {"role": "system", "content": "You are a human in a Turing test. Respond naturally and briefly. /no_think"},
            {"role": "user", "content": body.get("last_msg_text", body.get("text", ""))},
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload, timeout=60)
    data = resp.json()
    choice = data["choices"][0]["message"] if "choices" in data else {}
    reply = choice.get("content") or choice.get("reasoning_content") or "I don't know what to say."
    return JSONResponse({
        "new_msg_text": reply,
        "dialog_id": body.get("dialog_id", ""),
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
