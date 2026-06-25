import psycopg2
from psycopg2.extensions import connection as _connection
from uuid import UUID
from config import Config


def get_db_connection() -> _connection:
    return psycopg2.connect(Config().db_url)


def init_db() -> None:
    conn = get_db_connection()
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
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    finally:
        conn.close()


def insert_prediction(
    prediction_id: UUID,
    message_id: UUID,
    dialog_id: UUID,
    text: str,
    participant_index: int,
    is_bot_probability: float,
) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (id, message_id, dialog_id, text, participant_index, is_bot_probability)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (str(prediction_id), str(message_id), str(dialog_id), text, participant_index, is_bot_probability),
            )
        conn.commit()
    finally:
        conn.close()
