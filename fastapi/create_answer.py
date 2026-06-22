from uuid import UUID
from openai import OpenAI

from config import logger, Config, PRE_PROMPT
from database import select_messages_by_dialog


def query_llm(base_url: str, model: str, system_prompt: str, messages: list[dict]) -> str:
    client = OpenAI(api_key="not-needed", base_url=base_url)
    chat_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = "user" if m["participant_index"] == 0 else "assistant"
        chat_messages.append({"role": role, "content": m["text"]})
    chat_completion = client.chat.completions.create(
        messages=chat_messages,
        model=model,
        temperature=0.7,
        max_tokens=256,
    )
    client.close()
    return chat_completion.choices[0].message.content


def get_response_from_llm(dialog_id: UUID, request_message_text: str) -> str:
    cfg = Config()

    history = []
    if dialog_id:
        try:
            history = select_messages_by_dialog(dialog_id)
        except Exception as e:
            logger.error(f"DB error fetching history: {e}")

    if not history:
        history = [{"text": request_message_text, "participant_index": 0}]

    logger.info(f"Calling LLM at {cfg.LLM_BASE_URL} model={cfg.LLM_MODEL} history_len={len(history)}")

    try:
        response = query_llm(
            base_url=cfg.LLM_BASE_URL,
            model=cfg.LLM_MODEL,
            system_prompt=PRE_PROMPT,
            messages=history,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        response = ""

    if not response:
        logger.warning("Empty LLM response, echoing input")
        response = request_message_text

    return response