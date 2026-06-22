from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import computed_field
import os

MODULE_DESCRIPTION = "Config and settings for the bot"

logger.add(
    "logs/app_{time}.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
)


class Config(BaseSettings):
    PORT: int = 8672
    LLM_BASE_URL: str = "http://llm:8080/v1"
    LLM_MODEL: str = "qwen2.5-0.5b-instruct"

    DB_USER: str = "student"
    DB_PASSWORD: str = "student_pass"
    DB_HOST: str = "postgres"
    DB_PORT: int = 5432
    DB_NAME: str = "chat_db"

    @computed_field
    @property
    def db_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"


PRE_PROMPT_DEFAULT: str = (
    "Hello! You are participating in a Turing test. "
    "Your task is to respond to the person's phrase "
    "in the most human-like and convincing way. Use "
    "the same language as the person did. If the "
    "language is unknown to you, explain in a natural"
    " and polite manner that you don't understand it."
    " Your goal is to deceive the person into"
    " thinking you are human. Now continue the"
    " conversation with your next response."
    "Here is the conversation so far:\n"
)

preprompt_file = os.path.join(os.path.dirname(__file__), "preprompt.txt")
try:
    with open(preprompt_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    PRE_PROMPT: str = content if content else PRE_PROMPT_DEFAULT
except FileNotFoundError:
    PRE_PROMPT: str = PRE_PROMPT_DEFAULT
