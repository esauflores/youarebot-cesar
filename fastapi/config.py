from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import computed_field


class Config(BaseSettings):
    PORT: int = 8672

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
