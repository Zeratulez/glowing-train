import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_DIR = pathlib.Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    BASE_URL: str
    API_KEY: str
    AI_MODEL: str

    model_config = SettingsConfigDict(env_file=ENV_DIR / ".env", extra="ignore")

settings = Settings()