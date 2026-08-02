import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_DIR = pathlib.Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    BASE_URL: str
    API_KEY: str
    AI_MODEL: str

    INPUT_CACHE_HIT: float = 0.30
    INPUT_CACHE_MISS: float = 3.00
    OUTPUT_TOKENS: float = 15.00

    model_config = SettingsConfigDict(env_file=ENV_DIR / ".env", extra="ignore")

settings = Settings()