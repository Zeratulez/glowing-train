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

    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    FILEPATH: str
    OLLAMA_URL: str

    @property
    def ASYNC_ENGINE_CONNECT(self):
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@db:5432/{self.DB_NAME}"
    
    model_config = SettingsConfigDict(env_file=ENV_DIR / ".env", extra="ignore")

settings = Settings()