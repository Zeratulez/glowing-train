from openai import AsyncOpenAI

from src.core.settings import settings

client = AsyncOpenAI(
    base_url="http://host.docker.internal:11434/v1",
    api_key="ollama",
)