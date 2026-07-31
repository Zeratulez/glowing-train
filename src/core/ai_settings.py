from openai import AsyncOpenAI

from src.core.settings import settings

client = AsyncOpenAI(
    base_url=settings.BASE_URL,
    api_key=settings.API_KEY,
)