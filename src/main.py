from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.endpoints import chat, rag_embed
from src.dependencies.rag import file_get
from src.database import AsyncSessionLocal
from src.core.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSessionLocal() as session:
        filepath = Path(settings.FILEPATH).resolve()
        await file_get(session, filepath)
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)
app.include_router(rag_embed.router)