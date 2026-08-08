from typing import Annotated

from fastapi import APIRouter, Depends, Body
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import async_session
from src.dependencies.rag import prompt_embedding
from src.dependencies.ai_dependencies import ai_chat
from src.dependencies.utils import get_cookie
from src.schemas.test_schema import Cookies

router = APIRouter()

@router.post("/rag/query")
async def chunking_input(session: Annotated[AsyncSession, Depends(async_session)], cookies: Annotated[Cookies, Depends(get_cookie)], prompt: Annotated[str, Body()]):
    embeddings = await prompt_embedding(session, prompt)
    return await ai_chat(prompt, cookies, embeddings)
