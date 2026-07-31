from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.sse import EventSourceResponse, ServerSentEvent

from src.dependencies.ai_dependencies import ai_chat, ai_chat_extract, ai_chat_stream

router = APIRouter()

message_history = []


@router.post("/chat")
async def chat(prompt: Annotated[str | None, Body()] = ""):
    return await ai_chat(prompt)

@router.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(prompt: Annotated[str | None, Body()] = "") -> AsyncIterable[ServerSentEvent]:
    async for chunk in ai_chat_stream(prompt):
        if chunk["type"] == "text":
            yield ServerSentEvent(data=chunk["data"])
        elif chunk["type"] == "metrics":
            yield ServerSentEvent(data=chunk["data"], event="metrics")

@router.post("/extract")
async def excract(text: Annotated[str, Body()]):
    return await ai_chat_extract(text)