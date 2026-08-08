import asyncio
import time

from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import ValidationError
from openai import LengthFinishReasonError
from tenacity import retry, retry_if_exception, wait_exponential_jitter, stop_after_attempt

from src.core.ai_settings import client
from src.core.settings import settings
from src.schemas.test_schema import Test_Schema, Cookies, Metrics
from src.models.chunks import Chunk
from src.dependencies.utils import is_retryable, map_openai_errors, map_openai_errors_stream, metrics_formatted, metrics_formatted_stream

message_history: dict[str, list] = {}

sem = asyncio.Semaphore(3)

@retry(wait=wait_exponential_jitter(max=5), retry=retry_if_exception(is_retryable), stop=stop_after_attempt(3), reraise=True)
async def _ai_request(messages: list):
    async with sem:
        return await client.chat.completions.create(
                    model=settings.AI_MODEL,
                    messages=messages,
                    reasoning_effort="medium",
                    timeout=30
                )

@retry(wait=wait_exponential_jitter(max=5), retry=retry_if_exception(is_retryable), stop=stop_after_attempt(3), reraise=True)
async def _ai_request_stream(messages: list):
    return await client.chat.completions.create(
        model=settings.AI_MODEL,
        messages=messages,
        extra_body={"reasoning": {"enabled": True}},
        stream=True,
        stream_options={"include_usage": True},
        timeout=30
    )

async def _ai_request_extract(messages: list):
    attempts = 2
    messages.append({"role": "system", "content": "extract data from text"})
    messages = messages.copy()
    response = None
    for attempt in range(attempts):
        try:
            async with sem:
                response = await client.chat.completions.parse(
                            model=settings.AI_MODEL,
                            messages=messages,
                            response_format=Test_Schema,
                            timeout=30
                        )
            return response
        except (ValidationError, LengthFinishReasonError) as e:
                if attempt < attempts - 1 and response is not None:
                    messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    messages.append({"role": "system", "content": f"You previous answer raised error, here is error {e}, fix it"})
                else:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot validate info")
        except Exception:
            raise

@map_openai_errors
async def ai_chat(prompt: str, cookies: Cookies, embeddings: list[tuple[Chunk, float]] | None = None) -> Metrics:
    start_time = time.time()
    history = message_history.setdefault(cookies.session_id, [])
    if embeddings:
        context_list = []
        chunks = []
        for idx, embedding in enumerate(embeddings):
            context_list.append(f"[{idx+1}] {embedding[0].content}")
            chunks.append(f"[{idx+1}] {embedding[0].content} - {embedding[1]}")
        context = "\n\n".join(context_list)
        current_message = history + [{"role": "system", "content": "отвечай только на основе предоставленного контекста, "
                                    "если ответа нет в контексте — так и скажи"},
                                    {"role": "user", "content": f"Контекст: {context}\nПромпт: {prompt}"}]
    else:
        current_message = history + [{"role": "user", "content": prompt}]
                                
    response = await _ai_request(current_message)
    
    metrics = metrics_formatted(response, time.time()-start_time)
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": metrics.response})

    return {"response": metrics, "chunks": chunks} if embeddings else metrics

@map_openai_errors_stream
async def ai_chat_stream(prompt: str, cookies: Cookies):
    start_time = time.time()
    history = message_history.setdefault(cookies.session_id, [])
    current_message = history + [{"role": "user", "content": prompt}]

    response = await _ai_request_stream(current_message)
    
    full_response = ""
    final_usage = None

    try:
        async with sem:
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    yield {"type": "text", "data": chunk.choices[0].delta.content}
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    final_usage = chunk.usage
    except Exception as e:
        yield {"type": "error", "data": str(e)}

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": full_response})

    if final_usage:
        metrics = metrics_formatted_stream(final_usage, time.time()-start_time)
        yield {"type": "metrics", "data": metrics}

@map_openai_errors
async def ai_chat_extract(prompt: str, cookies: Cookies):
    start_time = time.time()
    history = message_history.setdefault(cookies.session_id, [])
    current_message = history + [{"role": "user", "content": prompt}]

    response = await _ai_request_extract(current_message)

    metrics = metrics_formatted(response, time.time()-start_time, parse=True)
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": metrics.response})

    return metrics