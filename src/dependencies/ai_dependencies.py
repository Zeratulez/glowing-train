import time

from fastapi import status
from fastapi.exceptions import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError
from tenacity import retry, retry_if_exception, wait_exponential_jitter

from src.core.ai_settings import client
from src.core.settings import settings
from src.schemas.test_schema import Test_Schema

message_history = []

def is_retryable(exception: Exception):
    if isinstance(exception, APIStatusError):
        return exception.response.status_code >= 500
    return bool(isinstance(exception, (APIConnectionError, APITimeoutError)))

@retry(wait=wait_exponential_jitter(max=5), retry=retry_if_exception(is_retryable), reraise=True)
async def _ai_request(messages: list):
    return await client.chat.completions.create(
                model=settings.AI_MODEL,
                messages=messages,
                extra_body={"reasoning": {"enabled": True}},
                timeout=30
            )

@retry(wait=wait_exponential_jitter(max=5), retry=retry_if_exception(is_retryable), reraise=True)
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
    for attempt in range(attempts):
        try:
            response = await client.chat.completions.parse(
                        model=settings.AI_MODEL,
                        messages=messages,
                        response_format=Test_Schema,
                        timeout=30
                    )
            response.choices[0].message.parsed
            return response
        except ValidationError as e:
                if attempt < attempts - 1:
                    messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    messages.append({"role": "system", "content": f"You previous answer raised validation error, here is error {e.errors()}, fix it"})
                else:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot validate info")

async def ai_chat(prompt: str):
    start_time = time.time()
    current_message = message_history + [{"role": "user", "content": prompt}]
    try:
        response = await _ai_request(current_message)
    except APIStatusError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Ошибка на стороне сервиса: {e.message}")
    except (APIConnectionError, APITimeoutError) as e:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Истекло время ожидания")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка сервера")
    
    ai_response = response.choices[0].message.content
    message_history.append({"role": "user", "content": prompt})
    message_history.append({"role": "assistant", "content": ai_response})

    return {"response":ai_response, "latency": time.time()-start_time, "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cost": ((response.usage.prompt_tokens*5/1_000_000.0)+(response.usage.completion_tokens*30/1_000_000.0))}

async def ai_chat_stream(prompt: str):
    start_time = time.time()
    current_message = message_history + [{"role": "user", "content": prompt}]
    try:
        response = await _ai_request_stream(current_message)
    except APIStatusError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Ошибка на стороне сервиса: {e.message}")
    except (APIConnectionError, APITimeoutError) as e:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Истекло время ожидания")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка сервера")
    
    full_response = ""
    final_usage = None

    try:
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                yield {"type": "text", "data": chunk.choices[0].delta.content}
            if hasattr(chunk, "usage") and chunk.usage is not None:
                final_usage = chunk.usage
    except Exception as e:
        yield {"type": "error", "data": str(e)}

    message_history.append({"role": "user", "content": prompt})
    message_history.append({"role": "assistant", "content": full_response})

    if final_usage:
        yield {"type": "metrics", "data": {"latency": time.time()-start_time, "input_tokens": chunk.usage.prompt_tokens,
        "output_tokens": chunk.usage.completion_tokens,
        "cost": ((chunk.usage.prompt_tokens*5/1_000_000.0)+(chunk.usage.completion_tokens*30/1_000_000.0))}}

async def ai_chat_extract(prompt: str):
    start_time = time.time()
    current_message = message_history + [{"role": "user", "content": prompt}]
    try:
        response = await _ai_request_extract(current_message)
    except APIStatusError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Ошибка на стороне сервиса: {e.message}")
    except (APIConnectionError, APITimeoutError) as e:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Истекло время ожидания")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка сервера")

    ai_response = response.choices[0].message.parsed
    message_history.append({"role": "user", "content": prompt})
    message_history.append({"role": "assistant", "content": ai_response})

    return {"response":ai_response, "latency": time.time()-start_time, "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cost": ((response.usage.prompt_tokens*5/1_000_000.0)+(response.usage.completion_tokens*30/1_000_000.0))}