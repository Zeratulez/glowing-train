import functools
import uuid

from typing import Annotated

from asteval import Interpreter
from fastapi import status, HTTPException, Response, Cookie
from openai import APIStatusError, APIConnectionError, APITimeoutError
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from openai.types.completion_usage import CompletionUsage

from src.schemas.test_schema import Metrics, Cookies

aeval = Interpreter()

def is_retryable(exception: Exception):
    if isinstance(exception, APIStatusError):
        return exception.response.status_code >= 500
    return bool(isinstance(exception, (APIConnectionError, APITimeoutError)))

def map_openai_errors(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except APIStatusError as e:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Ошибка на стороне сервиса: {e.message}")
        except (APIConnectionError, APITimeoutError) as e:
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Истекло время ожидания | {e}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка сервера, {e}")
    return wrapper

def map_openai_errors_stream(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            async for chunk in func(*args, **kwargs):
                yield chunk
        except APIStatusError as e:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Ошибка на стороне сервиса: {e.message}")
        except (APIConnectionError, APITimeoutError) as e:
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Истекло время ожидания | {e}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка сервера, {e}")
    return wrapper

def metrics_formatted(response: ChatCompletion | ParsedChatCompletion, response_time: float, parse: bool = False) -> Metrics:
    message = response.choices[0].message
    if parse:
        if message.refusal:
            content = message.refusal
        else:
            content = message.parsed
    else:
        content = message.content
    return Metrics(
        response=content,
        latency=response_time,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        # cached_tokens=response.usage.prompt_tokens_details.cached_tokens
    )

def metrics_formatted_stream(response: CompletionUsage, response_time: float) -> Metrics:
    return Metrics(
        latency=response_time,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        # cached_tokens=response.prompt_tokens_details.cached_tokens
    )

def get_cookie(response: Response, cookies: Annotated[Cookies | None, Cookie()] = None):
    if not cookies or not cookies.session_id:
        new_session_id = str(uuid.uuid4())
        cookies = Cookies(session_id=new_session_id)
        response.set_cookie(key="session_id", value=new_session_id, httponly=True, max_age=3600)
    return cookies

def calculation(expression: str):
    result = aeval(expression)
    if aeval.error:
        raise Exception(f"Ошибка {aeval.error} с сообщением: {aeval.error_msg}")
    return result