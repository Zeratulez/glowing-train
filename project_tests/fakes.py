from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, APIStatusError, LengthFinishReasonError

from types import SimpleNamespace


def make_request(url: str = "http://test/v1/chat/completions") -> httpx.Request:
    return httpx.Request("POST", url)


def make_response(status_code: int, request: httpx.Request | None = None, json_body: Any = None) -> httpx.Response:
    return httpx.Response(status_code, request=request or make_request(), json=json_body)


def make_completion(
    text: str = "",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "test-model",
    parsed: Any = None,
    refusal: str | None = None,
) -> Any:
    message = SimpleNamespace(content=text, parsed=parsed, refusal=refusal)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def make_stream_chunk(text: str | None = None, usage: Any = None) -> Any:
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice] if text else [], usage=usage)


def make_stream_iter(chunks: list[Any]) -> AsyncIterator[Any]:
    async def gen() -> AsyncIterator[Any]:
        for c in chunks:
            yield c

    return gen()


def make_api_status_error(status_code: int = 500, message: str = "boom") -> APIStatusError:
    request = make_request()
    response = make_response(status_code, request=request)
    return APIStatusError(message, response=response, body=None)


def make_api_connection_error() -> APIConnectionError:
    return APIConnectionError(request=make_request())


def make_api_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=make_request())


def make_length_finish_error() -> LengthFinishReasonError:
    return LengthFinishReasonError(completion=make_completion())