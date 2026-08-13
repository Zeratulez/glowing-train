import pytest

from unittest.mock import MagicMock

import src.dependencies.ai_dependencies as ai
from src.schemas.test_schema import Cookies
from project_tests import fakes


async def test_chat_returns_metrics(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.return_value = fakes.make_completion(
        text="Привет, ИИ!", prompt_tokens=20, completion_tokens=7
    )

    result = await ai.ai_chat("Привет, ИИ!", cookies)

    assert result.response == "Привет, ИИ!"
    assert result.input_tokens == 20
    assert result.output_tokens == 7
    assert isinstance(result.latency, float)


async def test_chat_stream_yields_text_chunks(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.return_value = fakes.make_stream_iter(
        [
            fakes.make_stream_chunk(text="Hello "),
            fakes.make_stream_chunk(text="world"),
            fakes.make_stream_chunk(usage=fakes.SimpleNamespace(prompt_tokens=5, completion_tokens=3)),
        ]
    )

    chunks = [c async for c in ai.ai_chat_stream("hi", cookies)]

    texts = [c["data"] for c in chunks if c["type"] == "text"]
    assert "".join(texts) == "Hello world"
    assert any(c["type"] == "metrics" for c in chunks)


async def test_extract_returns_metrics(client_mock: MagicMock, cookies: Cookies) -> None:
    parsed = {"text": "extracted"}
    client_mock.chat.completions.parse.return_value = fakes.make_completion(text="raw", parsed=parsed)

    result = await ai.ai_chat_extract("some text", cookies)

    assert result.response.text == "extracted"
    assert result.input_tokens == 10
    assert result.output_tokens == 5