import pytest

from fastapi import HTTPException
from unittest.mock import MagicMock

import src.dependencies.ai_dependencies as ai
from src.schemas.test_schema import Cookies
from project_tests import fakes


async def test_retry_on_retryable_then_succeeds(client_mock: MagicMock) -> None:
    completion = fakes.make_completion(text="ok")
    client_mock.chat.completions.create.side_effect = [
        fakes.make_api_connection_error(),
        fakes.make_api_connection_error(),
        completion,
    ]

    result = await ai._ai_request([{"role": "user", "content": "hi"}])

    assert client_mock.chat.completions.create.call_count == 3
    assert result is completion


async def test_retry_stops_after_max_attempts(client_mock: MagicMock) -> None:
    client_mock.chat.completions.create.side_effect = fakes.make_api_connection_error()

    with pytest.raises(Exception) as exc_info:
        await ai._ai_request([{"role": "user", "content": "hi"}])

    assert client_mock.chat.completions.create.call_count == 3
    assert isinstance(exc_info.value, Exception)


async def test_retry_not_triggered_on_4xx(client_mock: MagicMock) -> None:
    client_mock.chat.completions.create.side_effect = fakes.make_api_status_error(status_code=400)

    with pytest.raises(Exception):
        await ai._ai_request([{"role": "user", "content": "hi"}])

    assert client_mock.chat.completions.create.call_count == 1


async def test_retry_triggered_on_5xx_then_succeeds(client_mock: MagicMock) -> None:
    completion = fakes.make_completion(text="ok")
    client_mock.chat.completions.create.side_effect = [
        fakes.make_api_status_error(status_code=500),
        completion,
    ]

    result = await ai._ai_request([{"role": "user", "content": "hi"}])

    assert client_mock.chat.completions.create.call_count == 2
    assert result is completion


async def test_extract_length_finish_error_maps_to_400(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.parse.side_effect = fakes.make_length_finish_error()

    with pytest.raises(HTTPException) as exc_info:
        await ai.ai_chat_extract("text", cookies)

    assert exc_info.value.status_code == 400


async def test_extract_success_returns_parsed_metrics(client_mock: MagicMock, cookies: Cookies) -> None:
    parsed = {"text": "parsed value"}
    client_mock.chat.completions.parse.return_value = fakes.make_completion(text="raw", parsed=parsed)

    metrics = await ai.ai_chat_extract("text", cookies)

    assert metrics.response.text == "parsed value"
    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 5