import pytest

from fastapi import HTTPException
from unittest.mock import MagicMock

import src.dependencies.ai_dependencies as ai
from src.schemas.test_schema import Cookies
from project_tests import fakes


async def test_api_status_error_maps_to_502(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.side_effect = fakes.make_api_status_error(status_code=400)

    with pytest.raises(HTTPException) as exc_info:
        await ai.ai_chat("hi", cookies)

    assert exc_info.value.status_code == 502


async def test_api_connection_error_maps_to_408(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.side_effect = fakes.make_api_connection_error()

    with pytest.raises(HTTPException) as exc_info:
        await ai.ai_chat("hi", cookies)

    assert exc_info.value.status_code == 408


async def test_api_timeout_error_maps_to_408(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.side_effect = fakes.make_api_timeout_error()

    with pytest.raises(HTTPException) as exc_info:
        await ai.ai_chat("hi", cookies)

    assert exc_info.value.status_code == 408


async def test_generic_error_maps_to_500(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.side_effect = ValueError("boom")

    with pytest.raises(HTTPException) as exc_info:
        await ai.ai_chat("hi", cookies)

    assert exc_info.value.status_code == 500


async def test_stream_api_status_error_maps_to_502(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.side_effect = fakes.make_api_status_error(status_code=400)

    with pytest.raises(HTTPException) as exc_info:
        async for _ in ai.ai_chat_stream("hi", cookies):
            pass

    assert exc_info.value.status_code == 502


async def test_stream_generic_error_maps_to_500(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.side_effect = ValueError("boom")

    with pytest.raises(HTTPException) as exc_info:
        async for _ in ai.ai_chat_stream("hi", cookies):
            pass

    assert exc_info.value.status_code == 500