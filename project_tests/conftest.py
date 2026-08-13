import pytest
from pytest_mock import MockerFixture
from unittest.mock import AsyncMock, MagicMock
from openai import AsyncOpenAI

import src.dependencies.ai_dependencies as ai
from src.schemas.test_schema import Cookies


@pytest.fixture(autouse=True)
def reset_history():
    ai.message_history.clear()
    yield
    ai.message_history.clear()


@pytest.fixture
def client_mock(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=AsyncOpenAI)
    m.chat.completions.create = AsyncMock()
    m.chat.completions.parse = AsyncMock()
    mocker.patch.object(ai, "client", m)
    return m


@pytest.fixture
def cookies() -> Cookies:
    return Cookies(session_id="session-test-1")


@pytest.fixture
def session_mock(mocker: MockerFixture) -> AsyncMock:
    return mocker.AsyncMock()
