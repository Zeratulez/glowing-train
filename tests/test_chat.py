import pytest
import pathlib
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", cookies={"session_id": "test_cookie_123"}) as test_client:
        yield test_client


async def test_chat_with_cookies(client: AsyncClient):
    response = await client.post(
        "/chat", 
        json="Привет, ИИ!",
    )
    print("Ответ сервера:", response.json())
    assert response.status_code == 200

async def test_chat_with_stream_cookies(client: AsyncClient):
    async with client.stream(
        "POST",
        "/chat/stream", 
        json="Привет, ИИ!",
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                print("chunk:", line)

async def test_chat_with_extract_cookies(client: AsyncClient):
    response = await client.post(
        "/extract", 
        json="Привет, ИИ!",
    )
    print("Ответ сервера:", response.json())
    assert response.status_code == 200