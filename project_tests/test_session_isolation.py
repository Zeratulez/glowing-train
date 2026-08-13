import pytest

from unittest.mock import MagicMock

import src.dependencies.ai_dependencies as ai
from src.schemas.test_schema import Cookies
from project_tests import fakes


async def test_histories_are_independent_per_session(client_mock: MagicMock) -> None:
    completion = fakes.make_completion(text="ok")
    client_mock.chat.completions.create.return_value = completion

    await ai.ai_chat("hello a", Cookies(session_id="a"))
    await ai.ai_chat("hello b", Cookies(session_id="b"))

    assert "hello a" in [m["content"] for m in ai.message_history["a"]]
    assert "hello b" in [m["content"] for m in ai.message_history["b"]]
    assert ai.message_history["a"] != ai.message_history["b"]
    assert not any(m["content"] == "hello b" for m in ai.message_history["a"])


async def test_history_appends_user_and_assistant_in_order(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.return_value = fakes.make_completion(text="answer")

    await ai.ai_chat("question", cookies)

    history = ai.message_history[cookies.session_id]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "question"
    assert history[1]["content"] == "answer"


async def test_stream_appends_history(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.return_value = fakes.make_stream_iter(
        [
            fakes.make_stream_chunk(text="Hello "),
            fakes.make_stream_chunk(text="world"),
            fakes.make_stream_chunk(usage=fakes.SimpleNamespace(prompt_tokens=5, completion_tokens=3)),
        ]
    )

    chunks = [c async for c in ai.ai_chat_stream("stream me", cookies)]

    history = ai.message_history[cookies.session_id]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "Hello world"
    assert any(c["type"] == "metrics" for c in chunks)
    assert any(c["type"] == "text" for c in chunks)


async def test_embedding_context_included_in_message(client_mock: MagicMock, cookies: Cookies) -> None:
    client_mock.chat.completions.create.return_value = fakes.make_completion(text="ok")

    chunk = fakes.SimpleNamespace(content="knowledge base text")
    embeddings = [(chunk, 0.1)]

    await ai.ai_chat("query", cookies, embeddings=embeddings)

    create_kwargs = client_mock.chat.completions.create.call_args.kwargs
    messages = create_kwargs["messages"]
    assert any("knowledge base text" in m.get("content", "") for m in messages)
    assert any("отвечай только на основе" in m.get("content", "") for m in messages)