from unittest.mock import AsyncMock

import httpx
import pytest

from agent.utils.omnia import post_omnia_dm_event


@pytest.mark.asyncio
async def test_callback_retries_transient_deployment_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIA_CALLBACK_URL", "https://omnia.example/callback")
    monkeypatch.setenv("OMNIA_CALLBACK_SECRET", "secret")
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts < 3 else 200, request=request)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    monkeypatch.setattr("agent.utils.omnia.asyncio.sleep", AsyncMock())

    success, error = await post_omnia_dm_event({"kind": "message"})

    assert success is True
    assert error is None
    assert attempts == 3


@pytest.mark.asyncio
async def test_callback_does_not_retry_contract_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIA_CALLBACK_URL", "https://omnia.example/callback")
    monkeypatch.setenv("OMNIA_CALLBACK_SECRET", "secret")
    client = AsyncMock()
    client.__aenter__.return_value.post.return_value = httpx.Response(422)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)

    success, error = await post_omnia_dm_event({"kind": "message"})

    assert success is False
    assert error == "Omnia callback returned HTTP 422"
    client.__aenter__.return_value.post.assert_awaited_once()
