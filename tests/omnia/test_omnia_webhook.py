import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, HTTPException, Request

from agent.webhooks import omnia_routes


def _request(body: bytes, signature: str) -> Request:
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/webhooks/omnia",
            "headers": [(b"x-omnia-signature", signature.encode())],
        },
        receive,
    )


def _payload() -> dict[str, str]:
    return {
        "event_id": "note-123",
        "dm_thread_id": "dm-kyle-luna",
        "message": "Fix task seven and return review evidence.",
        "sender_id": "user-123",
        "sender_name": "Kyle",
        "sender_email": "kyle@example.com",
        "github_login": "kylebp2025",
        "repo_owner": "PetLovers-hq",
        "repo_name": "Omnia",
    }


@pytest.mark.asyncio
async def test_omnia_webhook_accepts_signed_event(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps(_payload()).encode()
    secret = "test-omnia-secret"
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    monkeypatch.setenv("OMNIA_WEBHOOK_SECRET", secret)
    monkeypatch.setattr(omnia_routes.common, "_is_repo_allowed", lambda _repo: True)
    callback = AsyncMock(return_value=(True, None))
    monkeypatch.setattr(omnia_routes, "post_omnia_dm_event", callback)
    tasks = BackgroundTasks()

    response = await omnia_routes.omnia_webhook(_request(body, signature), tasks)

    assert response["status"] == "accepted"
    assert len(tasks.tasks) == 1
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_omnia_webhook_rejects_bad_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIA_WEBHOOK_SECRET", "expected")
    body = json.dumps(_payload()).encode()

    with pytest.raises(HTTPException) as exc:
        await omnia_routes.omnia_webhook(_request(body, "sha256=wrong"), BackgroundTasks())

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_process_omnia_dm_uses_luna_and_durable_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert = AsyncMock()
    dispatch = AsyncMock()
    monkeypatch.setattr(omnia_routes.common, "upsert_agent_thread_owner_metadata", upsert)
    monkeypatch.setattr(omnia_routes, "dispatch_agent_run", dispatch)
    event = omnia_routes.OmniaDmEvent.model_validate(_payload())

    await omnia_routes.process_omnia_dm(event)

    await_args = dispatch.await_args
    assert await_args is not None
    configurable = await_args.args[2]
    assert configurable["source"] == "omnia"
    assert configurable["agent_model_id"] == "openai:gpt-5.6-luna"
    assert configurable["agent_effort"] == "high"
    assert await_args.kwargs["source"] == "omnia"
    upsert.assert_awaited_once()
