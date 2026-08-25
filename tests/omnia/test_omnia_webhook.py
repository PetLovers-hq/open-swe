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
    assert await_args.kwargs["multitask_strategy"] == "enqueue"
    upsert.assert_awaited_once()


def test_omnia_scope_lanes_are_stable_and_conservative() -> None:
    dm = "dm-kyle-luna"
    app_thread = omnia_routes._thread_id(dm, "Add threaded replies to chat")
    profile_thread = omnia_routes._thread_id(dm, "Make team profiles clickable")
    inventory_thread = omnia_routes._thread_id(dm, "Fix the inventory SKU reconciliation")

    assert app_thread == profile_thread
    assert inventory_thread != app_thread
    assert inventory_thread == omnia_routes._thread_id(dm, "Review FBA inventory stock")


def test_cross_domain_request_fails_safe_to_shared_app_lane() -> None:
    dm = "dm-kyle-luna"

    assert omnia_routes._scope_key("Compare inventory margin and ad campaign results") == "app"
    assert omnia_routes._thread_id(
        dm, "Compare inventory margin and ad campaign results"
    ) == omnia_routes._thread_id(dm, "Update the shared Omnia shell")


def test_omnia_thread_epoch_rotates_sandbox_without_changing_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dm = "dm-kyle-luna"
    before = omnia_routes._thread_id(dm, "Make team profiles clickable")
    monkeypatch.setenv("OMNIA_THREAD_EPOCH", "browser-v1")

    after = omnia_routes._thread_id(dm, "Make team profiles clickable")

    assert after != before
    assert after == omnia_routes._thread_id(dm, "Add threaded replies to chat")


@pytest.mark.asyncio
async def test_process_omnia_dm_sends_screenshot_as_native_image_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = AsyncMock()
    image_block = {"type": "image", "base64": "c2NyZWVuc2hvdA==", "mime_type": "image/png"}
    fetch_image = AsyncMock(return_value=image_block)
    monkeypatch.setattr(omnia_routes.common, "upsert_agent_thread_owner_metadata", AsyncMock())
    monkeypatch.setattr(omnia_routes.common, "fetch_image_block", fetch_image)
    monkeypatch.setattr(omnia_routes, "dispatch_agent_run", dispatch)
    event = omnia_routes.OmniaDmEvent.model_validate(
        _payload()
        | {
            "attachments": [
                {
                    "name": "desired-layout.png",
                    "mime": "image/png",
                    "url": "https://storage.example.com/signed-layout",
                }
            ]
        }
    )

    await omnia_routes.process_omnia_dm(event)

    await_args = dispatch.await_args
    assert await_args is not None
    content = await_args.args[1]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "desired-layout.png" in content[0]["text"]
    assert "signed-layout" not in content[0]["text"]
    assert content[1] == image_block
    fetch_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_omnia_dm_preserves_multiple_screenshot_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = AsyncMock()
    fetch_image = AsyncMock(
        side_effect=[
            {"type": "image", "base64": "b25l", "mime_type": "image/png"},
            {"type": "image", "base64": "dHdv", "mime_type": "image/jpeg"},
        ]
    )
    monkeypatch.setattr(omnia_routes.common, "upsert_agent_thread_owner_metadata", AsyncMock())
    monkeypatch.setattr(omnia_routes.common, "fetch_image_block", fetch_image)
    monkeypatch.setattr(omnia_routes, "dispatch_agent_run", dispatch)
    event = omnia_routes.OmniaDmEvent.model_validate(
        _payload()
        | {
            "attachments": [
                {"name": "first.png", "mime": "image/png", "url": "https://x/first"},
                {"name": "notes.pdf", "mime": "application/pdf", "url": "https://x/notes"},
                {"name": "second.jpg", "mime": "image/jpeg", "url": "https://x/second"},
            ]
        }
    )

    await omnia_routes.process_omnia_dm(event)

    await_args = dispatch.await_args
    assert await_args is not None
    content = await_args.args[1]
    assert [block["base64"] for block in content[1:]] == ["b25l", "dHdv"]
    assert [call.args[0] for call in fetch_image.await_args_list] == [
        "https://x/first",
        "https://x/second",
    ]
    assert "notes.pdf" in content[0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mime", "image_result", "expected"),
    [
        ("image/png", None, "could not be loaded"),
        (
            "image/png",
            {"type": "text", "text": "An attached image was skipped because it exceeded the limit."},
            "exceeded the limit",
        ),
        ("image/svg+xml", None, "is unsupported"),
    ],
)
async def test_process_omnia_dm_reports_unavailable_screenshot(
    monkeypatch: pytest.MonkeyPatch,
    mime: str,
    image_result: dict[str, str] | None,
    expected: str,
) -> None:
    dispatch = AsyncMock()
    fetch_image = AsyncMock(return_value=image_result)
    monkeypatch.setattr(omnia_routes.common, "upsert_agent_thread_owner_metadata", AsyncMock())
    monkeypatch.setattr(omnia_routes.common, "fetch_image_block", fetch_image)
    monkeypatch.setattr(omnia_routes, "dispatch_agent_run", dispatch)
    event = omnia_routes.OmniaDmEvent.model_validate(
        _payload()
        | {
            "attachments": [
                {"name": "guide", "mime": mime, "url": "https://x/guide"},
            ]
        }
    )

    await omnia_routes.process_omnia_dm(event)

    await_args = dispatch.await_args
    assert await_args is not None
    content = await_args.args[1]
    assert expected in content[1]["text"]
    if mime == "image/svg+xml":
        fetch_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_omnia_dm_uses_vision_fallback_for_text_only_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = AsyncMock()
    monkeypatch.setenv("OMNIA_AGENT_MODEL", "anthropic:claude-text-only")
    monkeypatch.setenv("OMNIA_AGENT_EFFORT", "low")
    monkeypatch.setattr(omnia_routes.common, "model_supports_images", lambda _model: False)
    monkeypatch.setattr(
        omnia_routes.common,
        "default_vision_model_pair",
        lambda: ("openai:gpt-5.6-luna", "high"),
    )
    monkeypatch.setattr(omnia_routes.common, "upsert_agent_thread_owner_metadata", AsyncMock())
    monkeypatch.setattr(
        omnia_routes.common,
        "fetch_image_block",
        AsyncMock(return_value={"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"}),
    )
    monkeypatch.setattr(omnia_routes, "dispatch_agent_run", dispatch)
    event = omnia_routes.OmniaDmEvent.model_validate(
        _payload()
        | {
            "attachments": [
                {"name": "guide.png", "mime": "image/png", "url": "https://x/guide"},
            ]
        }
    )

    await omnia_routes.process_omnia_dm(event)

    await_args = dispatch.await_args
    assert await_args is not None
    configurable = await_args.args[2]
    assert configurable["agent_model_id"] == "openai:gpt-5.6-luna"
    assert configurable["agent_effort"] == "high"
