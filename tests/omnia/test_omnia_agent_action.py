from unittest.mock import AsyncMock

import pytest

from agent.tools import omnia_agent_action


@pytest.mark.asyncio
async def test_omnia_agent_action_uses_trusted_run_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("agent.tools.omnia_agent_action", fromlist=["omnia_agent_action"])
    monkeypatch.setattr(
        module,
        "get_config",
        lambda: {
            "configurable": {
                "thread_id": "agent-thread-1",
                "user_email": "kyle@example.com",
                "omnia_thread": {"thread_id": "dm-kyle-luna"},
            }
        },
    )
    post = AsyncMock(return_value={"success": True, "tasks": []})
    monkeypatch.setattr(module, "post_omnia_agent_action", post)

    result = await omnia_agent_action("list_tasks")

    assert result["success"] is True
    await_args = post.await_args
    assert await_args is not None
    payload = await_args.args[0]
    assert payload["sender_email"] == "kyle@example.com"
    assert payload["dm_thread_id"] == "dm-kyle-luna"
    assert payload["idempotency_key"].startswith("open-swe:unknown-run:")


@pytest.mark.asyncio
async def test_omnia_agent_action_validates_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("agent.tools.omnia_agent_action", fromlist=["omnia_agent_action"])
    monkeypatch.setattr(
        module,
        "get_config",
        lambda: {"configurable": {"omnia_thread": {"thread_id": "dm-1"}}},
    )
    post = AsyncMock()
    monkeypatch.setattr(module, "post_omnia_agent_action", post)

    assert (await omnia_agent_action("create_task"))["success"] is False
    assert (await omnia_agent_action("merge_task"))["success"] is False
    assert (await omnia_agent_action("browser_session"))["success"] is False
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_omnia_agent_action_requests_a_preview_browser_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("agent.tools.omnia_agent_action", fromlist=["omnia_agent_action"])
    monkeypatch.setattr(
        module,
        "get_config",
        lambda: {
            "run_id": "run-9",
            "configurable": {
                "thread_id": "agent-thread-9",
                "user_email": "kyle@example.com",
                "omnia_thread": {"thread_id": "dm-kyle-luna"},
            },
        },
    )
    post = AsyncMock(return_value={"success": True, "browser_session_url": "https://preview/session"})
    monkeypatch.setattr(module, "post_omnia_agent_action", post)

    result = await omnia_agent_action(
        "browser_session",
        preview_url="https://omnia-preview.vercel.app/chat",
        redirect_path="/chat",
    )

    assert result["success"] is True
    payload = post.await_args.args[0]
    assert payload["preview_url"] == "https://omnia-preview.vercel.app/chat"
    assert payload["redirect_path"] == "/chat"
    assert payload["sender_email"] == "kyle@example.com"
