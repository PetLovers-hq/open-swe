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
    payload = post.await_args.args[0]
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
    post.assert_not_awaited()
