from unittest.mock import AsyncMock

import httpx
import pytest

from agent.utils.omnia import post_omnia_agent_action


@pytest.mark.parametrize("failure", [502, 409, "disconnect"])
async def test_replays_identical_signed_request_after_transient_failure(monkeypatch, failure):
    monkeypatch.setenv("OMNIA_TOOL_URL", "https://omnia.test/tool")
    monkeypatch.setenv("OMNIA_TOOL_SECRET", "test-only")
    calls = []

    async def post(self, url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if failure == "disconnect":
                raise httpx.ReadTimeout("lost response")
            return httpx.Response(failure, json={"error": "temporary"})
        return httpx.Response(202, json={"release_accepted": True})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr("agent.utils.omnia.asyncio.sleep", AsyncMock())
    result = await post_omnia_agent_action({"action": "merge_task", "idempotency_key": "fixed-key"})
    assert result == {"success": True, "release_accepted": True}
    assert len(calls) == 2
    assert calls[0] == calls[1]


async def test_permission_failure_is_not_retried(monkeypatch):
    monkeypatch.setenv("OMNIA_TOOL_URL", "https://omnia.test/tool")
    monkeypatch.setenv("OMNIA_TOOL_SECRET", "test-only")
    post = AsyncMock(return_value=httpx.Response(403, json={"error": "denied"}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    result = await post_omnia_agent_action({"idempotency_key": "fixed-key"})
    assert result == {"success": False, "error": "denied"}
    assert post.await_count == 1


async def test_http_success_does_not_override_tool_failure(monkeypatch):
    monkeypatch.setenv("OMNIA_TOOL_URL", "https://omnia.test/tool")
    monkeypatch.setenv("OMNIA_TOOL_SECRET", "test-only")
    post = AsyncMock(
        return_value=httpx.Response(200, json={"success": False, "error": "action rejected"})
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    result = await post_omnia_agent_action({"idempotency_key": "fixed-key"})
    assert result == {"success": False, "error": "action rejected"}
    assert post.await_count == 1
