import base64
from unittest.mock import AsyncMock

import pytest

from agent.tools import omnia_dm_reply


def _config() -> dict:
    return {
        "run_id": "run-1",
        "configurable": {
            "thread_id": "agent-thread-1",
            "omnia_thread": {
                "thread_id": "dm-kyle-luna",
                "event_id": "note-123",
                "journal_run_id": 44,
            },
        },
    }


@pytest.mark.asyncio
async def test_omnia_reply_sends_review_png_as_native_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("agent.tools.omnia_dm_reply", fromlist=["omnia_dm_reply"])
    monkeypatch.setattr(module, "get_config", _config)
    png = b"\x89PNG\r\n\x1a\nfinished-ui"
    monkeypatch.setattr(
        module,
        "_native_png",
        AsyncMock(
            return_value={
                "name": "finished-ui.png",
                "mime": "image/png",
                "data_base64": base64.b64encode(png).decode(),
            }
        ),
    )
    post = AsyncMock(return_value=(True, None))
    monkeypatch.setattr(module, "post_omnia_dm_event", post)

    result = await omnia_dm_reply(
        "Threaded replies now open beside the conversation.",
        screenshot_path="evidence/threaded-replies.png",
        completion=True,
        task_number=7,
        commit_sha="a" * 40,
        preview_url="https://omnia-test.vercel.app/chat",
        auth_receipt="11111111-1111-4111-8111-111111111111",
        passed_checks=["focused tests", "build"],
    )

    assert result == {"success": True}
    await_args = post.await_args
    assert await_args is not None
    payload = await_args.args[0]
    assert payload["attachments"] == [
        {
            "name": "finished-ui.png",
            "mime": "image/png",
            "data_base64": base64.b64encode(png).decode(),
        }
    ]
    assert "github" not in payload["message"].lower()
    assert payload["purpose"] == "review"
    assert payload["event_id"] == "note-123"
    assert payload["journal_run_id"] == 44
    assert payload["evidence"]["task_number"] == 7
    assert payload["evidence"]["auth_receipt"] == "11111111-1111-4111-8111-111111111111"


@pytest.mark.asyncio
async def test_omnia_reply_rejects_non_png_review_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("agent.tools.omnia_dm_reply", fromlist=["omnia_dm_reply"])
    monkeypatch.setattr(module, "get_config", _config)
    monkeypatch.setattr(
        module,
        "_native_png",
        AsyncMock(side_effect=ValueError("Omnia review evidence must be an actual PNG file")),
    )
    post = AsyncMock()
    monkeypatch.setattr(module, "post_omnia_dm_event", post)

    result = await omnia_dm_reply("Finished.", screenshot_path="preview.svg")

    assert result["success"] is False
    assert "actual PNG" in result["error"]
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_coding_completion_requires_png(monkeypatch: pytest.MonkeyPatch) -> None:
    module = __import__("agent.tools.omnia_dm_reply", fromlist=["omnia_dm_reply"])
    monkeypatch.setattr(module, "get_config", _config)
    post = AsyncMock()
    monkeypatch.setattr(module, "post_omnia_dm_event", post)

    result = await omnia_dm_reply("Finished.", completion=True)

    assert result["success"] is False
    assert "requires a real PNG" in result["error"]
    post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_outcome", "purpose", "terminal_status"),
    [("blocker", "blocker", None), ("failure", "progress", "error")],
)
async def test_omnia_reply_can_send_terminal_non_review_outcome(
    monkeypatch: pytest.MonkeyPatch,
    terminal_outcome: str,
    purpose: str,
    terminal_status: str | None,
) -> None:
    module = __import__("agent.tools.omnia_dm_reply", fromlist=["omnia_dm_reply"])
    monkeypatch.setattr(module, "get_config", _config)
    post = AsyncMock(return_value=(True, None))
    monkeypatch.setattr(module, "post_omnia_dm_event", post)

    result = await omnia_dm_reply(
        "The authenticated preview is unavailable.",
        terminal_outcome=terminal_outcome,
    )

    assert result == {"success": True}
    payload = post.await_args.args[0]
    assert payload["purpose"] == purpose
    assert payload.get("terminal_status") == terminal_status


def test_omnia_prompt_hides_developer_plumbing_and_requires_native_png() -> None:
    prompt = __import__("agent.prompt", fromlist=["OMNIA_SOURCE_GUIDANCE"])
    guidance = prompt.OMNIA_SOURCE_GUIDANCE
    assert "never mention pull requests" in guidance
    assert "actual PNG" in guidance
    assert "completion=True" in guidance
    assert "native previewable attachment" in guidance
    assert "Smithbox" in guidance
    assert "Confirm this visual proof" in guidance
    assert 'terminal_outcome="blocker"' in guidance
    assert 'terminal_outcome="failure"' in guidance
