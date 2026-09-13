import base64
import hashlib
import importlib
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

module = importlib.import_module("agent.tools.omnia_capture_view")
PNG = b"\x89PNG\r\n\x1a\nreal-capture-fixture"
SESSION = "https://omnia.example/bootstrap?token=secret"
RECEIPT = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def capture(monkeypatch):
    events = []
    files = {}
    backend = SimpleNamespace()
    backend.stage_failure = False
    backend.capture_failure = False
    backend.bad_hash = False

    async def execute(command, timeout):
        if "git fetch" in command:
            events.append("stage")
            return SimpleNamespace(exit_code=int(backend.stage_failure), output="stage failed")
        events.append("capture")
        args = shlex.split(command)
        script = next(i for i, arg in enumerate(args) if arg.endswith("luna-visual-proof.mts"))
        assert args[script + 1] == SESSION
        png_path = args[script + 2]
        result_path = args[args.index(">") + 1]
        error_path = args[args.index("2>") + 1]
        files[error_path] = f"Wrong page: Company Wiki. Session {SESSION}".encode()
        files[png_path] = PNG
        files[result_path] = json.dumps(
            {
                "screenshot_path": png_path,
                "png_sha256": "bad" if backend.bad_hash else hashlib.sha256(PNG).hexdigest(),
                "receipt": RECEIPT,
                "page": {"text": "Last edited"},
            }
        ).encode()
        return SimpleNamespace(exit_code=int(backend.capture_failure), output="")

    async def write(path, content):
        events.append("write")
        files[path] = content.encode()
        return SimpleNamespace(error=None)

    async def download(paths):
        return [{"content": files.get(path)} for path in paths]

    async def session(*args, **kwargs):
        events.append("session")
        return {
            "success": True,
            "browser_session_url": SESSION,
            "commit_sha": "a" * 40,
            "preview_url": "https://preview.example",
        }

    backend.aexecute = AsyncMock(side_effect=execute)
    backend.awrite = AsyncMock(side_effect=write)
    backend.adownload_files = AsyncMock(side_effect=download)
    config = {
        "configurable": {
            "source": "omnia",
            "thread_id": "thread-1",
            "omnia_thread": {"thread_id": "dm-kyle-luna"},
            "repo": {"owner": "PetLovers-hq", "name": "Omnia"},
        }
    }
    monkeypatch.setattr(module, "get_config", lambda: config)
    monkeypatch.setattr(module, "get_sandbox_backend", AsyncMock(return_value=backend))
    monkeypatch.setattr(module, "aresolve_repo_dir", AsyncMock(return_value="/sandbox/Omnia"))
    monkeypatch.setattr(module, "_execute_omnia_agent_action", AsyncMock(side_effect=session))
    return backend, events, files, config


async def test_capture_stages_before_fresh_session_and_returns_bound_png(capture):
    backend, events, files, _ = capture
    results = []
    for _ in range(2):
        results.append(
            await module.omnia_capture_view(
                30,
                "docs-phone",
                ["Last edited"],
                path="/docs",
                viewport="phone",
                steps=[{"click_text": "🐾About the Company", "within": "main aside"}],
                time_zone="Asia/Bangkok",
            )
        )
    assert events == ["stage", "write", "session", "capture"] * 2
    for blocks in results:
        assert blocks[1]["type"] == "image_url"
        assert base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1]) == PNG
    results = [json.loads(blocks[0]["text"]) for blocks in results]
    assert results[0]["screenshot_path"] != results[1]["screenshot_path"]
    for result in results:
        assert result["success"] is True
        assert result["commit_sha"] == "a" * 40
        assert result["auth_receipt"] == RECEIPT
        assert files[result["screenshot_path"]] == PNG
        assert SESSION not in json.dumps(result)
    view = json.loads(backend.awrite.await_args.args[1])
    assert view["viewport"] == {"width": 390, "height": 844}
    assert view["steps"] == [{"clickText": "🐾About the Company", "within": "main aside"}]
    assert view["timeZone"] == "Asia/Bangkok"


async def test_staging_failure_does_not_consume_session(capture):
    backend, events, _, _ = capture
    backend.stage_failure = True
    result = await module.omnia_capture_view(30, "docs", ["Last edited"])
    assert result["success"] is False
    assert events == ["stage"]


async def test_failed_view_returns_diagnostic_without_session_secret(capture):
    backend, _, _, _ = capture
    backend.capture_failure = True
    result = await module.omnia_capture_view(30, "docs", ["Last edited"])
    assert result["success"] is False
    assert "Company Wiki" in result["error"]
    assert SESSION not in result["error"]
    assert "auth_receipt" not in result


async def test_mismatched_png_cannot_be_reported_as_success(capture):
    backend, _, _, _ = capture
    backend.bad_hash = True
    result = await module.omnia_capture_view(30, "docs", ["Last edited"])
    assert result["success"] is False
    assert "did not match" in result["error"]


@pytest.mark.parametrize(
    "name,path", [("../escape", "/docs"), ("ok", "//evil.example"), ("ok", "/\\evil")]
)
async def test_unsafe_capture_input_never_reaches_sandbox(capture, name, path):
    _, events, _, _ = capture
    result = await module.omnia_capture_view(30, name, ["Last edited"], path=path)
    assert result["success"] is False
    assert events == []


async def test_non_omnia_context_cannot_capture(capture):
    _, events, _, config = capture
    config["configurable"]["source"] = "slack"
    result = await module.omnia_capture_view(30, "docs", ["Last edited"])
    assert result["success"] is False
    assert events == []


async def test_pending_preview_waits_and_captures_without_model_retry(capture, monkeypatch):
    backend, events, _, _ = capture
    original = module._execute_omnia_agent_action.side_effect
    pending = {
        "success": False,
        "error": "The current task commit does not have a ready preview yet. Wait for its build and retry browser_session; do not ask the user to resolve a deployment URL.",
    }
    calls = 0

    async def session(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            return pending
        return await original(*args, **kwargs)

    monkeypatch.setattr(module, "_execute_omnia_agent_action", AsyncMock(side_effect=session))
    sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = await module.omnia_capture_view(30, "docs", ["Last edited"])
    assert json.loads(result[0]["text"])["success"] is True
    assert calls == 3
    assert sleep.await_count == 2
    assert events == ["stage", "write", "session", "capture"]
    assert backend.aexecute.await_count == 2


async def test_preview_wait_is_bounded_and_does_not_capture(capture, monkeypatch):
    backend, events, _, _ = capture
    request = AsyncMock(
        return_value={
            "success": False,
            "error": "The current task commit does not have a ready preview yet.",
        }
    )
    monkeypatch.setattr(module, "_execute_omnia_agent_action", request)
    sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = await module.omnia_capture_view(30, "docs", ["Last edited"])
    assert result["success"] is False
    assert "ten minutes" in result["error"]
    assert request.await_count == 31
    assert sleep.await_count == 30
    assert events == ["stage", "write"]
    assert backend.aexecute.await_count == 1


async def test_authentication_error_is_not_retried_as_pending_build(capture, monkeypatch):
    _, events, _, _ = capture
    request = AsyncMock(return_value={"success": False, "error": "Unauthorized conversation"})
    monkeypatch.setattr(module, "_execute_omnia_agent_action", request)
    sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = await module.omnia_capture_view(30, "docs", ["Last edited"])
    assert result == {"success": False, "error": "Unauthorized conversation"}
    request.assert_awaited_once()
    sleep.assert_not_awaited()
    assert events == ["stage", "write"]


async def test_capture_image_reaches_langchain_tool_message_without_stringification(capture):
    from langchain_core.tools import StructuredTool

    tool = StructuredTool.from_function(coroutine=module.omnia_capture_view)
    result = await tool.ainvoke(
        {
            "name": "omnia_capture_view",
            "args": {
                "task_number": 33,
                "name": "phone",
                "wait_for_text": ["Last edited"],
                "viewport": "phone",
            },
            "id": "capture-test",
            "type": "tool_call",
        }
    )
    assert result.tool_call_id == "capture-test"
    assert isinstance(result.content, list)
    assert result.content[1]["type"] == "image_url"
    assert base64.b64decode(result.content[1]["image_url"]["url"].split(",", 1)[1]) == PNG
    assert json.loads(result.content[0]["text"])["viewport"] == "phone"

    from langchain_openai.chat_models.base import _construct_responses_api_input

    request = _construct_responses_api_input([result])
    assert request[0]["type"] == "function_call_output"
    image = request[0]["output"][1]
    assert image["type"] == "input_image"
    assert base64.b64decode(image["image_url"].split(",", 1)[1]) == PNG


@pytest.mark.parametrize(
    "step,expected",
    [
        (
            {"fill_placeholder": "Search", "text": "invoice"},
            {"fillPlaceholder": "Search", "text": "invoice"},
        ),
        ({"fill_placeholder": "Search", "text": ""}, {"fillPlaceholder": "Search", "text": ""}),
        (
            {"press_key": "ArrowDown", "within": "input"},
            {"pressKey": "ArrowDown", "within": "input"},
        ),
        ({"scroll_text": "Pictures"}, {"scrollText": "Pictures"}),
    ],
)
def test_normalizes_browser_actions(step, expected):
    assert module._capture_step(step) == expected


@pytest.mark.parametrize(
    "step",
    [
        {"click_text": "Go", "press_key": "Enter"},
        {"fill_placeholder": "Search"},
        {"press_key": "javascript"},
        {"scroll_text": ""},
        {"click_text": "Go", "script": "bad"},
        {"fill_placeholder": "Search", "text": "x" * 10_001},
    ],
)
def test_rejects_ambiguous_or_unsupported_browser_actions(step):
    assert module._capture_step(step) is None


async def test_capture_forwards_interactions_and_keeps_native_evidence(capture):
    backend, events, _, _ = capture
    blocks = await module.omnia_capture_view(
        30,
        "search-phone",
        ["Search results"],
        steps=[
            {"click_text": "Search"},
            {"fill_placeholder": "Search…", "text": "invoice"},
            {"press_key": "ArrowDown"},
            {"scroll_text": "Search results"},
        ],
    )
    assert events == ["stage", "write", "session", "capture"]
    view = json.loads(backend.awrite.await_args.args[1])
    assert view["steps"] == [
        {"clickText": "Search"},
        {"fillPlaceholder": "Search…", "text": "invoice"},
        {"pressKey": "ArrowDown"},
        {"scrollText": "Search results"},
    ]
    assert base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1]) == PNG
