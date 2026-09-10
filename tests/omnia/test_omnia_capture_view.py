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
