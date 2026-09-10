"""Capture authenticated Omnia review evidence in the durable coding sandbox."""

import asyncio
import hashlib
import json
import posixpath
import re
import shlex
import uuid
from typing import Any, Literal, NotRequired

from langgraph.config import get_config
from typing_extensions import TypedDict

from ..utils.sandbox_paths import aresolve_repo_dir
from ..utils.sandbox_state import get_sandbox_backend
from .omnia_agent_action import _execute_omnia_agent_action
from .omnia_dm_reply import _download_bytes


class CaptureStep(TypedDict):
    click_text: str
    within: NotRequired[str]


async def _ready_preview_session(task_number: int, path: str) -> dict[str, Any]:
    for attempt in range(31):
        session = await _execute_omnia_agent_action(
            "browser_session", task_number=task_number, redirect_path=path
        )
        if session.get("success") or not str(session.get("error", "")).startswith(
            "The current task commit does not have a ready preview yet."
        ):
            return session
        if attempt < 30:
            await asyncio.sleep(20)
    return {
        "success": False,
        "error": "The preview still is not ready after ten minutes. Inspect its build status and repair any failure before capturing again. Do not ask the user to resolve deployment details.",
    }


async def omnia_capture_view(
    task_number: int,
    name: str,
    wait_for_text: list[str],
    path: str = "/chat",
    viewport: Literal["desktop", "phone"] = "desktop",
    steps: list[CaptureStep] | None = None,
    time_zone: str | None = None,
) -> dict[str, Any]:
    """Capture one real preview PNG and its fresh authenticated receipt.

    Handles current-driver staging and automatic waiting for pending preview builds,
    one-use browser authentication, navigation,
    screenshot hashing, and receipt registration. Do not run proof scripts manually.
    Use one call per requested view, then read_file each returned screenshot_path
    and send all views together through omnia_dm_reply screenshots.

    path must be an observed route; never guess record IDs. wait_for_text must
    describe the requested screen. steps click exact visible text (including emoji).
    If a label exists in both a sidebar and a card, set within to the intended CSS
    container, e.g. main aside, based on repository markup. A failed view returns
    its actual page/controls; change the view before retrying. Every call creates
    fresh authentication automatically. No approval or deployment occurs here.
    """
    if (
        not isinstance(task_number, int)
        or isinstance(task_number, bool)
        or task_number < 1
        or not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name)
        or not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or "\\" in path
        or viewport not in ("desktop", "phone")
        or not isinstance(wait_for_text, list)
        or not wait_for_text
        or len(wait_for_text) > 10
        or any(not isinstance(t, str) or not t.strip() for t in wait_for_text)
    ):
        return {
            "success": False,
            "error": "Provide a task, safe image name, local route, viewport, and expected screen text",
        }
    if steps is not None and (
        not isinstance(steps, list)
        or len(steps) > 10
        or any(
            not isinstance(s, dict)
            or not isinstance(s.get("click_text"), str)
            or not s["click_text"].strip()
            or ("within" in s and (not isinstance(s["within"], str) or not s["within"].strip()))
            for s in steps
        )
    ):
        return {"success": False, "error": "Provide at most ten steps with exact click_text"}
    config = get_config().get("configurable", {})
    repo = config.get("repo", {})
    thread_id = config.get("thread_id")
    if (
        config.get("source") != "omnia"
        or not config.get("omnia_thread")
        or not thread_id
        or repo.get("owner", "").lower() != "petlovers-hq"
        or repo.get("name", "").lower() != "omnia"
    ):
        return {"success": False, "error": "An authenticated Omnia task context is required"}

    bootstrap_url = ""
    try:
        backend = await get_sandbox_backend(str(thread_id))
        repo_dir = await aresolve_repo_dir(backend, repo["name"])
        capture_dir = posixpath.join(repo_dir, ".luna-evidence", "captures", uuid.uuid4().hex)
        script_dir = posixpath.join(capture_dir, "scripts")
        q = shlex.quote
        files = (
            "scripts/luna-visual-proof.mts",
            "scripts/lib/tieout.mts",
            "scripts/lib/chromeProfile.mts",
        )
        stage = [
            f"cd {q(repo_dir)}",
            "git fetch origin main",
            f"mkdir -p {q(posixpath.join(script_dir, 'lib'))}",
        ]
        stage.extend(
            f"git show {q('origin/main:' + source)} > {q(posixpath.join(capture_dir, source))}"
            for source in files
        )
        staged = await backend.aexecute(" && ".join(stage), timeout=120)
        if staged.exit_code != 0:
            return {
                "success": False,
                "error": f"Could not stage current capture driver: {staged.output[-3000:]}",
            }
        view: dict[str, Any] = {
            "path": path,
            "viewport": {"width": 390, "height": 844}
            if viewport == "phone"
            else {"width": 1500, "height": 950},
            "waitForText": wait_for_text,
            "steps": [
                {"clickText": s["click_text"], **({"within": s["within"]} if "within" in s else {})}
                for s in steps or []
            ],
        }
        if time_zone:
            view["timeZone"] = time_zone
        view_path = posixpath.join(capture_dir, "view.json")
        written = await backend.awrite(view_path, json.dumps(view))
        if written.error:
            return {"success": False, "error": "Could not write capture view configuration"}
        # Mint only after staging and configuration finish. No model turn occurs
        # between receiving the one-use launcher and consuming it.
        session = await _ready_preview_session(task_number, path)
        if not session.get("success"):
            return session
        bootstrap_url = session["browser_session_url"]
        commit_sha = session["commit_sha"]
        preview_url = session["preview_url"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            return {"success": False, "error": "Preview did not resolve an exact task commit"}
        png_path = posixpath.join(capture_dir, name.removesuffix(".png") + ".png")
        result_path = posixpath.join(capture_dir, "result.json")
        error_path = posixpath.join(capture_dir, "capture-error.txt")
        argv = [
            "node",
            "--experimental-transform-types",
            posixpath.join(script_dir, "luna-visual-proof.mts"),
            bootstrap_url,
            png_path,
            str(task_number),
            commit_sha,
            preview_url,
            "4500",
            view_path,
        ]
        captured = await backend.aexecute(
            f"cd {q(repo_dir)} && {shlex.join(argv)} > {q(result_path)} 2> {q(error_path)}",
            timeout=180,
        )
        if captured.exit_code != 0:
            errors = await backend.adownload_files([error_path])
            diagnostic = (
                _download_bytes(errors[0]) if errors else None
            ) or b"Capture did not finish"
            return {
                "success": False,
                "error": diagnostic.decode(errors="replace")[-12000:].replace(
                    bootstrap_url, "[one-use session]"
                ),
            }
        outputs = await backend.adownload_files([result_path, png_path])
        result_bytes = _download_bytes(outputs[0]) if outputs else None
        png_bytes = _download_bytes(outputs[1]) if len(outputs) > 1 else None
        if not result_bytes or not png_bytes or not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return {"success": False, "error": "Capture did not return real PNG evidence"}
        proof = json.loads(result_bytes)
        if (
            proof.get("screenshot_path") != png_path
            or proof.get("png_sha256") != hashlib.sha256(png_bytes).hexdigest()
            or not re.fullmatch(r"[0-9a-f-]{36}", proof.get("receipt", ""))
        ):
            return {"success": False, "error": "Capture evidence and receipt did not match"}
        return {
            "success": True,
            "task_number": task_number,
            "commit_sha": commit_sha,
            "preview_url": preview_url,
            "screenshot_path": png_path,
            "auth_receipt": proof["receipt"],
            "page": proof.get("page", {}),
            "next_step": "Open this exact PNG with read_file. Deliver every requested view together in one omnia_dm_reply review.",
        }
    except Exception as exc:
        detail = str(exc)
        if bootstrap_url:
            detail = detail.replace(bootstrap_url, "[one-use session]")
        return {"success": False, "error": f"Capture failed: {detail[:3000]}"}
