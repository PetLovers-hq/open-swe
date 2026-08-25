"""Reply to the Omnia direct-message thread that triggered the run."""

import base64
import posixpath
from collections.abc import Mapping
from typing import Any, Literal

from langgraph.config import get_config

from ..utils.omnia import post_omnia_dm_event
from .create_sandbox_file_download_url import _resolve_sandbox_file

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_MAX_REVIEW_PNG_BYTES = 3 * 1024 * 1024


def _download_bytes(result: Any) -> bytes | None:
    for attr in ("content", "data", "bytes"):
        value = result.get(attr) if isinstance(result, dict) else getattr(result, attr, None)
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode()
    return None


async def _native_png(file_path: str) -> dict[str, str]:
    backend, resolved_path, _ = await _resolve_sandbox_file(file_path)
    if not resolved_path.lower().endswith(".png"):
        raise ValueError("Omnia review evidence must be an actual PNG file")
    downloads = await backend.adownload_files([resolved_path])
    content = _download_bytes(downloads[0]) if downloads else None
    if content is None:
        raise ValueError("Could not read the PNG from the sandbox")
    if not content.startswith(_PNG_MAGIC):
        raise ValueError("Review evidence has a .png name but is not PNG data")
    if len(content) > _MAX_REVIEW_PNG_BYTES:
        raise ValueError("Review PNG is over the 3 MB Omnia callback limit")
    return {
        "name": posixpath.basename(resolved_path),
        "mime": "image/png",
        "data_base64": base64.b64encode(content).decode(),
    }


async def omnia_dm_reply(
    message: str,
    screenshot_path: str | None = None,
    completion: bool = False,
    task_number: int | None = None,
    commit_sha: str | None = None,
    preview_url: str | None = None,
    auth_receipt: str | None = None,
    passed_checks: list[str] | None = None,
    terminal_outcome: Literal["blocker", "failure"] | None = None,
) -> dict[str, Any]:
    """Send a human-readable update to Luna's Omnia DM.

    For every successful coding completion, screenshot_path is required and must point to a real
    PNG captured from the working product. Omnia stores it as a native, previewable chat attachment.
    Never substitute an SVG, mockup, GitHub link, or sandbox download URL.
    """
    if not message.strip():
        return {"success": False, "error": "Message cannot be empty"}
    if completion and screenshot_path is None:
        return {
            "success": False,
            "error": "A successful coding completion requires a real PNG screenshot_path",
        }
    if completion and terminal_outcome is not None:
        return {"success": False, "error": "A completion cannot also be a blocker or failure"}
    if completion and (
        not isinstance(task_number, int)
        or not isinstance(commit_sha, str)
        or len(commit_sha) != 40
        or not isinstance(preview_url, str)
        or not preview_url.startswith("https://")
        or not isinstance(auth_receipt, str)
        or len(auth_receipt) != 36
        or len(passed_checks or []) < 2
    ):
        return {
            "success": False,
            "error": "Completion requires task, exact commit, ready preview, authenticated visual-proof receipt, and at least two passed checks",
        }
    config: Mapping[str, Any] = get_config()
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return {"success": False, "error": "Missing run configuration"}
    omnia_thread = configurable.get("omnia_thread")
    if not isinstance(omnia_thread, dict):
        return {"success": False, "error": "Missing omnia_thread configuration"}
    thread_id = omnia_thread.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        return {"success": False, "error": "Missing Omnia DM thread id"}
    run_id = config.get("run_id") or configurable.get("run_id")
    attachments: list[dict[str, str]] = []
    if screenshot_path is not None:
        try:
            attachments.append(await _native_png(screenshot_path))
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
    payload: dict[str, Any] = {
        "kind": "message",
        "dm_thread_id": thread_id,
        "message": message.strip(),
        "agent_thread_id": configurable.get("thread_id"),
        "run_id": str(run_id) if run_id else None,
        "event_id": omnia_thread.get("event_id"),
        "journal_run_id": omnia_thread.get("journal_run_id"),
        "attachments": attachments,
        "purpose": "review"
        if completion
        else "blocker"
        if terminal_outcome == "blocker"
        else "progress",
    }
    if terminal_outcome == "failure":
        payload["terminal_status"] = "error"
    if completion:
        payload["evidence"] = {
            "task_number": task_number,
            "commit_sha": commit_sha,
            "preview_url": preview_url,
            "auth_receipt": auth_receipt,
            "checks": [{"name": name, "passed": True} for name in passed_checks or []],
        }
    success, error = await post_omnia_dm_event(payload)
    return {"success": success, **({"error": error} if error else {})}
