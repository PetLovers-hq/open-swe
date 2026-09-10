"""Reply to the Omnia direct-message thread that triggered the run."""

import base64
import json
import posixpath
from collections.abc import Mapping
from typing import Any, Literal

from langgraph.config import get_config
from typing_extensions import TypedDict

from ..utils.omnia import post_omnia_dm_event
from .create_sandbox_file_download_url import _resolve_sandbox_file

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_MAX_REVIEW_PNG_BYTES = 3 * 1024 * 1024


class ReviewScreenshot(TypedDict):
    screenshot_path: str
    auth_receipt: str


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
    screenshots: list[ReviewScreenshot] | None = None,
) -> dict[str, Any]:
    """Send a human-readable update to Luna's Omnia DM.

    Every coding completion requires real PNGs captured from the working product.
    Use screenshots=[{"screenshot_path": "/absolute/desktop.png", "auth_receipt": "..."},
    {"screenshot_path": "/absolute/phone.png", "auth_receipt": "..."}] to deliver ALL requested
    images together in ONE review. Each needs its own receipt for the same task, commit and preview.
    One to five images are supported, at most 3 MB total. Never claim an image is attached unless
    it is included in this call. Omnia publishes the entire set atomically, or none of it.
    For one image, the legacy screenshot_path plus auth_receipt arguments also work.
    Open the exact PNG with read_file and visually inspect it before sending.
    Only claim states visible in that image; after rejection capture a new image.
    Never substitute an SVG, mockup, GitHub link, or sandbox download URL.
    The PNG must be the exact saved bytes hashed for auth_receipt. A different
    capture of the same screen is not interchangeable. If Omnia rejects the
    receipt, save a fresh capture, register its hash in the authenticated browser,
    and retry with that exact file and new receipt. Never make the user repair it.
    """
    if not message.strip():
        return {"success": False, "error": "Message cannot be empty"}
    if screenshots is not None and screenshot_path is not None:
        return {"success": False, "error": "Use screenshots or screenshot_path, not both"}
    if screenshots is not None and (
        not 1 <= len(screenshots) <= 5
        or any(
            not item.get("screenshot_path") or len(item.get("auth_receipt", "")) != 36
            for item in screenshots
        )
    ):
        return {
            "success": False,
            "error": "Provide one to five screenshots, each with a path and its own auth_receipt",
        }
    primary_receipt = screenshots[0]["auth_receipt"] if screenshots else auth_receipt
    if completion and screenshot_path is None and not screenshots:
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
        or not isinstance(primary_receipt, str)
        or len(primary_receipt) != 36
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
    try:
        if screenshots:
            if len({item["auth_receipt"] for item in screenshots}) != len(screenshots):
                raise ValueError("Each screenshot requires a distinct visual-proof receipt")
            for item in screenshots:
                attachment = await _native_png(item["screenshot_path"])
                attachment["auth_receipt"] = item["auth_receipt"]
                attachments.append(attachment)
        elif screenshot_path is not None:
            attachments.append(await _native_png(screenshot_path))
        if (
            sum(len(base64.b64decode(item["data_base64"])) for item in attachments)
            > _MAX_REVIEW_PNG_BYTES
        ):
            raise ValueError(
                "Review screenshots together exceed 3 MB; reduce size and register fresh receipts"
            )
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
            "auth_receipt": primary_receipt,
            "checks": [{"name": name, "passed": True} for name in passed_checks or []],
        }
    if len(json.dumps(payload).encode()) > 4_400_000:
        return {
            "success": False,
            "error": "Review exceeds the callback size limit; reduce size and register fresh receipts",
        }
    success, error = await post_omnia_dm_event(payload)
    return {"success": success, **({"error": error} if error else {})}
