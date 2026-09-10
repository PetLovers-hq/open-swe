"""Authorized Omnia task and approval actions for Luna."""

import hashlib
import json
import uuid
from typing import Any, Literal

from langgraph.config import get_config

from ..utils.omnia import post_omnia_agent_action


async def omnia_agent_action(
    action: Literal["list_tasks", "create_task", "merge_task", "browser_session"],
    title: str | None = None,
    task_number: int | None = None,
    preview_url: str | None = None,
    redirect_path: str | None = None,
) -> dict[str, Any]:
    """Read/create Omnia tasks or execute an explicitly approved merge.

    For an approved release, call merge_task even if GitHub already shows the
    task merged: it verifies the approved head and restores the live DM handoff.
    A release needs no new post-deployment screenshot. Follow next_action.

    Use omnia_capture_view for preview screenshots. The legacy browser_session
    action returns a migration instruction and never exposes a one-use launcher.
    """
    if action == "browser_session":
        return {
            "success": False,
            "error": "Browser sessions are now owned by omnia_capture_view. Call that tool with task_number, name, wait_for_text, path, viewport, and optional steps/time_zone. It creates and consumes fresh authentication automatically. Do not execute old capture scripts or reuse session URLs.",
        }
    return await _execute_omnia_agent_action(
        action,
        title=title,
        task_number=task_number,
        preview_url=preview_url,
        redirect_path=redirect_path,
    )


async def _execute_omnia_agent_action(
    action: Literal["list_tasks", "create_task", "merge_task", "browser_session"],
    title: str | None = None,
    task_number: int | None = None,
    preview_url: str | None = None,
    redirect_path: str | None = None,
) -> dict[str, Any]:
    config = get_config()
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return {"success": False, "error": "Missing run configuration"}
    omnia_thread = configurable.get("omnia_thread")
    if not isinstance(omnia_thread, dict):
        return {"success": False, "error": "Missing Omnia conversation"}
    if action == "create_task" and not (isinstance(title, str) and title.strip()):
        return {"success": False, "error": "title is required for create_task"}
    if action == "merge_task" and not isinstance(task_number, int):
        return {"success": False, "error": "task_number is required for merge_task"}
    if action == "browser_session" and not (
        isinstance(task_number, int)
        or (isinstance(preview_url, str) and preview_url.startswith("https://"))
    ):
        return {
            "success": False,
            "error": "task_number is required to resolve browser_session (legacy preview_url is also accepted)",
        }
    run_id = config.get("run_id") or configurable.get("run_id") or "unknown-run"
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "action": action,
                "title": title,
                "task_number": task_number,
                "preview_url": preview_url,
                "redirect_path": redirect_path,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    if action in {"browser_session", "merge_task"}:
        # A new merge attempt must not replay a failed request forever. Omnia
        # revalidates approval and recovers an already merged exact head. The
        # same payload key remains stable inside transport-level retries.
        fingerprint = uuid.uuid4().hex
    result = await post_omnia_agent_action(
        {
            "action": action,
            "title": title.strip() if isinstance(title, str) else None,
            "task_number": task_number,
            "preview_url": preview_url,
            "redirect_path": redirect_path,
            "dm_thread_id": omnia_thread.get("thread_id"),
            "sender_email": configurable.get("user_email"),
            "agent_thread_id": configurable.get("thread_id"),
            "idempotency_key": f"open-swe:{run_id}:{fingerprint}",
        }
    )
    if action == "create_task" and result.get("success"):
        task = result.get("task")
        number = task.get("taskNumber") if isinstance(task, dict) else None
        journal = omnia_thread.get("journal_run_id")
        if isinstance(number, int) and isinstance(journal, int):
            result["branch_name"] = f"agent/luna/task-{number}-run-{journal}"
    return result
