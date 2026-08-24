"""Authorized Omnia task and approval actions for Luna."""

import hashlib
import json
from typing import Any, Literal

from langgraph.config import get_config

from ..utils.omnia import post_omnia_agent_action


async def omnia_agent_action(
    action: Literal["list_tasks", "create_task", "merge_task"],
    title: str | None = None,
    task_number: int | None = None,
) -> dict[str, Any]:
    """Read Luna's tasks, create assigned work, or execute an explicitly approved merge.

    Use create_task only when the human clearly hands Luna new coding work. Use merge_task
    only after a clear approval for that exact task in this Omnia conversation.
    """
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
    run_id = config.get("run_id") or configurable.get("run_id") or "unknown-run"
    fingerprint = hashlib.sha256(
        json.dumps(
            {"action": action, "title": title, "task_number": task_number},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    return await post_omnia_agent_action(
        {
            "action": action,
            "title": title.strip() if isinstance(title, str) else None,
            "task_number": task_number,
            "dm_thread_id": omnia_thread.get("thread_id"),
            "sender_email": configurable.get("user_email"),
            "agent_thread_id": configurable.get("thread_id"),
            "idempotency_key": f"open-swe:{run_id}:{fingerprint}",
        }
    )
