"""Reply to the Omnia direct-message thread that triggered the run."""

from collections.abc import Mapping
from typing import Any

from langgraph.config import get_config

from ..utils.omnia import post_omnia_dm_event


async def omnia_dm_reply(message: str) -> dict[str, Any]:
    """Send a concise progress, blocker, question, or final report to Luna's Omnia DM."""
    if not message.strip():
        return {"success": False, "error": "Message cannot be empty"}
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
    success, error = await post_omnia_dm_event(
        {
            "kind": "message",
            "dm_thread_id": thread_id,
            "message": message.strip(),
            "agent_thread_id": configurable.get("thread_id"),
            "run_id": str(run_id) if run_id else None,
            "journal_run_id": omnia_thread.get("journal_run_id"),
        }
    )
    return {"success": success, **({"error": error} if error else {})}
