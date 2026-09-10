"""End the approval worker after Omnia accepts ownership of deployment."""

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from .ensure_no_empty_msg import get_every_message_since_last_human


class OmniaReleaseHandoffMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        configurable = get_config().get("configurable", {})
        if configurable.get("source") != "omnia":
            return None
        messages = get_every_message_since_last_human(state)
        calls = {
            call["id"]: call.get("args", {})
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
            if call.get("name") == "omnia_agent_action"
            and call.get("args", {}).get("action") == "merge_task"
        }
        for message in reversed(messages):
            if not isinstance(message, ToolMessage) or message.tool_call_id not in calls:
                continue
            try:
                result = json.loads(message.content) if isinstance(message.content, str) else None
            except (ValueError, TypeError):
                continue
            if not isinstance(result, dict) or result.get("success") is not True:
                continue
            merged = result.get("merged")
            task = result.get("task")
            if (
                isinstance(merged, dict)
                and isinstance(merged.get("mergeSha"), str)
                and len(merged["mergeSha"]) == 40
                and isinstance(task, dict)
                and task.get("number") == calls[message.tool_call_id].get("task_number")
                and isinstance(result.get("approval_note_id"), int)
            ):
                # The signed deployment callback owns the DM confirmation. No
                # further model call, polling, screenshot, or blocker is needed.
                return {"jump_to": "end"}
        return None
