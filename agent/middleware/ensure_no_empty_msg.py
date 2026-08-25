from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentState, after_model
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from ..input_messages import message_sender_id
from ..utils.dashboard_handoff import DASHBOARD_HANDOFF_SENDER_ID

_DASHBOARD_SOURCE = "dashboard"
_OMNIA_SOURCE = "omnia"


def get_every_message_since_last_human(state: AgentState) -> list[AnyMessage]:
    messages = state["messages"]
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            last_human_idx = i
            break
    return messages[last_human_idx + 1 :]


def check_if_model_messaged_user(messages: list[AnyMessage]) -> bool:
    for msg in messages:
        if msg.type == "tool" and msg.name in [
            "slack_thread_reply",
            "linear_comment",
        ]:
            return True
    return False


def check_if_confirming_completion(messages: list[AnyMessage]) -> bool:
    for msg in messages:
        if msg.type == "tool" and msg.name == "confirming_completion":
            return True
    return False


def check_if_no_op(messages: list[AnyMessage]) -> bool:
    for msg in messages:
        if msg.type == "tool" and msg.name == "no_op":
            return True
    return False


def check_if_omnia_terminal_delivered(messages: list[AnyMessage]) -> bool:
    successful_tool_ids = {
        msg.tool_call_id
        for msg in messages
        if isinstance(msg, ToolMessage)
        and (
            '"success": true' in str(msg.content).lower()
            or "'success': true" in str(msg.content).lower()
        )
    }
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls:
            if call.get("name") != "omnia_dm_reply" or call.get("id") not in successful_tool_ids:
                continue
            args = call.get("args")
            if isinstance(args, dict) and (
                args.get("completion") is True
                or args.get("terminal_outcome") in {"blocker", "failure"}
            ):
                return True
    return False


def _last_human_is_dashboard_handoff(state: AgentState) -> bool:
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return message_sender_id(msg.content) == DASHBOARD_HANDOFF_SENDER_ID
    return False


def _is_dashboard_source() -> bool:
    try:
        config = get_config()
    except RuntimeError:
        return False
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return False
    return configurable.get("source") == _DASHBOARD_SOURCE


def _is_omnia_source() -> bool:
    try:
        config = get_config()
    except RuntimeError:
        return False
    configurable = config.get("configurable", {})
    return isinstance(configurable, dict) and configurable.get("source") == _OMNIA_SOURCE


def _force_omnia_terminal(last_msg: AIMessage) -> dict[str, Any]:
    tc_id = str(uuid4())
    last_msg.tool_calls = [{"name": "confirming_completion", "args": {}, "id": tc_id}]
    reminder = ToolMessage(
        content=(
            "This Omnia run cannot end yet. The current user turn has no successfully delivered "
            "terminal omnia_dm_reply. Continue until you call it with completion=True, "
            'terminal_outcome="blocker", or terminal_outcome="failure".'
        ),
        name="confirming_completion",
        tool_call_id=tc_id,
    )
    return {"messages": [last_msg, reminder]}


@after_model
def ensure_no_empty_msg(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        return None
    has_contents = bool(last_msg.text)
    has_tool_calls = bool(last_msg.tool_calls)
    if not has_tool_calls and not has_contents:
        messages_since_last_human = get_every_message_since_last_human(state)
        if _is_omnia_source():
            if not check_if_omnia_terminal_delivered(messages_since_last_human):
                return _force_omnia_terminal(last_msg)
            return None
        if check_if_no_op(messages_since_last_human):
            return None

        if check_if_model_messaged_user(messages_since_last_human):
            return None

        tc_id = str(uuid4())
        last_msg.tool_calls = [{"name": "no_op", "args": {}, "id": tc_id}]
        no_op_tool_msg = ToolMessage(
            content="No operation performed."
            + "Please continue with the task, ensuring you ALWAYS call at least one tool in"
            + " every message unless you are absolutely sure the task has been fully completed.",
            tool_call_id=tc_id,
        )

        return {"messages": [last_msg, no_op_tool_msg]}

    if has_contents and not has_tool_calls:
        messages_since_last_human = get_every_message_since_last_human(state)

        if _is_omnia_source():
            if not check_if_omnia_terminal_delivered(messages_since_last_human):
                return _force_omnia_terminal(last_msg)
            return None

        if (
            check_if_model_messaged_user(messages_since_last_human)
            or check_if_confirming_completion(messages_since_last_human)
            or _is_dashboard_source()
            or _last_human_is_dashboard_handoff(state)
        ):
            return None

        tc_id = str(uuid4())
        last_msg.tool_calls = [{"name": "confirming_completion", "args": {}, "id": tc_id}]
        no_op_tool_msg = ToolMessage(
            content="Confirming task completion. I see you did not call a tool, which would end the task, however you haven't called a tool to message the user or open a pull request."
            + "This may indicate premature termination - please ensure you fully complete the task before ending it. "
            + "If you do not call any tools it will end the task.",
            name="confirming_completion",
            tool_call_id=tc_id,
        )

        return {"messages": [last_msg, no_op_tool_msg]}

    return None
