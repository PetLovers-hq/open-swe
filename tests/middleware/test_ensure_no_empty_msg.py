from typing import Any
from unittest.mock import MagicMock, patch

from langchain.agents.middleware import AgentState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agent.input_messages import system_input
from agent.middleware.ensure_no_empty_msg import (
    check_if_confirming_completion,
    check_if_model_messaged_user,
    check_if_omnia_terminal_delivered,
    ensure_no_empty_msg,
    get_every_message_since_last_human,
)


class TestGetEveryMessageSinceLastHuman:
    def test_returns_messages_after_last_human(self) -> None:
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="first human"),
                AIMessage(content="ai response"),
                HumanMessage(content="second human"),
                AIMessage(content="final ai"),
            ]
        }

        result = get_every_message_since_last_human(state)

        assert len(result) == 1
        assert result[0].content == "final ai"

    def test_returns_all_messages_when_no_human(self) -> None:
        state: AgentState[Any] = {
            "messages": [
                AIMessage(content="ai 1"),
                AIMessage(content="ai 2"),
            ]
        }

        result = get_every_message_since_last_human(state)

        assert len(result) == 2
        assert result[0].content == "ai 1"
        assert result[1].content == "ai 2"

    def test_returns_empty_when_human_is_last(self) -> None:
        state: AgentState[Any] = {
            "messages": [
                AIMessage(content="ai response"),
                HumanMessage(content="human last"),
            ]
        }

        result = get_every_message_since_last_human(state)

        assert len(result) == 0

    def test_returns_multiple_messages_after_human(self) -> None:
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="human"),
                AIMessage(content="ai 1"),
                ToolMessage(content="tool result", tool_call_id="123"),
                AIMessage(content="ai 2"),
            ]
        }

        result = get_every_message_since_last_human(state)

        assert len(result) == 3
        assert result[0].content == "ai 1"
        assert result[1].content == "tool result"
        assert result[2].content == "ai 2"


class TestCheckIfModelMessagedUser:
    def test_returns_true_for_slack_thread_reply(self) -> None:
        messages: list[AnyMessage] = [
            ToolMessage(content="sent", tool_call_id="123", name="slack_thread_reply"),
        ]

        assert check_if_model_messaged_user(messages) is True

    def test_returns_true_for_linear_comment(self) -> None:
        messages: list[AnyMessage] = [
            ToolMessage(content="commented", tool_call_id="123", name="linear_comment"),
        ]

        assert check_if_model_messaged_user(messages) is True

    def test_returns_false_for_other_tools(self) -> None:
        messages: list[AnyMessage] = [
            ToolMessage(content="result", tool_call_id="123", name="bash"),
            ToolMessage(content="result", tool_call_id="456", name="read_file"),
        ]

        assert check_if_model_messaged_user(messages) is False

    def test_returns_false_for_empty_list(self) -> None:
        assert check_if_model_messaged_user([]) is False


class TestCheckIfConfirmingCompletion:
    def test_returns_true_when_confirming_completion_called(self) -> None:
        messages: list[AnyMessage] = [
            ToolMessage(content="confirmed", tool_call_id="123", name="confirming_completion"),
        ]

        assert check_if_confirming_completion(messages) is True

    def test_returns_false_for_other_tools(self) -> None:
        messages: list[AnyMessage] = [
            ToolMessage(content="result", tool_call_id="123", name="bash"),
        ]

        assert check_if_confirming_completion(messages) is False

    def test_returns_false_for_empty_list(self) -> None:
        assert check_if_confirming_completion([]) is False

    def test_finds_confirming_completion_among_other_messages(self) -> None:
        messages = [
            AIMessage(content="working"),
            ToolMessage(content="done", tool_call_id="1", name="bash"),
            ToolMessage(content="confirmed", tool_call_id="2", name="confirming_completion"),
            AIMessage(content="finished"),
        ]

        assert check_if_confirming_completion(messages) is True


class TestEnsureNoEmptyMsgNotify:
    def _make_runtime(self) -> MagicMock:
        return MagicMock()

    def test_returns_none_when_user_messaged(self) -> None:
        empty_ai = AIMessage(content="")
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="fix the bug"),
                ToolMessage(content="message sent", tool_call_id="1", name="slack_thread_reply"),
                empty_ai,
            ]
        }

        result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is None

    def test_returns_none_with_linear_comment(self) -> None:
        empty_ai = AIMessage(content="")
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="fix the bug"),
                ToolMessage(content="commented", tool_call_id="1", name="linear_comment"),
                empty_ai,
            ]
        }

        result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is None

    def test_injects_no_op_when_user_not_messaged(self) -> None:
        empty_ai = AIMessage(content="")
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="fix the bug"),
                ToolMessage(content="result", tool_call_id="1", name="bash"),
                empty_ai,
            ]
        }

        result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is not None
        assert len(result["messages"]) == 2
        assert result["messages"][0].tool_calls[0]["name"] == "no_op"

    def test_omnia_progress_and_prior_no_op_cannot_end_turn(self) -> None:
        empty_ai = AIMessage(content="")
        progress_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "omnia_dm_reply",
                    "args": {"message": "Still working."},
                    "id": "progress-1",
                }
            ],
        )
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="finish the task"),
                progress_call,
                ToolMessage(
                    content='{"success": true}',
                    tool_call_id="progress-1",
                    name="omnia_dm_reply",
                ),
                ToolMessage(content="No operation performed.", tool_call_id="old", name="no_op"),
                empty_ai,
            ]
        }

        with patch(
            "agent.middleware.ensure_no_empty_msg.get_config",
            return_value={"configurable": {"source": "omnia"}},
        ):
            result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is not None
        assert result["messages"][0].tool_calls[0]["name"] == "confirming_completion"

    def test_omnia_successful_terminal_delivery_can_end_turn(self) -> None:
        terminal_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "omnia_dm_reply",
                    "args": {"message": "Blocked.", "terminal_outcome": "blocker"},
                    "id": "terminal-1",
                }
            ],
        )
        messages: list[AnyMessage] = [
            terminal_call,
            ToolMessage(
                content='{"success": true}',
                tool_call_id="terminal-1",
                name="omnia_dm_reply",
            ),
        ]
        assert check_if_omnia_terminal_delivered(messages) is True

        state: AgentState[Any] = {
            "messages": [HumanMessage(content="finish the task"), *messages, AIMessage(content="")]
        }
        with patch(
            "agent.middleware.ensure_no_empty_msg.get_config",
            return_value={"configurable": {"source": "omnia"}},
        ):
            result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is None

    def test_omnia_failed_terminal_delivery_cannot_end_turn(self) -> None:
        messages: list[AnyMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "omnia_dm_reply",
                        "args": {"message": "Blocked.", "terminal_outcome": "blocker"},
                        "id": "terminal-1",
                    }
                ],
            ),
            ToolMessage(
                content='{"success": false, "error": "callback failed"}',
                tool_call_id="terminal-1",
                name="omnia_dm_reply",
            ),
        ]

        assert check_if_omnia_terminal_delivered(messages) is False

    def test_returns_none_when_only_user_messaged(self) -> None:
        empty_ai = AIMessage(content="")
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="fix the bug"),
                ToolMessage(content="message sent", tool_call_id="1", name="slack_thread_reply"),
                empty_ai,
            ]
        }

        result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is None

    def test_skips_confirming_completion_for_dashboard_source(self) -> None:
        ai = AIMessage(content="Hi! How can I help?")
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content="hello"),
                ai,
            ]
        }

        with patch(
            "agent.middleware.ensure_no_empty_msg.get_config",
            return_value={"configurable": {"source": "dashboard"}},
        ):
            result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is None
        assert not ai.tool_calls

    def test_skips_confirming_completion_for_dashboard_handoff(self) -> None:
        ai = AIMessage(content="Done in web.")
        handoff = system_input(
            "Continue responses in Web.",
            {
                "sender_id": "system:dashboard-handoff",
                "surface": "automation",
                "kind": "system",
            },
        )["content"]
        assert isinstance(handoff, str)
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content=handoff),
                ai,
            ]
        }

        with patch(
            "agent.middleware.ensure_no_empty_msg.get_config",
            return_value={"configurable": {"source": "slack"}},
        ):
            result = ensure_no_empty_msg.after_model(state, self._make_runtime())

        assert result is None
        assert not ai.tool_calls
