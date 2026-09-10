import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware import AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.middleware.omnia_release_handoff import OmniaReleaseHandoffMiddleware


def state(result: dict[str, Any], action: str = "merge_task") -> AgentState:
    return {
        "messages": [
            HumanMessage(content="Yes, please make it live."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "omnia_agent_action",
                        "args": {"action": action, "task_number": 18},
                        "id": "merge",
                    }
                ],
            ),
            ToolMessage(
                content=json.dumps(result), tool_call_id="merge", name="omnia_agent_action"
            ),
        ]
    }


def receipt() -> dict[str, Any]:
    return {
        "success": True,
        "merged": {"mergeSha": "a" * 40},
        "task": {"number": 18},
        "approval_note_id": 100,
    }


async def run(value: AgentState, source: str = "omnia") -> dict[str, Any] | None:
    with patch(
        "agent.middleware.omnia_release_handoff.get_config",
        return_value={"configurable": {"source": source}},
    ):
        return await OmniaReleaseHandoffMiddleware().abefore_model(value, MagicMock())


async def test_ends_without_another_model_call_after_verified_merge() -> None:
    assert await run(state(receipt())) == {"jump_to": "end"}


@pytest.mark.parametrize(
    "override",
    [{"success": False}, {"task": {"number": 19}}, {"merged": None}, {"approval_note_id": None}],
)
async def test_does_not_end_for_failed_or_mismatched_merge(override: dict[str, Any]) -> None:
    assert await run(state({**receipt(), **override})) is None


async def test_does_not_end_for_prose_or_other_actions() -> None:
    assert await run(state({"success": True, "message": "Merged successfully"})) is None
    assert await run(state(receipt(), "browser_session")) is None
    assert await run(state(receipt()), "slack") is None


async def test_a_previous_turn_merge_cannot_end_new_work() -> None:
    value = state(receipt())
    value["messages"].append(HumanMessage(content="New request: add task search."))
    assert await run(value) is None


async def test_real_graph_stops_before_invoking_the_model() -> None:
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    model = FakeListChatModel(responses=["should not be invoked"])
    graph = create_agent(model=model, middleware=[OmniaReleaseHandoffMiddleware()])
    with patch.object(
        FakeListChatModel, "_call", side_effect=AssertionError("unexpected model call")
    ):
        result = await graph.ainvoke(
            {"messages": [*state(receipt())["messages"]]},
            config={"configurable": {"source": "omnia"}},
        )
    assert len(result["messages"]) == 3
    assert isinstance(result["messages"][-1], ToolMessage)
