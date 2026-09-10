from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.middleware.sanitize_openai_responses import _sanitize_messages


def test_stateless_responses_replay_preserves_tool_history_without_mutation() -> None:
    messages = [
        HumanMessage("test the todo middleware"),
        AIMessage(
            content=[
                {
                    "type": "reasoning",
                    "id": "rs_unstored",
                    "summary": [],
                    "encrypted_content": None,
                },
                {
                    "type": "reasoning",
                    "id": "rs_reasoning",
                    "summary": [],
                    "encrypted_content": "encrypted-reasoning",
                },
                {
                    "type": "function_call",
                    "id": "fc_execute",
                    "call_id": "call_execute",
                    "name": "execute",
                    "arguments": '{"command":"pwd"}',
                },
            ],
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "pwd"},
                    "id": "call_execute",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="/workspace/open-swe",
            tool_call_id="call_execute",
            name="execute",
        ),
        HumanMessage("continue"),
    ]
    original_messages = deepcopy(messages)
    model = ChatOpenAI(
        model="gpt-5.6-sol",
        api_key=SecretStr("test"),
        use_responses_api=True,
        store=False,
        include=["reasoning.encrypted_content"],
        output_version="responses/v1",
    )

    first_payload = model._get_request_payload(_sanitize_messages(messages))
    second_payload = model._get_request_payload(_sanitize_messages(messages))

    expected_call = {
        "type": "function_call",
        "id": "fc_execute",
        "call_id": "call_execute",
        "name": "execute",
        "arguments": '{"command":"pwd"}',
    }
    expected_output = {
        "type": "function_call_output",
        "output": "/workspace/open-swe",
        "call_id": "call_execute",
    }
    assert not any(item.get("id") == "rs_unstored" for item in first_payload["input"])
    assert expected_call in first_payload["input"]
    assert expected_output in first_payload["input"]
    assert second_payload["input"] == first_payload["input"]
    assert messages == original_messages


def test_invalid_review_call_gets_error_result_without_execution_or_mutation() -> None:
    malformed = '{"message":"Fresh screenshots are attached for review.'
    invalid = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "fc_rejected",
                "call_id": "call_rejected",
                "name": "omnia_dm_reply",
                "arguments": malformed,
            }
        ],
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "name": "omnia_dm_reply",
                "id": "call_rejected",
                "args": malformed,
                "error": "Invalid JSON",
            }
        ],
        tool_calls=[
            {
                "type": "tool_call",
                "name": "confirming_completion",
                "id": "reminder",
                "args": {},
            }
        ],
    )
    messages = [
        HumanMessage("Show phone and desktop screenshots"),
        invalid,
        ToolMessage("Continue until delivered", tool_call_id="reminder"),
    ]
    original = deepcopy(messages)
    model = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=SecretStr("test"),
        use_responses_api=True,
        store=False,
        output_version="responses/v1",
    )
    sanitized = _sanitize_messages(messages)
    payload = model._get_request_payload(sanitized)["input"]
    calls = {item["call_id"] for item in payload if item.get("type") == "function_call"}
    outputs = [item for item in payload if item.get("type") == "function_call_output"]
    assert calls == {item["call_id"] for item in outputs} == {"call_rejected", "reminder"}
    error = next(item for item in outputs if item["call_id"] == "call_rejected")
    assert "NOT executed" in error["output"]
    assert "no message or screenshot was sent" in error["output"]
    assert (
        next(
            m for m in sanitized if isinstance(m, ToolMessage) and m.tool_call_id == "call_rejected"
        ).status
        == "error"
    )
    assert _sanitize_messages(sanitized) == sanitized
    assert messages == original


def test_existing_invalid_call_result_is_preserved_and_valid_calls_are_not_fabricated() -> None:
    messages = [
        AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "execute",
                    "id": "bad",
                    "args": "{",
                    "error": "Invalid JSON",
                }
            ],
            tool_calls=[{"name": "execute", "id": "valid", "args": {"command": "pwd"}}],
        ),
        ToolMessage("Original validation failure", tool_call_id="bad", status="error"),
    ]
    sanitized = _sanitize_messages(messages)
    assert sanitized == messages
    assert not any(isinstance(m, ToolMessage) and m.tool_call_id == "valid" for m in sanitized)
