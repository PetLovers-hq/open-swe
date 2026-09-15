from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from agent.utils.omnia_models import OmniaModelSelection, resolve_omnia_model
from agent.webhooks import omnia_routes


def test_default_stays_luna_high_despite_profile_override():
    assert resolve_omnia_model(
        {"agent_model_id": "anthropic:claude-opus-5"}, fable_enabled=True
    ) == ("openai:gpt-5.6-luna", "high")


@pytest.mark.parametrize(
    "model", ["anthropic:claude-sonnet-5", "anthropic:claude-opus-5", "openai:gpt-5.6-sol"]
)
def test_saved_choice_wins_over_previous_thread_settings(model):
    assert resolve_omnia_model(
        {
            "agent_model_id": "openai:gpt-5.6-luna",
            "omnia_model_selection": {"model_id": model, "effort": "high"},
        },
        fable_enabled=True,
    ) == (model, "high")


def test_disabled_fable_fails_without_substitution():
    with pytest.raises(ValueError, match="disabled"):
        resolve_omnia_model(
            {"omnia_model_selection": {"model_id": "anthropic:claude-fable-5"}}, fable_enabled=False
        )


def test_invalid_choice_and_effort_rejected():
    with pytest.raises(ValidationError):
        OmniaModelSelection(model_id="invented:model")
    with pytest.raises(ValidationError):
        OmniaModelSelection.model_validate({"model_id": "openai:gpt-5.6-luna", "effort": "low"})


@pytest.mark.asyncio
async def test_webhook_dispatch_retains_exact_selected_model(monkeypatch):
    dispatch = AsyncMock()
    monkeypatch.setattr(omnia_routes, "dispatch_agent_run", dispatch)
    monkeypatch.setattr(omnia_routes.common, "upsert_agent_thread_owner_metadata", AsyncMock())
    event = omnia_routes.OmniaDmEvent.model_validate(
        {
            "event_id": "test",
            "dm_thread_id": "test",
            "sender_id": "test",
            "message": "test",
            "model_selection": {"model_id": "anthropic:claude-opus-5", "effort": "high"},
        }
    )
    await omnia_routes.process_omnia_dm(event)
    assert dispatch.await_args is not None
    config = dispatch.await_args.args[2]
    assert config["agent_model_id"] == "anthropic:claude-opus-5"
    assert resolve_omnia_model(config, fable_enabled=True) == ("anthropic:claude-opus-5", "high")
    assert (
        dispatch.await_args.kwargs["metadata"]["omnia_model_selection"]
        == config["omnia_model_selection"]
    )
