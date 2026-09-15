"""Exact model choices supplied by Omnia's authenticated administrator action."""

import os
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from ..dashboard.options import FABLE_MODEL_IDS, SUPPORTED_MODEL_IDS, model_supports_effort


class OmniaModelSelection(BaseModel):
    model_id: str
    effort: Literal["high"] = "high"

    @field_validator("model_id")
    @classmethod
    def supported_model(cls, value: str) -> str:
        if (
            value not in SUPPORTED_MODEL_IDS
            or not value.startswith(("openai:", "anthropic:"))
            or not model_supports_effort(value, "high")
        ):
            raise ValueError("Unsupported Omnia model")
        return value


def resolve_omnia_model(configurable: dict[str, Any], *, fable_enabled: bool) -> tuple[str, str]:
    selection = OmniaModelSelection.model_validate(
        configurable.get("omnia_model_selection")
        or {"model_id": "openai:gpt-5.6-luna", "effort": "high"}
    )
    if selection.model_id in FABLE_MODEL_IDS and not fable_enabled:
        raise ValueError("Claude Fable is disabled for this workspace; choose another model")
    return selection.model_id, selection.effort


def check_omnia_model_credentials(model_id: str) -> None:
    from .gateway import gateway_env_default, gateway_overrides

    if gateway_env_default() and gateway_overrides(model_id) is not None:
        return
    key = "ANTHROPIC_API_KEY" if model_id.startswith("anthropic:") else "OPENAI_API_KEY"
    if not os.environ.get(key):
        raise ValueError(f"The cloud runtime needs {key} before this model can be selected")
