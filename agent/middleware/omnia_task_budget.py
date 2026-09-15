"""The signed task deadline bounds calls; retries cannot grant more time."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config


class OmniaTaskBudgetExceeded(TimeoutError):
    pass


def remaining_task_seconds(configurable: dict[str, Any]) -> float | None:
    if configurable.get("source") != "omnia":
        return None
    deadline = configurable.get("omnia_deadline_at")
    if deadline is None:
        return None  # Legacy runs remain bounded by Omnia's durable watchdog.
    remaining = float(deadline) - time.time()
    if remaining <= 0:
        raise OmniaTaskBudgetExceeded("Luna exceeded the 25-minute task budget.")
    return remaining


class OmniaTaskBudgetMiddleware(AgentMiddleware):
    async def _bounded(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        remaining = remaining_task_seconds(get_config().get("configurable", {}))
        if remaining is None:
            return await handler(request)
        try:
            async with asyncio.timeout(remaining):
                return await handler(request)
        except TimeoutError:
            remaining_task_seconds(get_config().get("configurable", {}))
            raise

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await self._bounded(request, handler)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await self._bounded(request, handler)
