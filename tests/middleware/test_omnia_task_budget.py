import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.middleware.omnia_task_budget import OmniaTaskBudgetExceeded, OmniaTaskBudgetMiddleware


@pytest.mark.parametrize("method", ["awrap_model_call", "awrap_tool_call"])
async def test_deadline_cancels_a_stalled_call_and_cannot_reset_on_retry(method):
    completed = False

    async def handler(_request):
        nonlocal completed
        await asyncio.sleep(1)
        completed = True

    config = {"configurable": {"source": "omnia", "omnia_deadline_at": time.time() + 0.015}}
    with patch("agent.middleware.omnia_task_budget.get_config", return_value=config):
        middleware = OmniaTaskBudgetMiddleware()
        with pytest.raises(OmniaTaskBudgetExceeded):
            await getattr(middleware, method)(MagicMock(), handler)
        with pytest.raises(OmniaTaskBudgetExceeded):
            await getattr(OmniaTaskBudgetMiddleware(), method)(MagicMock(), handler)
    assert not completed
