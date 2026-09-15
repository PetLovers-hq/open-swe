from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.utils import omnia_usage


@pytest.mark.asyncio
async def test_usage_is_per_run_and_preserves_missing_prices(monkeypatch):
    run_client = SimpleNamespace(
        runs=SimpleNamespace(
            get=AsyncMock(
                return_value={
                    "metadata": {
                        "source": "omnia",
                        "prepare_run_id": "prepare-1",
                        "source_context": {"omnia_thread": {"journal_run_id": 123}},
                    }
                }
            )
        )
    )
    monkeypatch.setattr(omnia_usage, "langgraph_client", lambda: run_client)
    rows = [
        SimpleNamespace(
            id="trace-1", end_time="done", total_cost=1.25, prompt_tokens=100, completion_tokens=10
        )
    ]

    async def traces(**kwargs):
        assert kwargs["is_root"] is True
        assert "prepare-1" in kwargs["filter"]
        for row in rows:
            yield row

    client = MagicMock(list_runs=traces)
    monkeypatch.setattr(omnia_usage, "_build_prod_langsmith_client", lambda: client)
    result = await omnia_usage.read_omnia_run_usage("thread-1", "run-1")
    assert result["cost_usd"] == 1.25
    assert result["journal_run_id"] == 123
    rows[0].total_cost = None
    result = await omnia_usage.read_omnia_run_usage("thread-1", "run-1")
    assert result["status"] == "unavailable"
    assert "cost_usd" not in result
