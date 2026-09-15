"""Read traced usage for one Omnia run, never the cumulative conversation."""

import math
from typing import Any

from .langsmith import _build_prod_langsmith_client, _langsmith_metadata_filter
from .thread_ops import langgraph_client
from .tracing import AGENT_TRACING_PROJECT


async def read_omnia_run_usage(thread_id: str, run_id: str) -> dict[str, Any]:
    run = await langgraph_client().runs.get(thread_id, run_id)
    metadata = run.get("metadata") or {}
    context = metadata.get("source_context", {}).get("omnia_thread", {})
    journal_id = context.get("journal_run_id")
    if metadata.get("source") != "omnia" or not isinstance(journal_id, int):
        raise ValueError("This is not a journaled Omnia run")
    prepare_id = metadata.get("prepare_run_id")
    client = _build_prod_langsmith_client()
    if not prepare_id or client is None:
        return {"status": "unavailable", "journal_run_id": journal_id}
    roots = [
        trace
        async for trace in client.list_runs(
            project_name=AGENT_TRACING_PROJECT,
            is_root=True,
            filter=_langsmith_metadata_filter("prepare_run_id", str(prepare_id)),
            select=["id", "end_time", "total_cost", "prompt_tokens", "completion_tokens"],
            limit=100,
        )
    ]
    if not roots or len(roots) >= 100 or any(trace.end_time is None for trace in roots):
        return {"status": "pending", "journal_run_id": journal_id}
    costs = [float(trace.total_cost) if trace.total_cost is not None else None for trace in roots]
    if any(cost is None or not math.isfinite(cost) or cost < 0 for cost in costs):
        return {"status": "unavailable", "journal_run_id": journal_id}
    return {
        "status": "reported",
        "journal_run_id": journal_id,
        "runtime_run_id": run_id,
        "cost_usd": sum(cost for cost in costs if cost is not None),
        "tokens_input": sum(
            trace.prompt_tokens for trace in roots if trace.prompt_tokens is not None
        )
        if all(trace.prompt_tokens is not None for trace in roots)
        else None,
        "tokens_output": sum(
            trace.completion_tokens for trace in roots if trace.completion_tokens is not None
        )
        if all(trace.completion_tokens is not None for trace in roots)
        else None,
        "source": "langsmith-run-roots",
        "source_ids": [str(trace.id) for trace in roots],
    }
