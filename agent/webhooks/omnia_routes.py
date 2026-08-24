"""Signed Omnia DM trigger for Luna."""

import os
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from ..dispatch import dispatch_agent_run
from ..utils.omnia import post_omnia_dm_event, verify_omnia_signature
from . import common

router = APIRouter()


class OmniaDmEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    dm_thread_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=100_000)
    sender_id: str = Field(min_length=1, max_length=200)
    sender_name: str | None = Field(default=None, max_length=200)
    sender_email: str | None = Field(default=None, max_length=320)
    github_login: str | None = Field(default=None, max_length=100)
    repo_owner: str | None = Field(default=None, max_length=100)
    repo_name: str | None = Field(default=None, max_length=100)
    attachments: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    journal_run_id: int | None = Field(default=None, gt=0)


def _thread_id(dm_thread_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://omnia.petlovers.com/dm/{dm_thread_id}"))


def _repo(event: OmniaDmEvent) -> dict[str, str]:
    return {
        "owner": event.repo_owner or os.environ.get("OMNIA_REPO_OWNER", "PetLovers-hq"),
        "name": event.repo_name or os.environ.get("OMNIA_REPO_NAME", "Omnia"),
    }


async def process_omnia_dm(event: OmniaDmEvent) -> None:
    thread_id = _thread_id(event.dm_thread_id)
    repo = _repo(event)
    omnia_thread: dict[str, Any] = {
        "thread_id": event.dm_thread_id,
        "event_id": event.event_id,
        "sender_id": event.sender_id,
        "journal_run_id": event.journal_run_id,
    }
    if event.sender_name:
        omnia_thread["sender_name"] = event.sender_name
    configurable: dict[str, Any] = {
        "repo": repo,
        "source": "omnia",
        "omnia_thread": omnia_thread,
        "agent_model_id": os.environ.get("OMNIA_AGENT_MODEL", "openai:gpt-5.6-luna"),
        "agent_effort": os.environ.get("OMNIA_AGENT_EFFORT", "high"),
    }
    if event.sender_email:
        configurable["user_email"] = event.sender_email
    if event.github_login:
        configurable["github_login"] = event.github_login
    source_context = {"omnia_thread": omnia_thread}
    await common.upsert_agent_thread_owner_metadata(
        thread_id,
        source="omnia",
        repo_config=repo,
        github_login=event.github_login or "",
        user_email=event.sender_email or "",
        title=event.message,
        source_context=source_context,
    )
    content = event.message
    if event.attachments:
        lines = ["\n\nOmnia attachments:"]
        for attachment in event.attachments:
            name = attachment.get("name", "attachment")
            url = attachment.get("url", "")
            mime = attachment.get("mime", "")
            if url:
                lines.append(f"- {name} ({mime}): {url}")
        content += "\n".join(lines)
    await dispatch_agent_run(
        thread_id,
        content,
        configurable,
        source="omnia",
        metadata={
            **common._AGENT_VERSION_METADATA,
            "source": "omnia",
            "source_context": source_context,
            "repo": repo,
            "event_id": event.event_id,
        },
    )


@router.post("/webhooks/omnia")
async def omnia_webhook(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    body = await request.body()
    if not verify_omnia_signature(body, request.headers.get("X-Omnia-Signature")):
        raise HTTPException(status_code=401, detail="Invalid signature")
    try:
        event = OmniaDmEvent.model_validate_json(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid Omnia DM event") from exc
    repo = _repo(event)
    if not common._is_repo_allowed(repo):
        raise HTTPException(status_code=403, detail="Repository not allowed")
    background_tasks.add_task(process_omnia_dm, event)
    await post_omnia_dm_event(
        {
            "kind": "run_status",
            "status": "accepted",
            "dm_thread_id": event.dm_thread_id,
            "event_id": event.event_id,
            "agent_thread_id": _thread_id(event.dm_thread_id),
        }
    )
    return {"status": "accepted", "thread_id": _thread_id(event.dm_thread_id)}


@router.get("/webhooks/omnia")
async def omnia_webhook_health() -> dict[str, str]:
    return {"status": "ok"}
