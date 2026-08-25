"""Signed Omnia DM trigger for Luna."""

import os
import re
import uuid
from typing import Any, cast

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from langchain_core.messages.content import create_text_block
from pydantic import BaseModel, Field

from ..dispatch import dispatch_agent_run
from ..utils.omnia import post_omnia_dm_event, verify_omnia_signature
from . import common

router = APIRouter()

_SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


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


_SCOPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("inventory", re.compile(r"\b(inventory|stock|sku|asin|fba|wfs|warehouse|freight|reorder)\b", re.I)),
    ("finance", re.compile(r"\b(finance|accounting|invoice|bill|payment|p&l|profit|margin|cogs|settlement)\b", re.I)),
    ("ads", re.compile(r"\b(ppc|advertis(?:e|ing)|campaign|keyword|bid|acos|roas)\b", re.I)),
    ("research", re.compile(r"\b(research|competitor|market research|search the web|live internet)\b", re.I)),
    ("operations", re.compile(r"\b(odoo|purchase order|supplier|shipment|fulfillment|operations)\b", re.I)),
)


def _scope_key(message: str) -> str:
    """Return a conservative code-ownership lane for an Omnia request.

    Unknown, cross-cutting, and product-shell work stays in ``app``. Only a
    request with one unambiguous specialist domain receives a parallel lane;
    messages that mention multiple domains fail safe to the shared app lane.
    """
    matches = [scope for scope, pattern in _SCOPE_PATTERNS if pattern.search(message)]
    return matches[0] if len(matches) == 1 else "app"


def _thread_id(dm_thread_id: str, message: str) -> str:
    scope = _scope_key(message)
    epoch = os.environ.get("OMNIA_THREAD_EPOCH", "").strip()
    # Preserve the original app-lane identity so a deployment does not orphan
    # an in-flight legacy run or its durable checkpoints.
    suffix = f"/scope/{scope}" if scope != "app" else ""
    if epoch:
        suffix += f"/runtime/{epoch}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://omnia.petlovers.com/dm/{dm_thread_id}{suffix}"))


def _repo(event: OmniaDmEvent) -> dict[str, str]:
    return {
        "owner": event.repo_owner or os.environ.get("OMNIA_REPO_OWNER", "PetLovers-hq"),
        "name": event.repo_name or os.environ.get("OMNIA_REPO_NAME", "Omnia"),
    }


def _attachment_context(event: OmniaDmEvent) -> str:
    if not event.attachments:
        return event.message
    lines = [event.message, "", "Omnia attachments:"]
    for attachment in event.attachments:
        name = attachment.get("name", "attachment")
        url = attachment.get("url", "")
        mime = attachment.get("mime", "")
        if url:
            location = "embedded below" if mime.lower().startswith("image/") else url
            lines.append(f"- {name} ({mime}): {location}")
    return "\n".join(lines)


async def _multimodal_content(
    event: OmniaDmEvent,
    model_id: str,
    effort: str,
) -> tuple[str | list[dict[str, Any]], str, str]:
    text = _attachment_context(event)
    images = [
        attachment
        for attachment in event.attachments
        if attachment.get("mime", "").lower().startswith("image/") and attachment.get("url")
    ]
    if not images:
        return text, model_id, effort

    if not common.model_supports_images(model_id):
        model_id, effort = common.default_vision_model_pair()

    blocks: list[dict[str, Any]] = [cast(dict[str, Any], create_text_block(text))]
    async with httpx.AsyncClient(timeout=common.DEFAULT_HTTP_TIMEOUT) as client:
        for attachment in images:
            name = attachment.get("name", "attachment")
            mime = attachment.get("mime", "").lower()
            if mime not in _SUPPORTED_IMAGE_MIME_TYPES:
                blocks.append(
                    cast(
                        dict[str, Any],
                        create_text_block(
                            f"The attached image {name} was not included because {mime or 'its type'} "
                            "is unsupported. Do not claim to have seen it."
                        ),
                    )
                )
                continue
            image_block = await common.fetch_image_block(attachment["url"], client)
            if image_block is None:
                blocks.append(
                    cast(
                        dict[str, Any],
                        create_text_block(
                            f"The attached image {name} could not be loaded. "
                            "Do not claim to have seen it."
                        ),
                    )
                )
                continue
            blocks.append(cast(dict[str, Any], image_block))
    return blocks, model_id, effort


async def process_omnia_dm(event: OmniaDmEvent) -> None:
    thread_id = _thread_id(event.dm_thread_id, event.message)
    repo = _repo(event)
    model_id = os.environ.get("OMNIA_AGENT_MODEL", "openai:gpt-5.6-luna")
    effort = os.environ.get("OMNIA_AGENT_EFFORT", "high")
    content, model_id, effort = await _multimodal_content(event, model_id, effort)
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
        "agent_model_id": model_id,
        "agent_effort": effort,
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
            "omnia_scope": _scope_key(event.message),
        },
        # A new request in the same ownership lane waits behind the current
        # run. It must never interrupt and discard active work.
        multitask_strategy="enqueue",
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
            "agent_thread_id": _thread_id(event.dm_thread_id, event.message),
        }
    )
    return {"status": "accepted", "thread_id": _thread_id(event.dm_thread_id, event.message)}


@router.get("/webhooks/omnia")
async def omnia_webhook_health() -> dict[str, str]:
    return {"status": "ok"}
