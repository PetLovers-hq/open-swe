"""Server-side Omnia DM transport."""

import asyncio
import hashlib
import hmac
import json
import os
from typing import Any

import httpx

from .http import DEFAULT_HTTP_TIMEOUT


def verify_omnia_signature(body: bytes, signature: str | None) -> bool:
    secret = os.environ.get("OMNIA_WEBHOOK_SECRET", "")
    if not secret or not signature:
        return False
    supplied = signature.removeprefix("sha256=")
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(supplied, expected)


async def post_omnia_dm_event(payload: dict[str, Any]) -> tuple[bool, str | None]:
    url = os.environ.get("OMNIA_CALLBACK_URL", "").strip()
    secret = os.environ.get("OMNIA_CALLBACK_SECRET", "")
    if not url or not secret:
        return False, "Omnia callback is not configured"
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    error = "Omnia callback failed"
    for attempt, delay in enumerate((0.5, 1.0, 2.0, 4.0, 0.0), start=1):
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                response = await client.post(
                    url,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Omnia-Signature": f"sha256={signature}",
                    },
                )
            if response.is_success:
                return True, None
            error = f"Omnia callback returned HTTP {response.status_code}"
            # Contract rejections contain the exact repair instructions. Hiding
            # them behind the status code strands a recoverable review forever.
            try:
                detail = response.json()
            except ValueError:
                detail = None
            if isinstance(detail, dict) and isinstance(detail.get("error"), str):
                error += f": {detail['error'][:2000]}"
            if response.status_code < 500 and response.status_code != 429:
                return False, error
        except httpx.HTTPError as exc:
            error = f"Omnia callback failed: {exc.__class__.__name__}"
        if attempt < 5:
            await asyncio.sleep(delay)
    return False, f"{error} after 5 attempts"


async def post_omnia_agent_action(payload: dict[str, Any]) -> dict[str, Any]:
    url = os.environ.get("OMNIA_TOOL_URL", "").strip()
    secret = os.environ.get("OMNIA_TOOL_SECRET", "")
    if not url or not secret:
        return {"success": False, "error": "Omnia agent tools are not configured"}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    error: dict[str, Any] = {"success": False, "error": "Omnia tool transport failed"}
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                response = await client.post(
                    url,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Omnia-Signature": f"sha256={signature}",
                    },
                )
            try:
                result = response.json()
            except ValueError:
                result = {"error": f"Omnia tool returned HTTP {response.status_code}"}
            if not isinstance(result, dict):
                result = {"result": result}
            if response.is_success:
                return {**result, "success": True}
            error = {**result, "success": False}
            if response.status_code not in {408, 409, 429} and response.status_code < 500:
                return error
        except httpx.HTTPError as exc:
            error = {"success": False, "error": f"Omnia tool failed: {exc.__class__.__name__}"}
        if attempt < 4:
            # Same serialized body and idempotency key: a lost response must not
            # create a second task or spend the approval on a second release.
            await asyncio.sleep(0.5 * 2**attempt)
    return error
