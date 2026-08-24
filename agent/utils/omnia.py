"""Server-side Omnia DM transport."""

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
        return False, f"Omnia callback returned HTTP {response.status_code}"
    except httpx.HTTPError as exc:
        return False, f"Omnia callback failed: {exc.__class__.__name__}"


async def post_omnia_agent_action(payload: dict[str, Any]) -> dict[str, Any]:
    url = os.environ.get("OMNIA_TOOL_URL", "").strip()
    secret = os.environ.get("OMNIA_TOOL_SECRET", "")
    if not url or not secret:
        return {"success": False, "error": "Omnia agent tools are not configured"}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
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
        if not response.is_success:
            return {"success": False, **result}
        return {"success": True, **result}
    except httpx.HTTPError as exc:
        return {"success": False, "error": f"Omnia tool failed: {exc.__class__.__name__}"}
