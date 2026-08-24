import hashlib
import hmac

from agent.utils.omnia import verify_omnia_signature


def test_verify_omnia_signature_is_fail_closed(monkeypatch) -> None:
    body = b'{"hello":"world"}'
    monkeypatch.delenv("OMNIA_WEBHOOK_SECRET", raising=False)
    assert verify_omnia_signature(body, "sha256=anything") is False

    secret = "secret"
    monkeypatch.setenv("OMNIA_WEBHOOK_SECRET", secret)
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_omnia_signature(body, f"sha256={signature}") is True
    assert verify_omnia_signature(body, "sha256=bad") is False
