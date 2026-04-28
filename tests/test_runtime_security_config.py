from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class _Request:
    def __init__(self, path="/api/chat", host="203.0.113.10", headers=None):
        self.url = SimpleNamespace(path=path)
        self.client = SimpleNamespace(host=host)
        self.headers = headers or {}


def test_runtime_security_fails_closed_when_token_disappears(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", None, raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(_Request())

    assert exc.value.status_code == 503


def test_runtime_security_keeps_health_probe_available_without_token(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", None, raising=False)

    auth.validate_runtime_security_request(_Request(path="/api/health"))


def test_runtime_security_rechecks_internal_only_per_request(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", True, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(_Request(host="198.51.100.2"))

    assert exc.value.status_code == 403


def test_runtime_security_accepts_valid_bearer_token(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    auth.validate_runtime_security_request(
        _Request(headers={"Authorization": "Bearer secret"})
    )


def test_runtime_security_rejects_invalid_external_token(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(
            _Request(headers={"Authorization": "Bearer wrong"})
        )

    assert exc.value.status_code == 401
