"""interface/auth.py
──────────────────
Extracted from server.py — shared authentication, authorization,
rate-limiting, and session management utilities used across route files.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from http.cookies import SimpleCookie
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request

from core.config import config

logger = logging.getLogger("Aura.Server.Auth")


# ── Constants ─────────────────────────────────────────────────

TRUSTED_IPS = {"127.0.0.1", "::1"}

CHEAT_CODE_COOKIE_NAME = "aura_owner_session"
CHEAT_CODE_COOKIE_TTL_SECS = 60 * 60 * 24 * 30


# ── Internal-only guard ──────────────────────────────────────

def _require_internal(request: Request) -> None:
    """Block non-localhost requests when AURA_INTERNAL_ONLY=1."""
    if not config.security.internal_only_mode:
        return
    host = request.client.host if request.client else "unknown"
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="External access denied")


# ── Token verification ───────────────────────────────────────

def _verify_token(x_api_token: Optional[str] = Header(default=None)) -> None:
    """Bearer-token check. Ensures fail-closed unless running in strict internal_only_mode."""
    expected = config.api_token
    internal_only = getattr(config.security, "internal_only_mode", False)

    if not expected:
        # Only allow missing token if we are strictly bound to localhost
        if internal_only:
            if not getattr(_verify_token, '_warned', False):
                logger.warning("AURA_API_TOKEN not set but running in internal_only_mode.")
                _verify_token._warned = True
            return

        logger.error("AURA_API_TOKEN not set and service is not internal-only. Blocking.")
        raise HTTPException(status_code=503, detail="Authentication not configured")

    if not x_api_token or not hmac.compare_digest(x_api_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Cookie management ────────────────────────────────────────

_CHEAT_CODE_COOKIE_SECRET: Optional[bytes] = None


def _get_cheat_code_cookie_secret() -> bytes:
    global _CHEAT_CODE_COOKIE_SECRET
    if _CHEAT_CODE_COOKIE_SECRET is None:
        secret_value: Optional[str] = None
        try:
            from core.zenith_secrets import get_secret

            secret_value = get_secret("AURA_CHEAT_CODE_COOKIE_SECRET")
        except Exception:
            secret_value = None
        secret_value = secret_value or config.api_token or secrets.token_urlsafe(32)
        _CHEAT_CODE_COOKIE_SECRET = secret_value.encode("utf-8")
    return _CHEAT_CODE_COOKIE_SECRET


def _encode_owner_session_cookie() -> str:
    payload = {
        "scope": "sovereign_owner",
        "issued_at": int(time.time()),
        "exp": int(time.time()) + CHEAT_CODE_COOKIE_TTL_SECS,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(
        _get_cheat_code_cookie_secret(),
        encoded.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded}.{signature}"


def _decode_owner_session_cookie(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(
        _get_cheat_code_cookie_secret(),
        encoded.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception:
        return None
    if payload.get("scope") != "sovereign_owner":
        return None
    if int(payload.get("exp", 0) or 0) < int(time.time()):
        return None
    return payload


def _restore_owner_session_from_request(request: Optional[Request]) -> bool:
    if request is None:
        return False
    token = None
    cookies = getattr(request, "cookies", None)
    if cookies is not None:
        try:
            token = cookies.get(CHEAT_CODE_COOKIE_NAME)
        except Exception:
            token = None
    if not token:
        headers = getattr(request, "headers", None) or {}
        cookie_header = headers.get("cookie") or headers.get("Cookie")
        if cookie_header:
            parsed = SimpleCookie()
            try:
                parsed.load(cookie_header)
            except Exception:
                parsed = SimpleCookie()
            morsel = parsed.get(CHEAT_CODE_COOKIE_NAME)
            if morsel is not None:
                token = morsel.value
    payload = _decode_owner_session_cookie(token)
    if not payload:
        return False
    try:
        from core.security.trust_engine import get_trust_engine
        from core.security.user_recognizer import get_user_recognizer

        get_user_recognizer().override_session_owner(reason="owner_session_cookie")
        get_trust_engine().establish_sovereign_session(
            reason="owner_session_cookie",
            announce=False,
        )
        return True
    except Exception as exc:
        logger.debug("Owner session cookie restore failed: %s", exc)
        return False


def _activate_cheat_code_for_request(code: Optional[str], *, silent: bool, source: str) -> Optional[Dict[str, Any]]:
    if not code:
        return None
    try:
        from core.security.cheat_codes import activate_cheat_code

        return activate_cheat_code(code, silent=silent, source=source)
    except Exception as exc:
        logger.debug("Cheat code activation failed: %s", exc)
        return {
            "ok": False,
            "status": "error",
            "message": "Cheat code activation failed.",
        }


# ── Rate Limiter ──────────────────────────────────────────────

class _RateLimiter:
    """Token-bucket rate limiter per client IP with automatic cleanup."""
    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self._max = max_requests
        self._window = window_seconds
        self._clients: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def check(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            # Periodic cleanup: evict stale IPs every 5 minutes
            if now - self._last_cleanup > 300:
                stale = [ip for ip, hits in self._clients.items() if not hits or now - hits[-1] > self._window]
                for ip in stale:
                    del self._clients[ip]
                self._last_cleanup = now

            hits = self._clients.get(client_ip, [])
            hits = [t for t in hits if now - t < self._window]
            if len(hits) >= self._max:
                return False
            hits.append(now)
            self._clients[client_ip] = hits
            return True

_rate_limiter = _RateLimiter(max_requests=30, window_seconds=60.0)


def _check_rate_limit(request: Request) -> None:
    """H-02: Rate limit check with Trusted IP bypass."""
    # Prioritize real client IP over the proxy's IP to prevent distributed DOS
    forwarded = request.headers.get("X-Forwarded-For")
    real_ip = request.headers.get("X-Real-IP")

    if forwarded:
        client_ip = forwarded.split(',')[0].strip()
    elif real_ip:
        client_ip = real_ip
    else:
        client_ip = request.client.host if request.client else "unknown"

    # Perplexity Audit Fix: Bypass rate limit for local/trusted telemetry
    if client_ip in TRUSTED_IPS:
        return

    if not _rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")
