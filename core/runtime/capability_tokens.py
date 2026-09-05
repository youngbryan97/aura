"""Universal capability tokens.

Audit constraint: every action checks a token with capability, scope,
expiry, trace_id, receipt_id, issuer, revoked_at. Tokens cannot be
re-used across turns once consumed.
"""
from __future__ import annotations


import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class TokenStatus(str, Enum):
    ISSUED = "issued"
    USED = "used"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class CapabilityToken:
    token_id: str
    capability: str
    scope: str
    issuer: str
    receipt_id: Optional[str]
    trace_id: Optional[str]
    issued_at: float
    expires_at: float
    status: TokenStatus = TokenStatus.ISSUED
    used_at: Optional[float] = None
    revoked_at: Optional[float] = None


class CapabilityTokenStore:
    def __init__(self):
        self._tokens: Dict[str, CapabilityToken] = {}
        self._lock = threading.RLock()

    def issue(
        self,
        *,
        capability: str,
        scope: str,
        issuer: str = "UnifiedWill",
        ttl_s: float = 3600.0,
        receipt_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> CapabilityToken:
        token = CapabilityToken(
            token_id=f"cap-{uuid.uuid4()}",
            capability=capability,
            scope=scope,
            issuer=issuer,
            receipt_id=receipt_id,
            trace_id=trace_id,
            issued_at=time.time(),
            expires_at=time.time() + ttl_s,
        )
        with self._lock:
            self._tokens[token.token_id] = token
        return token

    def consume(self, token_id: str) -> bool:
        with self._lock:
            tok = self._tokens.get(token_id)
            if tok is None or tok.status != TokenStatus.ISSUED:
                return False
            if time.time() >= tok.expires_at:
                tok.status = TokenStatus.EXPIRED
                return False
            tok.status = TokenStatus.USED
            tok.used_at = time.time()
            return True

    def revoke(self, token_id: str) -> None:
        with self._lock:
            tok = self._tokens.get(token_id)
            if tok and tok.status == TokenStatus.ISSUED:
                tok.status = TokenStatus.REVOKED
                tok.revoked_at = time.time()

    def get(self, token_id: str) -> Optional[CapabilityToken]:
        with self._lock:
            return self._tokens.get(token_id)


_global: Optional[CapabilityTokenStore] = None


def get_capability_token_store() -> CapabilityTokenStore:
    global _global
    if _global is None:
        _global = CapabilityTokenStore()
    return _global


def reset_capability_token_store() -> None:
    global _global
    _global = None
