"""core/agency/capability_token.py

Full capability-token lifecycle. Every token carries its origin, scope,
TTL, domain, requested action, approver, revocation status, parent receipt,
child execution receipt, and side-effect log.

Tokens are bound to:
  * the issuing thread / asyncio task and the tasks it starts (cross-thread
    use, and use from an unrelated task, is rejected)
  * the issuing process generation (post-shutdown use is rejected)
  * the originally requested domain (wrong-domain use is rejected)
  * a wall-clock TTL (expired use is rejected)
  * a single execution receipt (replay use is rejected)

The store is in-process; persistence is intentionally out of scope to
prevent cross-restart token reuse. ``revoke_all()`` is called on graceful
shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.CapabilityToken")


_PROCESS_GEN = f"{os.getpid()}-{int(time.time())}"


#: Tokens issued in this context, and so held by the task that issued them
#: and by every task it starts afterwards, which copy the context when they
#: are created. An unrelated task never has them.
#:
#: The binding was to the issuing task alone, and work a request starts runs
#: in tasks of its own: LIVE 2026-09-23, the chat turn issued the token for a
#: desktop action, the desktop task it started asked for computer_use, and
#: the check refused the request's own token as capability_token_cross_task
#: on every launch.
_ISSUED_HERE: ContextVar[frozenset[str]] = ContextVar(
    "aura_capability_tokens_issued_here", default=frozenset()
)


def _current_task_id() -> int | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        # not a failure: called off a loop, so there is no current task to
        # scope a token to, and the line below returns None for that too.
        return None
    return id(task) if task is not None else None


@dataclass
class CapabilityToken:
    token: str
    origin: str
    scope: str
    ttl_seconds: float
    domain: str
    requested_action: str
    approver: str
    parent_receipt: str
    issued_at: float = field(default_factory=lambda: time.time())
    process_gen: str = _PROCESS_GEN
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    task_id: int | None = field(default_factory=_current_task_id)
    revoked: bool = False
    revoked_reason: str | None = None
    consumed_at: float | None = None
    child_execution_receipt: str | None = None
    side_effects: list[str] = field(default_factory=list)

    def is_expired(self, now: float | None = None) -> bool:
        now = now or time.time()
        return (now - self.issued_at) > self.ttl_seconds

    def is_consumed(self) -> bool:
        return self.consumed_at is not None


class CapabilityTokenStore:
    """In-process token registry with replay/expiry/cross-thread checks."""

    def __init__(self) -> None:
        self._tokens: dict[str, CapabilityToken] = {}
        self._lock = threading.RLock()

    def issue(
        self,
        *,
        origin: str,
        scope: str,
        ttl_seconds: float,
        domain: str,
        requested_action: str,
        approver: str,
        parent_receipt: str,
    ) -> CapabilityToken:
        token_str = f"CT-{secrets.token_urlsafe(18)}"
        tok = CapabilityToken(
            token=token_str,
            origin=origin,
            scope=scope,
            ttl_seconds=float(ttl_seconds),
            domain=domain,
            requested_action=requested_action,
            approver=approver,
            parent_receipt=parent_receipt,
        )
        with self._lock:
            self._tokens[token_str] = tok
        _ISSUED_HERE.set(_ISSUED_HERE.get() | {token_str})
        return tok

    def validate(
        self,
        token_str: str,
        *,
        domain: str,
        action: str,
    ) -> CapabilityToken:
        """Return the token iff it passes ALL checks. Raises on failure."""
        with self._lock:
            tok = self._tokens.get(token_str)
            if tok is None:
                raise PermissionError("capability_token_unknown")
            if tok.revoked:
                raise PermissionError(f"capability_token_revoked:{tok.revoked_reason or '?'}")
            if tok.is_consumed():
                raise PermissionError("capability_token_replay")
            if tok.is_expired():
                raise PermissionError("capability_token_expired")
            if tok.process_gen != _PROCESS_GEN:
                raise PermissionError("capability_token_post_shutdown")
            if tok.thread_id != threading.get_ident():
                raise PermissionError("capability_token_cross_thread")
            current_task_id = _current_task_id()
            if (
                tok.task_id is not None
                and tok.task_id != current_task_id
                and token_str not in _ISSUED_HERE.get()
            ):
                raise PermissionError("capability_token_cross_task")
            if tok.domain != domain:
                raise PermissionError(f"capability_token_wrong_domain:{tok.domain}!={domain}")
            if tok.requested_action != action:
                raise PermissionError(f"capability_token_wrong_action:{tok.requested_action}!={action}")
            return tok

    def validate_and_consume(
        self,
        token_str: str,
        *,
        domain: str,
        action: str,
        child_receipt: str,
        side_effects: list[str] | None = None,
    ) -> CapabilityToken:
        """Atomically validate and spend a single-use capability.

        A separate ``validate`` followed by ``consume`` leaves a race where
        two callers can both validate before either records consumption.
        """

        with self._lock:
            tok = self.validate(token_str, domain=domain, action=action)
            tok.consumed_at = time.time()
            tok.child_execution_receipt = child_receipt
            if side_effects:
                tok.side_effects.extend(side_effects)
            return tok

    def consume(self, token_str: str, *, child_receipt: str, side_effects: list[str] | None = None) -> CapabilityToken:
        with self._lock:
            tok = self._tokens.get(token_str)
            if tok is None:
                raise PermissionError("capability_token_unknown_on_consume")
            if tok.is_consumed():
                raise PermissionError("capability_token_replay_on_consume")
            tok.consumed_at = time.time()
            tok.child_execution_receipt = child_receipt
            if side_effects:
                tok.side_effects.extend(side_effects)
            return tok

    def revoke(self, token_str: str, *, reason: str) -> None:
        with self._lock:
            tok = self._tokens.get(token_str)
            if tok is not None:
                tok.revoked = True
                tok.revoked_reason = reason

    def revoke_all(self, *, reason: str = "shutdown") -> int:
        with self._lock:
            n = 0
            for tok in self._tokens.values():
                if not tok.revoked:
                    tok.revoked = True
                    tok.revoked_reason = reason
                    n += 1
            return n

    def get(self, token_str: str) -> CapabilityToken | None:
        with self._lock:
            return self._tokens.get(token_str)


_STORE: CapabilityTokenStore | None = None


def get_token_store() -> CapabilityTokenStore:
    global _STORE
    if _STORE is None:
        _STORE = CapabilityTokenStore()
    return _STORE


def reset_token_store() -> None:
    """Revoke and replace the process-local token generation (tests/restart)."""

    global _STORE
    if _STORE is not None:
        _STORE.revoke_all(reason="token_store_reset")
    _STORE = None
