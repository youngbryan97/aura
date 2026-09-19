"""Memory consent / privacy controls.

Audit-driven mode set: remember_always / ask_before_remembering /
session_only / private_mode / forget. Explicit user commands like
"forget this", "remember this part", "private mode" are honored at
write time.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock


class MemoryConsentMode(StrEnum):
    REMEMBER_ALWAYS = "remember_always"
    ASK_BEFORE_REMEMBERING = "ask_before_remembering"
    SESSION_ONLY = "session_only"
    PRIVATE_MODE = "private_mode"


@dataclass
class StoredRecordRef:
    record_id: str
    family: str
    stored_at: float


class MemoryConsentPolicy:
    def __init__(self, *, default_mode: MemoryConsentMode = MemoryConsentMode.ASK_BEFORE_REMEMBERING):
        self.mode = default_mode
        self._session_only_records: list[StoredRecordRef] = []
        self._lock = checked_lock("runtime.memory_consent.requests", reentrant=True)

    def set_mode(self, mode: MemoryConsentMode) -> None:
        self.mode = mode

    def may_persist_long_term(self) -> bool:
        return self.mode == MemoryConsentMode.REMEMBER_ALWAYS

    def needs_user_approval(self) -> bool:
        return self.mode == MemoryConsentMode.ASK_BEFORE_REMEMBERING

    def is_session_only(self) -> bool:
        return self.mode == MemoryConsentMode.SESSION_ONLY

    def is_private(self) -> bool:
        return self.mode == MemoryConsentMode.PRIVATE_MODE

    def register_session_record(self, ref: StoredRecordRef) -> None:
        if self.mode == MemoryConsentMode.SESSION_ONLY:
            with self._lock:
                self._session_only_records.append(ref)

    def session_only_records(self) -> list[StoredRecordRef]:
        with self._lock:
            return list(self._session_only_records)

    def clear_session_records(self) -> list[StoredRecordRef]:
        with self._lock:
            cleared = list(self._session_only_records)
            self._session_only_records.clear()
            return cleared


_global: MemoryConsentPolicy | None = None


def get_memory_consent_policy() -> MemoryConsentPolicy:
    global _global
    if _global is None:
        _global = MemoryConsentPolicy()
    return _global


def reset_memory_consent_policy() -> None:
    global _global
    _global = None


# --- user command parser ---------------------------------------------------


CONSENT_COMMANDS = {
    "remember always": MemoryConsentMode.REMEMBER_ALWAYS,
    "always remember": MemoryConsentMode.REMEMBER_ALWAYS,
    "ask before remembering": MemoryConsentMode.ASK_BEFORE_REMEMBERING,
    "session only": MemoryConsentMode.SESSION_ONLY,
    "private mode": MemoryConsentMode.PRIVATE_MODE,
    "go private": MemoryConsentMode.PRIVATE_MODE,
    "go private mode": MemoryConsentMode.PRIVATE_MODE,
}

_COMMAND_PREFIX = re.compile(
    r"^(?:aura[,:]?\s*)?(?:(?:can|could|would) you\s+)?(?:please\s+)?",
    re.IGNORECASE,
)
_COMMAND_SUFFIX = re.compile(r"(?:\s+please)?(?:\s+now)?[.!?]*$", re.IGNORECASE)
_DELETE_ALL_COMMANDS = frozenset(
    {
        "forget everything about me",
        "delete all relational memory",
        "erase all relationship memory",
    }
)


class _UnhonouredDeletionRequests:
    """How often somebody asked to be forgotten and was not answered.

    A log line is read by whoever is looking. A counter is read by the
    integrity surface, which is where a privacy control that is quietly
    doing nothing becomes visible without anyone having to look.
    """

    def __init__(self) -> None:
        self._lock = checked_lock("runtime.memory_consent.unhonoured")
        self.count = 0
        self.principals: set[str] = set()
        self.last_at: float = 0.0

    def record(self, principal: str) -> None:
        with self._lock:
            self.count += 1
            if principal:
                self.principals.add(principal[:160])
            self.last_at = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "unhonoured_deletion_requests": self.count,
                "people_affected": len(self.principals),
                "last_at": self.last_at or None,
            }

    def reset(self) -> None:
        with self._lock:
            self.count = 0
            self.principals.clear()
            self.last_at = 0.0


_UNHONOURED = _UnhonouredDeletionRequests()


def memory_consent_report() -> dict[str, Any]:
    """For the integrity surface."""

    return _UNHONOURED.status()


def reset_unhonoured_deletion_requests_for_test() -> None:
    _UNHONOURED.reset()


def _normalized_command_body(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    normalized = _COMMAND_PREFIX.sub("", normalized, count=1)
    normalized = _COMMAND_SUFFIX.sub("", normalized, count=1)
    return normalized.strip()


def parse_consent_command(text: str) -> MemoryConsentMode | None:
    body = _normalized_command_body(text)
    for command, mode in CONSENT_COMMANDS.items():
        if body == command:
            return mode
    return None


def is_delete_all_relational_memory_command(text: str) -> bool:
    return _normalized_command_body(text) in _DELETE_ALL_COMMANDS


def apply_relational_memory_command(
    authority: Any,
    agent_id: str,
    message: str,
    *,
    receipt_id: str = "",
) -> dict[str, Any] | None:
    """Apply one explicit exact-agent relational-memory control command."""
    exact_id = " ".join(str(agent_id or "").strip().split())[:160]
    if not exact_id:
        raise ValueError("relational memory control requires an exact agent_id")
    normalized = " ".join(str(message or "").strip().lower().split())
    mode = parse_consent_command(normalized)
    delete_all = is_delete_all_relational_memory_command(normalized)
    if mode is None and not delete_all:
        # Not a command — but if it MEANT erasure, saying nothing is the
        # defect. The exact-match set above is right to be exact, because
        # erasing what she knows about somebody is irreversible and "do not
        # forget everything about me" must never fire it. What was wrong is
        # that a near miss went out as an ordinary message: the person
        # asked, believed they had asked, and nothing happened and nothing
        # said so. Five of seven ordinary phrasings, measured 2026-09-18.
        if looks_like_a_deletion_request(normalized):
            _UNHONOURED.record(exact_id)
            record_degradation(
                "memory_consent.deletion_request",
                RuntimeError(
                    "a message asking to be forgotten matched no exact command "
                    f"and was not acted on: {normalized[:120]!r}"
                ),
                severity="warning",
                action=(
                    "left relational memory untouched; the person's request "
                    "was not in the command vocabulary"
                ),
            )
        return None
    evidence = hashlib.sha256(
        f"{exact_id}\n{normalized}".encode("utf-8", errors="replace")
    ).hexdigest()
    command_receipt_id = str(
        receipt_id or f"user-command-evidence-{evidence}"
    ).strip()[:200]
    if not command_receipt_id:
        raise ValueError("relational memory control requires command evidence")

    if delete_all:
        receipt = authority.delete_agent(
            exact_id,
            authorization_receipt_id=command_receipt_id,
        )
        return {
            "mode": "deleted",
            "receipt_id": receipt.receipt_id,
            "deleted_records": len(receipt.record_ids),
        }

    if mode is None:
        raise RuntimeError("relational memory command classification lost its mode")
    grant = None
    if mode == MemoryConsentMode.REMEMBER_ALWAYS:
        grant = authority.replace_consent(
            exact_id,
            kinds=authority.supported_kinds(),
            operations=["persist", "recall", "prompt"],
            receipt_id=command_receipt_id,
            source="explicit_user_command",
        )
    elif mode == MemoryConsentMode.SESSION_ONLY:
        grant = authority.replace_consent(
            exact_id,
            kinds=authority.supported_kinds(),
            operations=["recall", "prompt"],
            receipt_id=command_receipt_id,
            source="explicit_user_command",
        )
    else:
        authority.revoke_consent(
            exact_id,
            receipt_id=f"{command_receipt_id}:replace"[:200],
            delete_records=False,
        )
    return {
        "mode": mode.value,
        "grant_id": grant.grant_id if grant is not None else "",
        "persistence_requested": bool(
            grant is not None and "persist" in grant.operations
        ),
        "persistence_allowed": bool(
            grant is not None
            and "persist" in grant.operations
            and authority.persistence_available
        ),
        "persistence_available": bool(authority.persistence_available),
        "prompt_use_allowed": bool(
            grant is not None and "prompt" in grant.operations
        ),
    }


#: Erasure asked for in the imperative. Composed rather than enumerated:
#: a verb that means erase, and an object that means what you hold about
#: me. That covers phrasings nobody wrote down;
#: the exact-match command set below cannot, and the predicate it replaced
#: was five literal sentences called by nothing, one of them
#: "delete the movie session", a past test case left in a production check.
_ERASURE_VERBS = (
    "forget",
    "delete",
    "erase",
    "wipe",
    "remove",
    "scrub",
    "purge",
    "clear",
)
_MEMORY_OBJECTS = (
    "about me",
    "about myself",
    "you know about",
    "you have about",
    "you have on me",
    "you hold about",
    "have on me",
    "you remember about",
    "you stored",
    "you've stored",
    "my data",
    "my memory",
    "my memories",
    "my history",
    "my information",
    "my details",
    "relational memory",
    "everything about",
    "all of it",
)


def looks_like_a_deletion_request(text: str) -> bool:
    """Whether this reads as asking to be forgotten, however it is phrased.

    Deliberately NOT the trigger for the deletion itself. Erasing what she
    knows about somebody is irreversible, so the act stays behind the
    exact-match command set below, where "do not forget everything about
    me" cannot fire it.

    Loose on purpose, and safe because of what it does not do. "do not
    forget everything about me" reads as a deletion request here and costs
    a log line; it cannot cost a deletion, because deletion is decided
    elsewhere. A predicate whose only output is visibility can afford to be
    generous where one that erases cannot.

    This is the other half: a request that means erasure and does not match
    exactly must not be *silent*. Measured 2026-09-18, five of seven
    ordinary phrasings did nothing and said nothing — including "Aura,
    forget everything you know about me", whose prefix the normaliser
    strips and whose body is one word away from a command. The person
    asked, believed they had asked, and nothing happened.
    """

    body = _normalized_command_body(text)
    if not body:
        return False
    if not any(verb in body for verb in _ERASURE_VERBS):
        return False
    return any(obj in body for obj in _MEMORY_OBJECTS)
