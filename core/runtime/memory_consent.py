"""Memory consent / privacy controls.

Audit-driven mode set: remember_always / ask_before_remembering /
session_only / private_mode / forget. Explicit user commands like
"forget this", "remember this part", "private mode" are honored at
write time.
"""
from __future__ import annotations


import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MemoryConsentMode(str, Enum):
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
        self._session_only_records: List[StoredRecordRef] = []
        self._lock = threading.RLock()

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

    def session_only_records(self) -> List[StoredRecordRef]:
        with self._lock:
            return list(self._session_only_records)

    def clear_session_records(self) -> List[StoredRecordRef]:
        with self._lock:
            cleared = list(self._session_only_records)
            self._session_only_records.clear()
            return cleared


_global: Optional[MemoryConsentPolicy] = None


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
}


def parse_consent_command(text: str) -> Optional[MemoryConsentMode]:
    lower = text.lower().strip()
    for key, mode in CONSENT_COMMANDS.items():
        if key in lower:
            return mode
    return None


def is_forget_command(text: str) -> bool:
    lower = text.lower().strip()
    return any(
        cmd in lower
        for cmd in (
            "forget this",
            "delete this memory",
            "erase that",
            "forget the session",
            "delete the movie session",
        )
    )
