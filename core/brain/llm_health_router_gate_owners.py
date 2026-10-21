"""Whose turn a generation-gate lease holds, read from how it was admitted and what it is for.

Lifted whole out of `llm_health_router`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _generation_owner_is_user_foreground(owner: str) -> bool:
    owner = str(owner or "").strip().lower()
    if not owner:
        return False
    # Substring, deliberately: `owner` is the constructed `origin:purpose`
    # key, not a sentence — `desktop:response_generation_user`,
    # `voice_loop:reply`. The words in it run into their neighbours.
    return any(
        marker in owner
        for marker in (
            "user:",
            "desktop",
            "voice",
            "foreground",
            "response_generation_user",
        )
    )


def _lease_is_user_foreground(lease_id: int, owner: str) -> bool:
    """Whether a lease holds someone's reply: as it was admitted, or as its owner reads.

    Advisory work in a turn is admitted as foreground and is still not the
    reply. LIVE 27 Sep, the reports ground: the subtext pass held the gate
    after its phase had timed out, and the rating question's reply waited
    75s behind it and was refused as saturated on 11 of 96 arms.
    """
    from .llm_health_router import (
        _ADVISORY_PURPOSES,
        _GENERATION_GATE_FOREGROUND_LEASES,
        _GENERATION_GATE_STATE_LOCK,
        _generation_owner_is_user_foreground,
    )

    if str(owner or "").rsplit(":", 1)[-1].strip().lower() in _ADVISORY_PURPOSES:
        return False
    with _GENERATION_GATE_STATE_LOCK:
        admitted = lease_id in _GENERATION_GATE_FOREGROUND_LEASES
    return admitted or _generation_owner_is_user_foreground(owner)


def asks_for_a_reply(kwargs: Any, *, said_non_chat: bool = False) -> bool:
    """Whether a non-chat call is her reply to somebody, which the gate has to see as one.

    `think` marks what it carries as non-chat inference, which keeps it off the
    tool hand-off (that path calls `think` again), and the kernel's reply phase
    sends her reply to a person through it. The router turned non-chat into
    `internal_inference` for the gate, so the gate graded the reply as her own
    deliberation: no reply checks, and her cortex never named as the one that
    answered, so every arm of the reports ground read as one her cortex did not
    serve (21 October). A caller that says internal itself is taken at its word.
    """
    from .llm_health_router import HealthAwareLLMRouter

    held = kwargs if isinstance(kwargs, dict) else {}
    return (
        not said_non_chat
        and not held.get("internal_inference")
        and str(held.get("purpose") or "").lower() == "reply"
        and HealthAwareLLMRouter._is_user_facing_origin(str(held.get("origin") or ""))
    )

