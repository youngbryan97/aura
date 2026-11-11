"""Whose turn a generation-gate lease holds, read from how it was admitted and what it is for.

Lifted whole out of `llm_health_router`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from core.runtime.lockdep import checked_lock

#: How many people's turns are waiting for her model's generation lane now.
_PEOPLE_WAITING = 0
_PEOPLE_WAITING_LOCK = checked_lock("llm_router.people_waiting")


@contextmanager
def _a_person_waits() -> Iterator[None]:
    """Counted while a person's turn waits at the generation gate.

    Background work that holds her model across several generations reads the
    count between them and hands the model over (core/agency/her_goal_at_work.py).
    LIVE 11 November 2026: Bryan's three voice turns each ended with nothing
    generated while her goal pass held the lease for its whole pass.
    """
    global _PEOPLE_WAITING
    with _PEOPLE_WAITING_LOCK:
        _PEOPLE_WAITING += 1
    try:
        yield
    finally:
        with _PEOPLE_WAITING_LOCK:
            _PEOPLE_WAITING -= 1


def a_person_is_waiting_for_her_model() -> bool:
    """Whether a person's turn is waiting at the generation gate right now."""
    with _PEOPLE_WAITING_LOCK:
        return _PEOPLE_WAITING > 0


def _generation_gate_busy_result(owner: str) -> dict[str, Any]:
    from .llm_health_router import _GATE_SATURATION_RESULT

    result = dict(_GATE_SATURATION_RESULT)
    result["endpoint"] = "generation_gate_busy_foreground"
    result["error"] = (
        "local generation lane is busy with an active foreground user generation; "
        f"refusing to force-release owner={str(owner or 'unknown')[:120]}"
    )
    return result


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

