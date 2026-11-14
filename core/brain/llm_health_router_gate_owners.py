"""Whose turn a generation-gate lease holds, read from how it was admitted and what it is for.

Lifted whole out of `llm_health_router`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from core.runtime.lockdep import checked_lock


def _gate_budget_env_s(name: str, default: float) -> float:
    """Parse a gate timing budget safely at import time.

    A malformed value must not prevent the router module from importing,
    and NaN/inf/negative budgets must not reach semaphore timeouts where
    they disable or invert the wait semantics.
    """
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value < 0.0:
        return float(default)
    return value


logger = logging.getLogger("Aura.LLMRouter.Gate")

#: The generation-gate lease the request running in this context holds, while it holds it.
_HELD_LEASE: ContextVar[int | None] = ContextVar("aura_generation_gate_lease", default=None)


@contextmanager
def holding_the_gate(lease_id: int) -> Iterator[None]:
    token = _HELD_LEASE.set(int(lease_id))
    try:
        yield
    finally:
        _HELD_LEASE.reset(token)


def a_cpu_client_gives_back_the_gate(client: Any) -> bool:
    """A background generation that reaches a model on the CPU gives the generation gate back before it runs.

    The gate keeps one generation at a time on her GPU, and a CPU model never touches it. LIVE 11 November 2026, her
    narrative fell through to her CPU model holding the gate, prefilled 716 tokens there for 80 s without reaching a
    cancel check, and her voice reply waited the whole time for a GPU nobody was using. A person's own turn keeps
    its lease wherever it runs.
    """
    lease = _HELD_LEASE.get()
    settings = getattr(client, "kwargs", None)
    device = str((settings.get("device") if isinstance(settings, dict) else None) or getattr(client, "device", "") or "")
    if lease is None or device.lower() != "cpu":
        return False
    from .llm_health_router import (
        _GENERATION_GATE_FOREGROUND_LEASES,
        _GENERATION_GATE_STATE_LOCK,
        _release_generation_gate_after_call,
    )

    with _GENERATION_GATE_STATE_LOCK:
        if lease in _GENERATION_GATE_FOREGROUND_LEASES:
            return False
    _release_generation_gate_after_call(lease)
    _HELD_LEASE.set(None)
    logger.info("🔓 a background generation on a CPU model gave back her GPU's generation gate (lease %s)", lease)
    return True


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


_LIVE_TURNS: set[str] = set()


def a_live_turn(session: str, live: bool) -> None:
    """A person and she are in a turn of a live conversation: voice mode, from their first word until her last has
    played. Her background generation stands aside until it ends, and what of it holds her model now is asked to stop.

    Her voice is made on the same GPU as her cortex's work. LIVE 11 November 2026: one patch of her voice (160 ms of
    audio) took 122 ms to make with the GPU quiet and 415 ms beside her goal pass reading a 32,383-character prompt,
    so every piece of a reply after the first arrived late
    (~/.aura/voice-lab/kiwi-reference/clean-2026-11-11/patch_profile.py).
    """
    key = str(session or "voice")
    with _PEOPLE_WAITING_LOCK:
        began = live and key not in _LIVE_TURNS
        if live:
            _LIVE_TURNS.add(key)
        else:
            _LIVE_TURNS.discard(key)
    if began:
        _background_stands_aside(f"a live turn began: {key[:40]}")


def a_live_turn_is_under_way() -> bool:
    with _PEOPLE_WAITING_LOCK:
        return bool(_LIVE_TURNS)


_VOICE_SESSIONS: set[str] = set()


def a_voice_session(session: str, open_: bool) -> None:
    """A person has her voice mode open: between turns too, while her cortex and her voice's models are held for it.

    A background model loaded in a gap is memory her live conversation then cannot have. LIVE 13 November 2026, her
    memory watchdog reached its hard ceiling twice in one voice session (managed 53,177 MB, then 53,574 MB, with 10 GB
    of swap) and the only model worker it could stop was her cortex: Bryan's first question waited 41 s while it
    reloaded, and the second kill came in the middle of a reply. Her brainstem had begun loading during the session.
    """
    key = str(session or "voice")
    with _PEOPLE_WAITING_LOCK:
        if open_:
            _VOICE_SESSIONS.add(key)
        else:
            _VOICE_SESSIONS.discard(key)


def a_voice_session_is_open() -> bool:
    with _PEOPLE_WAITING_LOCK:
        return bool(_VOICE_SESSIONS)


def _background_stands_aside(reason: str) -> None:
    """Ask what of her background work holds her model to stop; a person's own reply is never asked."""
    from .llm_health_router import _oldest_generation_gate_lease

    holder = _oldest_generation_gate_lease()
    if holder is None or _lease_is_user_foreground(holder[0], holder[2]):
        return
    from core.brain.llm.mlx_client import soft_cancel_active_generations

    soft_cancel_active_generations(reason=reason)


def a_person_is_waiting_for_her_model() -> bool:
    """Whether a person's turn is waiting at the generation gate right now."""
    with _PEOPLE_WAITING_LOCK:
        return _PEOPLE_WAITING > 0 or bool(_LIVE_TURNS)


def _generation_gate_busy_result(owner: str) -> dict[str, Any]:
    from .llm_health_router import _GATE_SATURATION_RESULT

    result = dict(_GATE_SATURATION_RESULT)
    result["endpoint"] = "generation_gate_busy_foreground"
    result["error"] = (
        "local generation lane is busy with an active foreground user generation; "
        f"refusing to force-release owner={str(owner or 'unknown')[:120]}"
    )
    return result


def _background_generation_gate_deferred_result(owner: str) -> dict[str, Any]:
    from .llm_health_router import _GATE_SATURATION_RESULT

    result = dict(_GATE_SATURATION_RESULT)
    result.update(
        {
            "endpoint": "generation_gate_background_deferred",
            "deferred": True,
            "retryable": True,
            "error": (
                "background generation deferred while the local generation lane "
                f"is owned by {str(owner or 'unknown')[:120]}"
            ),
        }
    )
    return result


def _declared_background_deferral_reason(result: Mapping[str, Any]) -> str:
    """Extract only explicit admission deferrals from a router result."""
    error = str(result.get("error", "") or "").strip()
    normalized = error.lower()
    if normalized.startswith("background_deferred:"):
        return error.split(":", 1)[1].strip() or "background_deferred"
    if bool(result.get("deferred", False)):
        return error or str(result.get("endpoint", "") or "background_deferred")
    if normalized in {
        "foreground_busy",
        "foreground_quiet_window",
        "desktop_background_local_disabled",
    }:
        return error
    if normalized.startswith("desktop_background_headroom:"):
        return error
    return ""


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

