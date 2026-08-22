"""Whether live perception is needed RIGHT NOW, and who says so.

The acting half of Aura and the seeing half had no way to talk to each other.
``constitutive_compute_budget`` throttles continuous loops to ``foreground_hz``
whenever a user-facing generation is running, which is correct for background
curiosity and exactly backwards for perception the generation depends on:
``continuous_vision`` passes ``foreground_hz=0.1``, so the instant she began
working on a request her sight dropped to one frame every ten seconds.

Nothing failed and nothing was logged. Any task built on look-act-look — drag
something and watch it move, wait for a progress bar, verify a click landed,
react to a board that changes — is simply unreachable at that cadence, and the
reason is invisible from inside the task.

This module is the missing edge. A piece of code that is ABOUT to act on the
world declares that it needs to see while it does so; the perception loops read
that declaration and keep their eyes open. Neither side has to know the other
exists, which is what keeps this general: nothing here is aware of screens,
games, browsers or any particular task.

Deliberately refcounted and expiring:

* refcounted, because two tasks can want to see at once and the first to
  finish must not blind the second;
* expiring, because a task that crashes between raising demand and releasing it
  would otherwise pin perception at full rate forever. Demand is a claim about
  the present, so it has to decay on its own.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from core.runtime.lockdep import checked_lock

#: How long a single unreleased claim stays believable. Long enough for a slow
#: step to finish, short enough that a crashed task cannot hold the cameras on
#: indefinitely. Every renewal pushes it out again, so a live task is never cut
#: off mid-work by this bound.
CLAIM_TTL_S = 90.0


@dataclass(frozen=True, slots=True)
class PerceptionClaim:
    """One live reason perception is wanted."""

    reason: str
    at: float
    expires_at: float


_lock = checked_lock("core.runtime.perception_demand")
_claims: dict[int, PerceptionClaim] = {}
_next_id = 0


def _prune_locked(now: float) -> None:
    expired = [key for key, claim in _claims.items() if claim.expires_at <= now]
    for key in expired:
        _claims.pop(key, None)


def perception_is_demanded() -> bool:
    """True while any live claim wants perception at task cadence."""
    now = time.time()
    with _lock:
        _prune_locked(now)
        return bool(_claims)


def active_perception_reasons() -> tuple[str, ...]:
    """Why perception is being held open, for health surfaces and receipts.

    A cadence that changes for reasons nobody can name is the kind of thing
    this codebase keeps having to rediscover from behaviour.
    """
    now = time.time()
    with _lock:
        _prune_locked(now)
        return tuple(claim.reason for claim in _claims.values())


def claim_perception(reason: str, *, ttl_s: float = CLAIM_TTL_S) -> int:
    """Register a need to see. Returns a token for release_perception."""
    global _next_id
    now = time.time()
    text = " ".join(str(reason or "unspecified").split())[:120]
    with _lock:
        _prune_locked(now)
        _next_id += 1
        token = _next_id
        _claims[token] = PerceptionClaim(
            reason=text, at=now, expires_at=now + max(1.0, float(ttl_s))
        )
        return token


def renew_perception(token: int, *, ttl_s: float = CLAIM_TTL_S) -> bool:
    """Push a long-running claim's expiry out. False if it already lapsed."""
    now = time.time()
    with _lock:
        # Prune FIRST. Without this a claim that had already lapsed could be
        # renewed back to life, which defeats the expiry that exists so a task
        # dying between claim and release cannot hold perception open forever.
        _prune_locked(now)
        claim = _claims.get(token)
        if claim is None:
            return False
        _claims[token] = PerceptionClaim(
            reason=claim.reason, at=claim.at, expires_at=now + max(1.0, float(ttl_s))
        )
        return True


def release_perception(token: int) -> None:
    """Drop one claim. Safe to call twice, and safe after expiry."""
    with _lock:
        _claims.pop(token, None)


@contextmanager
def perception_demand(reason: str, *, ttl_s: float = CLAIM_TTL_S) -> Iterator[int]:
    """Hold perception open for the duration of an act-and-watch block."""
    token = claim_perception(reason, ttl_s=ttl_s)
    try:
        yield token
    finally:
        release_perception(token)


def reset_perception_demand() -> None:
    """Drop every claim. For tests and for a hard runtime reset."""
    with _lock:
        _claims.clear()


__all__ = [
    "CLAIM_TTL_S",
    "PerceptionClaim",
    "active_perception_reasons",
    "claim_perception",
    "perception_demand",
    "perception_is_demanded",
    "release_perception",
    "renew_perception",
    "reset_perception_demand",
]
