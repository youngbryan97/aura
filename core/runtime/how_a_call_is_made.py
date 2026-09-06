"""One place that says what happens when a phase or a service call goes wrong.

Every phase and every service call in Aura decides its own timeout, its own
retries, whether it caches, and what happens on cancellation. Reading a
failure therefore means reading the call site, and two call sites doing the
same work five lines apart can behave differently under load with nothing
saying so.

LangGraph gives every node one policy object and one executor that applies it.
This is that, with the parts Aura already had wired in rather than
reimplemented:

* **retry** — classified by :mod:`core.runtime.what_must_never_be_retried`, so
  a governance refusal is not asked twice.
* **budget** — spent through :mod:`core.runtime.what_is_left_to_spend`, so a
  retrying call cannot outspend the turn that contains it.
* **cancellation** — declared through :mod:`core.runtime.how_a_task_should_end`,
  so the drain deadline for a call belongs to whoever owns the work.

What this module adds is the two remaining fields and the question they make
answerable. **Criticality** says what a failure costs: whether the turn
continues without this call's result, degraded, or not at all. **Idempotency**
says whether a retry after a timeout risks doing the work twice — and a
timeout is exactly where you cannot tell whether the first attempt landed.

:func:`calls_with_no_policy` names every declared call site running on the
default. A default that nobody chose is the thing worth counting.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from core.runtime.what_is_left_to_spend import ABudget
from core.runtime.what_must_never_be_retried import TryAgain, how_to_treat, why_not

logger = logging.getLogger("Aura.HowACallIsMade")

__all__ = [
    "HowBadIsLosingThis",
    "IsItSafeToRepeat",
    "APolicy",
    "THE_DEFAULT_POLICY",
    "AnOutcome",
    "declare_a_call",
    "the_policy_for",
    "make_the_call",
    "make_the_call_async",
    "calls_with_no_policy",
    "how_the_calls_are_made",
    "forget_everything",
]


class HowBadIsLosingThis(StrEnum):
    """What the turn does when this call does not come back."""

    #: The turn continues and nothing downstream notices.
    OPTIONAL = "optional"
    #: The turn continues with a worse answer, and says so.
    DEGRADES = "degrades"
    #: The turn cannot be completed. Fail rather than answer from nothing.
    REQUIRED = "required"


class IsItSafeToRepeat(StrEnum):
    """Whether a retry after a timeout can do the work twice."""

    #: Repeating changes nothing. A read, a pure computation.
    YES = "yes"
    #: Repeating is safe only because the callee dedupes on a key.
    WITH_A_KEY = "with a key"
    #: Repeating may do the work twice, and after a timeout you cannot tell
    #: whether the first attempt landed. Do not retry a timeout here.
    NO = "no"


@dataclass(frozen=True, slots=True)
class APolicy:
    """Everything one call site decided, in one object."""

    name: str
    timeout_seconds: float = 30.0
    attempts: int = 1
    backoff_seconds: float = 0.25
    criticality: HowBadIsLosingThis = HowBadIsLosingThis.DEGRADES
    idempotent: IsItSafeToRepeat = IsItSafeToRepeat.NO
    #: Whose drain deadline applies when the runtime is going down.
    owner: str = ""
    #: Seconds a result may be reused, 0 for never.
    cache_seconds: float = 0.0
    why: str = ""

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"{self.name}: a call has to be attempted at least once")
        if self.attempts > 1 and self.idempotent is IsItSafeToRepeat.NO:
            # Not an error: a retry is still correct for failures that are not
            # timeouts, because those tell you the attempt did not land.
            logger.debug(
                "%s retries a call that is not safe to repeat; timeouts will not "
                "be retried, other failures will",
                self.name,
            )

    def may_retry(self, failure: BaseException) -> bool:
        """Whether this policy retries this particular failure."""
        if how_to_treat(failure) is not TryAgain.AGAIN:
            return False
        timed_out = isinstance(failure, (asyncio.TimeoutError, TimeoutError))
        if timed_out and self.idempotent is IsItSafeToRepeat.NO:
            return False
        return True


THE_DEFAULT_POLICY = APolicy(
    name="(undeclared)",
    why="nobody declared this call site, so the conservative defaults apply",
)


@dataclass
class AnOutcome:
    """What actually happened, whatever the policy said should."""

    name: str
    value: Any = None
    failure: BaseException | None = None
    attempts_made: int = 0
    seconds: float = 0.0
    from_cache: bool = False
    refused_retry_because: str = ""
    fell_back: bool = False

    @property
    def ok(self) -> bool:
        return self.failure is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "failure": type(self.failure).__name__ if self.failure else "",
            "attempts_made": self.attempts_made,
            "seconds": round(self.seconds, 4),
            "from_cache": self.from_cache,
            "refused_retry_because": self.refused_retry_because,
            "fell_back": self.fell_back,
        }


_POLICIES: dict[str, APolicy] = {}
_SEEN: dict[str, int] = {}
_HISTORY: list[dict[str, Any]] = []
_CACHE: dict[str, tuple[float, Any]] = {}
_LOCK = threading.Lock()
_KEEP = 300


def declare_a_call(policy: APolicy) -> APolicy:
    """Register what this call site decided."""
    with _LOCK:
        _POLICIES[policy.name] = policy
    return policy


def the_policy_for(name: str) -> APolicy:
    with _LOCK:
        found = _POLICIES.get(name)
    return found if found is not None else replace(THE_DEFAULT_POLICY, name=name)


def calls_with_no_policy() -> tuple[str, ...]:
    """Call sites that ran on the default because nobody chose one."""
    with _LOCK:
        return tuple(sorted(set(_SEEN) - set(_POLICIES)))


def _cache_read(policy: APolicy) -> tuple[bool, Any]:
    if policy.cache_seconds <= 0:
        return False, None
    with _LOCK:
        found = _CACHE.get(policy.name)
    if found is None:
        return False, None
    at, value = found
    if time.monotonic() - at > policy.cache_seconds:
        return False, None
    return True, value


def _cache_write(policy: APolicy, value: Any) -> None:
    if policy.cache_seconds <= 0:
        return
    with _LOCK:
        _CACHE[policy.name] = (time.monotonic(), value)


def _record(outcome: AnOutcome) -> AnOutcome:
    with _LOCK:
        _SEEN[outcome.name] = _SEEN.get(outcome.name, 0) + 1
        _HISTORY.append(outcome.to_dict())
        del _HISTORY[:-_KEEP]
    return outcome


def make_the_call(
    name: str,
    do: Callable[[], Any],
    *,
    fallback: Callable[[], Any] | None = None,
    budget: ABudget | None = None,
) -> AnOutcome:
    """Run a synchronous call under its declared policy."""
    policy = the_policy_for(name)
    outcome = AnOutcome(name=name)
    started = time.monotonic()

    hit, cached = _cache_read(policy)
    if hit:
        outcome.value = cached
        outcome.from_cache = True
        outcome.seconds = time.monotonic() - started
        return _record(outcome)

    last: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        if budget is not None and not budget.spend(1.0, on=f"{name}#{attempt}"):
            outcome.refused_retry_because = "the budget was spent"
            break
        outcome.attempts_made = attempt
        try:
            outcome.value = do()
            outcome.failure = None
            _cache_write(policy, outcome.value)
            outcome.seconds = time.monotonic() - started
            return _record(outcome)
        except BaseException as exc:  # noqa: BLE001 - classified, then re-raised or not
            last = exc
            if not policy.may_retry(exc):
                outcome.refused_retry_because = why_not(exc) or (
                    "a timeout on a call that is not safe to repeat"
                )
                break
            if attempt < policy.attempts and policy.backoff_seconds > 0:
                time.sleep(policy.backoff_seconds * attempt)

    outcome.failure = last
    if fallback is not None and policy.criticality is not HowBadIsLosingThis.REQUIRED:
        try:
            outcome.value = fallback()
            outcome.failure = None
            outcome.fell_back = True
        except BaseException as exc:  # noqa: BLE001
            outcome.failure = exc
    outcome.seconds = time.monotonic() - started
    return _record(outcome)


async def make_the_call_async(
    name: str,
    do: Callable[[], Awaitable[Any]],
    *,
    fallback: Callable[[], Awaitable[Any]] | None = None,
    budget: ABudget | None = None,
) -> AnOutcome:
    """Run an awaitable call under its declared policy, timeout included."""
    policy = the_policy_for(name)
    outcome = AnOutcome(name=name)
    started = time.monotonic()

    hit, cached = _cache_read(policy)
    if hit:
        outcome.value = cached
        outcome.from_cache = True
        outcome.seconds = time.monotonic() - started
        return _record(outcome)

    last: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        if budget is not None and not budget.spend(1.0, on=f"{name}#{attempt}"):
            outcome.refused_retry_because = "the budget was spent"
            break
        outcome.attempts_made = attempt
        try:
            outcome.value = await asyncio.wait_for(do(), timeout=policy.timeout_seconds)
            outcome.failure = None
            _cache_write(policy, outcome.value)
            outcome.seconds = time.monotonic() - started
            return _record(outcome)
        except asyncio.CancelledError:
            # A cancellation is the caller's decision, never this loop's to retry.
            outcome.attempts_made = attempt
            outcome.seconds = time.monotonic() - started
            _record(outcome)
            raise
        except BaseException as exc:  # noqa: BLE001
            last = exc
            if not policy.may_retry(exc):
                outcome.refused_retry_because = why_not(exc) or (
                    "a timeout on a call that is not safe to repeat"
                )
                break
            if attempt < policy.attempts and policy.backoff_seconds > 0:
                await asyncio.sleep(policy.backoff_seconds * attempt)

    outcome.failure = last
    if fallback is not None and policy.criticality is not HowBadIsLosingThis.REQUIRED:
        try:
            outcome.value = await fallback()
            outcome.failure = None
            outcome.fell_back = True
        except BaseException as exc:  # noqa: BLE001
            outcome.failure = exc
    outcome.seconds = time.monotonic() - started
    return _record(outcome)


def how_the_calls_are_made() -> dict[str, Any]:
    """For the health report: who declared a policy, and how the calls went."""
    with _LOCK:
        declared = {
            name: {
                "timeout_seconds": p.timeout_seconds,
                "attempts": p.attempts,
                "criticality": str(p.criticality),
                "idempotent": str(p.idempotent),
                "owner": p.owner,
                "cache_seconds": p.cache_seconds,
                "why": p.why,
            }
            for name, p in sorted(_POLICIES.items())
        }
        seen = dict(sorted(_SEEN.items()))
        history = list(_HISTORY)
    return {
        "declared": len(declared),
        "call_sites_that_ran": len(seen),
        "calls_with_no_policy": sorted(set(seen) - set(declared)),
        "required": sorted(
            n for n, p in declared.items() if p["criticality"] == "required"
        ),
        "retried_but_not_safe_to_repeat": sorted(
            n
            for n, p in declared.items()
            if p["attempts"] > 1 and p["idempotent"] == "no"
        ),
        "calls_recorded": len(history),
        "retries_refused": sum(1 for h in history if h["refused_retry_because"]),
        "fell_back": sum(1 for h in history if h["fell_back"]),
        "served_from_cache": sum(1 for h in history if h["from_cache"]),
        "policies": declared,
    }


def forget_everything() -> None:
    with _LOCK:
        _POLICIES.clear()
        _SEEN.clear()
        _HISTORY.clear()
        _CACHE.clear()
    _declare_what_ships()


def _declare_what_ships() -> None:
    """The call sites whose behaviour under failure was read out of the code."""
    declare_a_call(
        APolicy(
            name="model.generate",
            timeout_seconds=120.0,
            attempts=1,
            criticality=HowBadIsLosingThis.REQUIRED,
            idempotent=IsItSafeToRepeat.YES,
            owner="model_lane",
            why="a second generation costs a full inference and the first may still land",
        )
    )
    declare_a_call(
        APolicy(
            name="memory.recall",
            timeout_seconds=8.0,
            attempts=2,
            backoff_seconds=0.1,
            criticality=HowBadIsLosingThis.DEGRADES,
            idempotent=IsItSafeToRepeat.YES,
            owner="memory_consolidation",
            cache_seconds=2.0,
            why="a read; the answer is worse without it but still an answer",
        )
    )
    declare_a_call(
        APolicy(
            name="state.write",
            timeout_seconds=20.0,
            attempts=3,
            backoff_seconds=0.2,
            criticality=HowBadIsLosingThis.REQUIRED,
            idempotent=IsItSafeToRepeat.WITH_A_KEY,
            owner="file_write_gateway",
            why="the gateway writes atomically to a path, so repeating lands once",
        )
    )
    declare_a_call(
        APolicy(
            name="spine.append",
            timeout_seconds=10.0,
            attempts=3,
            backoff_seconds=0.1,
            criticality=HowBadIsLosingThis.REQUIRED,
            idempotent=IsItSafeToRepeat.NO,
            owner="event_spine",
            why="an append repeated is two links in the chain for one event",
        )
    )
    declare_a_call(
        APolicy(
            name="tool.execute",
            timeout_seconds=60.0,
            attempts=1,
            criticality=HowBadIsLosingThis.DEGRADES,
            idempotent=IsItSafeToRepeat.NO,
            owner="cognitive_engine",
            why="a tool may have side effects and a timeout cannot say whether it ran",
        )
    )
    declare_a_call(
        APolicy(
            name="telemetry.emit",
            timeout_seconds=2.0,
            attempts=1,
            criticality=HowBadIsLosingThis.OPTIONAL,
            idempotent=IsItSafeToRepeat.YES,
            owner="telemetry",
            why="a lost sample is a gap in a chart",
        )
    )
    declare_a_call(
        APolicy(
            name="curiosity.wonder",
            timeout_seconds=15.0,
            attempts=1,
            criticality=HowBadIsLosingThis.OPTIONAL,
            idempotent=IsItSafeToRepeat.YES,
            owner="curiosity",
            why="nothing downstream waits on it",
        )
    )


_declare_what_ships()
