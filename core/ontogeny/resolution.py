"""L2 — honest credit assignment: outcomes that arrive when they actually arrive.

The executive already logs a completion four milliseconds after the decision.
That is a dispatch acknowledgement, not an outcome. Whether admitting that
intent was *right* is answerable minutes to days later: did the spawned task
finish, did the retrieved memory get used in the reply, did the answer pass its
verifier, did Bryan act on it or ask again.

So each control point registers a resolver that knows two things: how long to
wait, and how to look. When the horizon elapses the sweeper asks the resolver
what happened, and the resolver is allowed — required, when it is true — to
answer "I could not observe this." That answer is recorded as UNOBSERVED and
excluded from training.

That refusal is the entire discipline. A sweeper that writes off unresolved
work as failure produces a corpus that looks rich and teaches despair; the
existing outcome ledger does exactly that to a million rows. Here the default
is ignorance, and a resolver has to earn the right to assert a label.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.ontogeny.experience import Episode, ExperienceSpine, Outcome, OutcomeKind
from core.runtime.errors import record_degradation
from core.runtime.foreground_guard import foreground_activity_reason
from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.Ontogeny.Resolution")

#: Episodes swept per pass. Bounded so a backlog costs many small passes
#: rather than one long stall.
_SWEEP_BATCH = 500


@runtime_checkable
class OutcomeResolver(Protocol):
    """Knows how to find out what came of one control point's decisions."""

    #: The control point this resolver answers for.
    control_point: str
    #: How long to wait before asking. The resolver's own judgement about when
    #: the consequence of this kind of decision has had time to exist.
    horizon_s: float

    def resolve(self, episode: Episode) -> Outcome | None:
        """Return the outcome, or ``None`` when it genuinely cannot be observed."""


@dataclass
class CallableResolver:
    """A resolver built from a plain function, for subsystems that have one."""

    control_point: str
    horizon_s: float
    fn: Callable[[Episode], Outcome | None]
    name: str = ""

    def resolve(self, episode: Episode) -> Outcome | None:
        return self.fn(episode)


class ResolverRegistry:
    """One resolver per control point, plus the sweeper that runs them."""

    def __init__(self) -> None:
        self._lock = checked_lock("ontogeny.resolvers", rank=LockRank.LEAF)
        self._resolvers: dict[str, OutcomeResolver] = {}
        self._swept = 0
        self._observed = 0
        self._unobserved = 0
        self._errors = 0

    def register(self, resolver: OutcomeResolver) -> None:
        with self._lock:
            existing = self._resolvers.get(resolver.control_point)
            if existing is not None and existing is not resolver:
                logger.info(
                    "ontogeny: replacing resolver for %s (%s -> %s)",
                    resolver.control_point, type(existing).__name__, type(resolver).__name__,
                )
            self._resolvers[resolver.control_point] = resolver

    def register_fn(
        self, control_point: str, horizon_s: float, fn: Callable[[Episode], Outcome | None]
    ) -> None:
        self.register(CallableResolver(control_point=control_point, horizon_s=horizon_s, fn=fn))

    def horizon_for(self, control_point: str, default: float = 900.0) -> float:
        with self._lock:
            resolver = self._resolvers.get(control_point)
        return float(resolver.horizon_s) if resolver else float(default)

    def resolve_episode(self, episode: Episode) -> Outcome:
        """Ask the registered resolver, defaulting to honest ignorance."""
        with self._lock:
            resolver = self._resolvers.get(episode.control_point)
        if resolver is None:
            return Outcome.unobserved("sweeper:no_resolver", control_point=episode.control_point)
        try:
            outcome = resolver.resolve(episode)
        except (RuntimeError, AttributeError, TypeError, ValueError, KeyError, OSError) as exc:
            self._errors += 1
            record_degradation(
                "ontogeny_resolution", exc, severity="warning",
                action=f"resolver for {episode.control_point} raised; episode left unobserved",
            )
            return Outcome.unobserved("sweeper:resolver_error", error=type(exc).__name__)
        if outcome is None:
            return Outcome.unobserved("sweeper:not_observable")
        return outcome

    def sweep(
        self,
        spine: ExperienceSpine,
        *,
        limit: int = _SWEEP_BATCH,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, int]:
        """Resolve every episode whose horizon has elapsed.

        Unresolvable episodes become UNOBSERVED, which closes them without
        teaching anything. That is the correct outcome for an unobserved
        consequence, and it is what keeps the corpus from filling with
        confident falsehoods.
        """
        due = spine.open_episodes(older_than_horizon=True, limit=limit)
        observed = unobserved = 0
        processed = 0
        for episode in due:
            if should_stop is not None and should_stop():
                break
            outcome = self.resolve_episode(episode)
            spine.resolve(episode.episode_id, outcome)
            processed += 1
            if outcome.kind is OutcomeKind.UNOBSERVED:
                unobserved += 1
            else:
                observed += 1
        self._swept += processed
        self._observed += observed
        self._unobserved += unobserved
        if processed:
            spine.flush()
        return {"swept": processed, "observed": observed, "unobserved": unobserved}

    def report(self) -> dict[str, object]:
        with self._lock:
            registered = {
                cp: {"horizon_s": r.horizon_s, "resolver": type(r).__name__}
                for cp, r in self._resolvers.items()
            }
        total = self._observed + self._unobserved
        return {
            "resolvers": registered,
            "swept": self._swept,
            "observed": self._observed,
            "unobserved": self._unobserved,
            "observation_rate": round(self._observed / total, 4) if total else 0.0,
            "resolver_errors": self._errors,
        }


class OutcomeSweeper:
    """Runs the registry on a cadence, off the cognitive path."""

    def __init__(self, registry: ResolverRegistry, spine: ExperienceSpine, *, interval_s: float = 60.0) -> None:
        self._registry = registry
        self._spine = spine
        self._interval = float(interval_s)
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_pass: dict[str, int] = {}
        self._last_at = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="ontogeny-sweeper", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stopped.wait(self._interval):
            try:
                if foreground_activity_reason():
                    continue
                self._last_pass = self._registry.sweep(
                    self._spine,
                    should_stop=lambda: bool(foreground_activity_reason()),
                )
                self._last_at = time.time()
            except (RuntimeError, OSError, ValueError) as exc:
                record_degradation(
                    "ontogeny_resolution", exc, severity="warning",
                    action="outcome sweep failed; retrying next cycle",
                )

    def stop(self) -> None:
        self._stopped.set()

    def report(self) -> dict[str, object]:
        return {
            "interval_s": self._interval,
            "running": self._thread is not None and self._thread.is_alive(),
            "last_pass": dict(self._last_pass),
            "last_pass_age_s": round(time.time() - self._last_at, 1) if self._last_at else None,
        }


_registry: ResolverRegistry | None = None
_registry_lock = checked_lock("core.ontogeny.resolution._registry_lock")


def get_resolvers() -> ResolverRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ResolverRegistry()
    return _registry


__all__ = [
    "CallableResolver",
    "OutcomeResolver",
    "OutcomeSweeper",
    "ResolverRegistry",
    "get_resolvers",
]
