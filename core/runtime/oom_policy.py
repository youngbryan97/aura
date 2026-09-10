"""core/runtime/oom_policy.py — badness scoring and victim selection.

Clean-room adoption of the Linux kernel OOM killer's *policy*, not its
mechanism.

The kernel's insight is that under memory exhaustion, dying is not the
worst outcome — dying **arbitrarily** is. So it does not kill whatever
allocated last. It scores every candidate by `oom_badness()`:
proportional footprint, adjusted by a per-process `oom_score_adj` that
policy sets ahead of time, with `OOM_SCORE_ADJ_MIN` marking processes that
must never be chosen. Then it kills the single best victim, logs the full
scoring table, and lets the machine live.

Aura has been on the wrong side of this twice in the recorded history: an
endurance run that reached 35GB and was SIGKILLed whole, and a duplicate
32B load that doubled memory and took the wedged runtime with it. In both
cases the host chose the victim, and the host's choice is always "the
biggest process", which is always the runtime itself. Choosing first, and
choosing a *part*, is strictly better than being chosen.

What this module provides:

* A registry of sheddable organs with declared `oom_score_adj` and a
  footprint provider.
* Badness scoring identical in shape to the kernel's: proportional
  footprint in thousandths, plus the adjustment, with an immune floor.
* Victim selection and staged shedding until a free-memory target is met.
* A permanent, append-only shed log — the kernel prints its scoring table
  before it kills, and being able to answer "why *that* organ" six hours
  later is the whole value.
* An honest terminal path: when nothing sheddable remains and pressure is
  still critical, this asks for a controlled restart rather than waiting
  to be SIGKILLed. A restart that saves state beats a kill that does not.

Registration is declarative and happens at organ construction, so the
policy table is complete before the pressure arrives — deciding who dies
during an emergency is how you get the wrong answer.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.taint import TaintFlag, taint

logger = logging.getLogger("Aura.OOM")

#: Kernel constants, same meaning.
OOM_SCORE_ADJ_MIN = -1000
OOM_SCORE_ADJ_MAX = 1000

#: Badness is reported in thousandths of available memory, like the kernel.
BADNESS_SCALE = 1000


@dataclass(frozen=True)
class OrganPolicy:
    """One registered candidate."""

    name: str
    oom_score_adj: int
    #: Returns the organ's current footprint in bytes, best effort.
    footprint: Callable[[], int]
    #: Frees the organ's memory. Returns bytes actually released (best
    #: effort) — a shed that frees nothing must say so, or the loop below
    #: will keep asking it.
    shed: Callable[[], int] | None
    #: Human sentence for the shed log and the incident narrator.
    rationale: str
    #: True when the organ can be brought back after shedding.
    recoverable: bool

    @property
    def immune(self) -> bool:
        return self.oom_score_adj <= OOM_SCORE_ADJ_MIN


@dataclass
class ShedEvent:
    at: float
    victim: str
    badness: int
    freed_bytes: int
    reason: str
    table: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "victim": self.victim,
            "badness": self.badness,
            "freed_bytes": self.freed_bytes,
            "reason": self.reason,
            "table": self.table,
        }


class OomPolicy:
    """The registry, the scoring, and the shed ladder."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._organs: dict[str, OrganPolicy] = {}
        self._events: list[ShedEvent] = []
        self._restart_requested = False

    # ── registration ──────────────────────────────────────────────────
    def register(
        self,
        name: str,
        *,
        oom_score_adj: int = 0,
        footprint: Callable[[], int] | None = None,
        shed: Callable[[], int] | None = None,
        rationale: str = "",
        recoverable: bool = True,
        nominal_bytes: int = 0,
    ) -> OrganPolicy:
        """Declare an organ's position in the shed order.

        ``oom_score_adj`` follows the kernel's convention exactly:
        ``-1000`` is immune, ``0`` is neutral, positive values volunteer.
        Give the load-bearing organs (the Will, the flight recorder, the
        health surface, the shutdown coordinator) ``-1000``; give
        speculative background work large positives.
        """
        adj = max(OOM_SCORE_ADJ_MIN, min(OOM_SCORE_ADJ_MAX, int(oom_score_adj)))
        provider = footprint or (lambda: int(nominal_bytes))
        policy = OrganPolicy(
            name=name,
            oom_score_adj=adj,
            footprint=provider,
            shed=shed,
            rationale=rationale or f"{name} registered without a stated rationale",
            recoverable=recoverable,
        )
        with self._lock:
            self._organs[name] = policy
        return policy

    def unregister(self, name: str) -> None:
        with self._lock:
            self._organs.pop(name, None)

    # ── scoring ───────────────────────────────────────────────────────
    @staticmethod
    def _safe_footprint(policy: OrganPolicy) -> int:
        try:
            return max(0, int(policy.footprint()))
        except Exception:  # noqa: BLE001 — a broken probe must not block shedding
            logger.debug("footprint probe failed for %s", policy.name, exc_info=True)
            return 0

    def badness(
        self, policy: OrganPolicy, total_bytes: int, *, footprint_bytes: int | None = None
    ) -> int:
        """Proportional footprint in thousandths, plus the adjustment.

        Matches the kernel's shape: an organ holding half of memory scores
        500 before adjustment, so an ``oom_score_adj`` of +500 makes a
        quarter-sized organ tie with it — which is what "volunteer twice
        as readily" should mean.
        """
        if policy.immune:
            return OOM_SCORE_ADJ_MIN
        if total_bytes <= 0:
            return policy.oom_score_adj
        footprint = self._safe_footprint(policy) if footprint_bytes is None else footprint_bytes
        share = footprint / float(total_bytes)
        return int(round(share * BADNESS_SCALE)) + policy.oom_score_adj

    def scoring_table(self, total_bytes: int | None = None) -> list[dict[str, Any]]:
        """The full table, highest badness first. Logged before every shed."""
        total = total_bytes if total_bytes is not None else _total_memory_bytes()
        with self._lock:
            organs = list(self._organs.values())
        rows = []
        for p in organs:
            footprint = self._safe_footprint(p)
            rows.append({
                "organ": p.name,
                "badness": self.badness(p, total, footprint_bytes=footprint),
                "footprint_bytes": footprint,
                "oom_score_adj": p.oom_score_adj,
                "sheddable": p.shed is not None,
                "immune": p.immune,
                "recoverable": p.recoverable,
                "rationale": p.rationale,
            })
        rows.sort(key=lambda r: (-r["badness"], r["organ"]))
        return rows

    def select_victim(self, total_bytes: int | None = None) -> OrganPolicy | None:
        """Highest badness among organs that can actually be shed."""
        total = total_bytes if total_bytes is not None else _total_memory_bytes()
        with self._lock:
            candidates = [p for p in self._organs.values() if p.shed is not None and not p.immune]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (self.badness(p, total), p.name))

    # ── shedding ──────────────────────────────────────────────────────
    def shed_until(
        self,
        *,
        target_free_bytes: int,
        free_bytes_now: Callable[[], int],
        reason: str,
        max_victims: int = 4,
    ) -> list[ShedEvent]:
        """Shed the best victims, in order, until the target is met.

        Stops as soon as ``free_bytes_now()`` clears the target, so a
        single well-chosen victim usually ends it. Each shed is logged
        with the full scoring table that justified it.
        """
        events: list[ShedEvent] = []
        total = _total_memory_bytes()
        for _ in range(max_victims):
            try:
                if free_bytes_now() >= target_free_bytes:
                    break
            except Exception:  # noqa: BLE001 — probe failure means keep shedding
                logger.debug("free-memory probe failed", exc_info=True)
            victim = self.select_victim(total)
            if victim is None:
                break
            table = self.scoring_table(total)
            badness = self.badness(victim, total)
            before = self._safe_footprint(victim)
            freed = 0
            try:
                freed = int(victim.shed() or 0) if victim.shed else 0
            except Exception as exc:  # noqa: BLE001 — organ callbacks are external boundaries
                logger.error("OOM shed of %s failed: %s", victim.name, exc)
                freed = 0
            if freed <= 0:
                freed = max(0, before - self._safe_footprint(victim))

            event = ShedEvent(
                at=time.time(),
                victim=victim.name,
                badness=badness,
                freed_bytes=freed,
                reason=reason,
                table=table,
            )
            events.append(event)
            with self._lock:
                self._events.append(event)
                if len(self._events) > 256:
                    del self._events[:-256]
            logger.warning(
                "🩸 OOM shed: %s (badness=%d) freed %.1fMB — %s",
                victim.name,
                badness,
                freed / 1e6,
                reason,
            )
            taint(
                TaintFlag.OOM_SHED,
                f"shed {victim.name} (badness={badness}) under: {reason}",
                subsystem="oom_policy",
            )
            _append_shed_log(event)
            if freed <= 0 and victim.shed is not None:
                # A victim that frees nothing must not be picked again in
                # this pass, or the loop degenerates.
                with self._lock:
                    self._organs[victim.name] = OrganPolicy(
                        name=victim.name,
                        oom_score_adj=victim.oom_score_adj,
                        footprint=victim.footprint,
                        shed=None,
                        rationale=victim.rationale + " [shed returned nothing]",
                        recoverable=victim.recoverable,
                    )
        return events

    def no_victim_available(self) -> bool:
        return self.select_victim() is None

    def request_controlled_restart(self, reason: str) -> bool:
        """Terminal path: nothing left to shed and pressure is still critical.

        Being SIGKILLed loses the episodic buffer, the flight-recorder
        tail, and any in-flight turn. A controlled restart does not.
        Returns True if a restart was requested (idempotent per process).
        """
        with self._lock:
            if self._restart_requested:
                return False
            self._restart_requested = True

        logger.critical(
            "🛑 OOM terminal: nothing sheddable remains — requesting controlled "
            "restart rather than waiting to be killed. reason=%s",
            reason,
        )
        taint(TaintFlag.OOM_SHED, f"terminal OOM, controlled restart: {reason}", subsystem="oom_policy")
        requested = False
        try:
            from core.runtime.shutdown_coordinator import request_shutdown

            # A clean shutdown runs the persistence phase and lets the
            # keepalive supervisor relaunch; SIGKILL does neither.
            request_shutdown(reason=f"oom_terminal: {reason}")
            requested = True
        except Exception:  # noqa: BLE001 — terminal recovery checks every available channel
            logger.debug("controlled shutdown channel unavailable", exc_info=True)
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "oom_policy",
                MemoryError(reason),
                severity="critical",
                action=(
                    "requested controlled restart"
                    if requested
                    else "no restart channel available; recorded terminal OOM"
                ),
            )
        except Exception:  # noqa: BLE001 — terminal evidence cannot block recovery
            logger.debug("terminal OOM degradation record failed", exc_info=True)
        return True

    def events(self) -> list[ShedEvent]:
        with self._lock:
            return list(self._events)

    def report(self) -> dict[str, Any]:
        total = _total_memory_bytes()
        table = self.scoring_table(total)
        with self._lock:
            events = [e.to_dict() for e in self._events[-16:]]
            registered = len(self._organs)
        return {
            "total_memory_bytes": total,
            "registered_organs": registered,
            "sheddable_organs": sum(1 for r in table if r["sheddable"]),
            "immune_organs": [r["organ"] for r in table if r["immune"]],
            "next_victim": next((r["organ"] for r in table if r["sheddable"]), None),
            "scoring_table": table,
            "recent_sheds": events,
            "restart_requested": self._restart_requested,
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._organs.clear()
            self._events.clear()
            self._restart_requested = False


def _total_memory_bytes() -> int:
    try:
        from core.runtime.resource_psutil import virtual_memory

        return int(getattr(virtual_memory(), "total", 0) or 0)
    except Exception:  # noqa: BLE001 — scoring degrades to adjustment-only
        logger.debug("virtual_memory unavailable for OOM scoring", exc_info=True)
        return 0


def _append_shed_log(event: ShedEvent) -> None:
    """Append-only shed log. Governed write; never blocks the shed path."""
    try:
        from core.config import config
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        path = config.paths.data_dir / "error_logs" / "memory" / "oom_shed_log.jsonl"
        line = json.dumps(event.to_dict(), separators=(",", ":")) + "\n"
        with local_internal_governed_scope("oom_policy.shed_log"):
            get_file_write_gateway().append_text(path, line, source="oom_policy")
    except Exception:  # noqa: BLE001 — the log is evidence, not a dependency
        logger.debug("OOM shed log append failed", exc_info=True)


_POLICY = OomPolicy()


def get_oom_policy() -> OomPolicy:
    return _POLICY


def register_organ(
    name: str,
    *,
    oom_score_adj: int = 0,
    footprint: Callable[[], int] | None = None,
    shed: Callable[[], int] | None = None,
    rationale: str = "",
    recoverable: bool = True,
    nominal_bytes: int = 0,
) -> OrganPolicy:
    return _POLICY.register(
        name,
        oom_score_adj=oom_score_adj,
        footprint=footprint,
        shed=shed,
        rationale=rationale,
        recoverable=recoverable,
        nominal_bytes=nominal_bytes,
    )


def oom_report() -> dict[str, Any]:
    return _POLICY.report()


def reset_oom_policy_for_test() -> None:
    _POLICY.reset_for_test()


__all__ = [
    "BADNESS_SCALE",
    "OOM_SCORE_ADJ_MAX",
    "OOM_SCORE_ADJ_MIN",
    "OomPolicy",
    "OrganPolicy",
    "ShedEvent",
    "get_oom_policy",
    "oom_report",
    "register_organ",
    "reset_oom_policy_for_test",
]
