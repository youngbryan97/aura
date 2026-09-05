"""core/interiority/census.py — what actually happens, once she is running.

N2 on the docket, and the one claim the constructed proofs cannot make.
Every measurement so far runs in a world the harness builds: it shows
that each mechanism reaches the consumers it names and that removing it
is detectable. None of it shows how often anything fires in ordinary
use, which is a different question and the one most likely to embarrass
the design.

This is the instrument for it. It accumulates across real turns and
keeps three things the constructed world cannot produce:

**A firing rate per faculty.** A mechanism that never fires in a month
of use is not necessarily wrong — several here are for events that are
rare on purpose — but a mechanism that fires on nine turns in ten is
almost certainly matching something it should not.

**A histogram of decline reasons.** Declining is correct behaviour and
the reason says which evidence is missing. If one appraisal check is
absent in most turns, that is not forty-three faculties being careful,
it is one unconnected input starving the layer, and the histogram names
it where an aggregate would hide it.

**Channel availability over time.** Which senses were actually carrying
anything, turn by turn. A read is only as good as the channels behind
it, and the constructed proofs supply channels the running system may
not have.

It is bounded, written through the governed gateway, and reports rates
rather than only totals, so a long session and a short one can be
compared.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Interiority.Census")

#: Decline reasons are free text; only the leading clause is counted, so
#: "required appraisal checks are absent: other_capability" and the same
#: reason for a different check stay distinguishable without the tail.
_REASON_CLAUSE = 96


class Census:
    """A running tally of what the interior actually did."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.interiority.census.Census", reentrant=True)
        self.started_at = time.time()
        self._turns = 0
        self._fired: Counter[str] = Counter()
        #: Summed intensity per faculty. A rate alone cannot tell a
        #: mechanism that fires hard every turn from one that emits 0.02
        #: and would be better described as declining, and a threshold on
        #: the count would just be a number nobody measured.
        self._intensity: dict[str, float] = {}
        self._declined: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._channels: Counter[str] = Counter()
        self._constraints: Counter[str] = Counter()
        self._tendencies: Counter[str] = Counter()
        self._conflict_sum = 0.0

    def observe(self, state: Any, *, channels: tuple[str, ...] = ()) -> None:
        """Record one real turn. Never raises."""
        try:
            with self._lock:
                self._turns += 1
                for faculty, intensity in getattr(state, "transmitted", {}).items():
                    if intensity > 0.0:
                        self._fired[faculty] += 1
                        self._intensity[faculty] = (
                            self._intensity.get(faculty, 0.0) + float(intensity)
                        )
                for faculty, reason in getattr(state, "declines", {}).items():
                    self._declined[faculty] += 1
                    self._reasons[str(reason)[:_REASON_CLAUSE]] += 1
                for channel in channels:
                    self._channels[channel] += 1
                for constraint in getattr(state, "hard_constraints", ()):
                    self._constraints[constraint.action_class] += 1
                dominant = getattr(state, "dominant", ("", 0.0))[0]
                if dominant:
                    self._tendencies[dominant] += 1
                self._conflict_sum += float(getattr(state, "tendency_conflict", 0.0))
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.census", exc, action="turn not counted")

    def report(self) -> dict[str, Any]:
        with self._lock:
            turns = self._turns
            if turns == 0:
                return {
                    "turns": 0,
                    "note": (
                        "nothing has run yet. Every claim about this layer so "
                        "far is measured in a constructed world"
                    ),
                }
            def rate(counter: Counter[str]) -> dict[str, float]:
                return {
                    k: round(v / turns, 4)
                    for k, v in sorted(counter.items(), key=lambda kv: -kv[1])
                }

            never = None
            return {
                "turns": turns,
                "running_for_s": round(time.time() - self.started_at, 1),
                "firing_rate": rate(self._fired),
                "mean_intensity_when_fired": {
                    faculty: round(total / self._fired[faculty], 4)
                    for faculty, total in sorted(
                        self._intensity.items(), key=lambda kv: -kv[1]
                    )
                    if self._fired[faculty]
                },
                "decline_rate": rate(self._declined),
                "decline_reasons": dict(self._reasons.most_common(12)),
                "channel_availability": rate(self._channels),
                "constraints_held": dict(self._constraints.most_common(12)),
                "dominant_tendency": rate(self._tendencies),
                "mean_tendency_conflict": round(self._conflict_sum / turns, 4),
                "never_fired": never,
            }

    def never_fired(self, all_faculties: tuple[str, ...]) -> tuple[str, ...]:
        """Faculties that have not fired once.

        Not a defect on its own — several of these are for events that are
        rare on purpose — but the list worth reading first.
        """
        with self._lock:
            return tuple(sorted(f for f in all_faculties if self._fired[f] == 0))

    def always_fires(
        self, threshold: float = 0.9
    ) -> tuple[tuple[str, float, float], ...]:
        """Faculties firing on almost every turn, and doing so meaningfully.

        The likelier defect. A mechanism for a specific relational
        situation that fires on nine turns in ten is matching something it
        should not — but only if it is actually producing a state. One
        emitting 0.02 every turn is closer to declining than to firing.

        So the mean intensity is returned beside the rate rather than
        filtered on. An earlier version used a cutoff of 0.1, which was
        exactly the kind of number this package refuses everywhere else:
        undeclared, unmeasured, and deciding something. The reader judges,
        and both numbers are in front of them.
        """
        with self._lock:
            if self._turns == 0:
                return ()
            out: list[tuple[str, float, float]] = []
            for faculty, count in self._fired.items():
                rate = count / self._turns
                if rate < threshold:
                    continue
                mean = self._intensity.get(faculty, 0.0) / max(1, count)
                out.append((faculty, round(rate, 4), round(mean, 4)))
            return tuple(sorted(out))

    def persist(self, path: Path | None = None) -> bool:
        """Write the census through the governed gateway. Never raises."""
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway
            from core.runtime.state_ownership import state_root

            target = path or (state_root() / "data" / "interiority_census.json")
            with local_internal_governed_scope("interiority.census.persist"):
                get_file_write_gateway().write_json(
                    target,
                    self.report(),
                    schema_version=1,
                    schema_name="aura.interiority.census",
                    source="core/interiority/census.py",
                )
            return True
        except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
            record_degradation(
                "interiority.census", exc, action="census not persisted"
            )
            return False

    def reset_for_test(self) -> None:
        with self._lock:
            self.__init__()


_CENSUS: Census | None = None
_LOCK = checked_lock("core.interiority.census.singleton")


def get_census() -> Census:
    global _CENSUS
    with _LOCK:
        if _CENSUS is None:
            _CENSUS = Census()
        return _CENSUS


__all__ = ["Census", "get_census"]
