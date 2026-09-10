"""L1 — observed counterfactuals: keeping the road not taken measurable.

A log of what you did and what happened lets you learn to *predict*. It does
not let you learn to *choose*. The difference is the counterfactual: to know
whether a different decision would have gone better, something has to
occasionally make the different decision and let the world answer.

This is the layer that is missing from almost every "shadow mode then promote"
proposal, and its absence is not a gap in the evidence — it is what makes the
evidence unfalsifiable. Once a learned controller takes over a decision, it
generates all its own future training data. Its mistakes stop being visible as
mistakes, because the alternative that would have exposed them is never run
again. The comparison that justified the promotion can never be repeated, and
the system slowly, confidently drifts with nothing able to contradict it.

So a slice of episodes is permanently reserved for the *other* decider, in
both directions:

  * While the head is still learning, a small probe slice lets the head
    actually act on low-stakes episodes — otherwise its preferred actions are
    never tried and never scored, and it can only ever learn to imitate the
    incumbent rather than to beat it.
  * After the head holds authority, a permanent slice goes back to the
    incumbent. This never expires. It is the cost of being able to say, a year
    later, "and here is the incumbent's score over the same period."

Reservations are deterministic in the episode seed, so a replayed corpus makes
the identical choices; rate-limited, so a burst of episodes cannot spend the
whole budget; and refused outright above a stakes ceiling, because the price
of evidence is paid in small change, never with something that matters.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.Ontogeny.Reservation")


class Decider(StrEnum):
    """Who actually chooses on this episode."""

    #: The hand-written rules that have always made this decision.
    INCUMBENT = "incumbent"
    #: The learned head. Either because it holds authority, or because this
    #: episode was reserved as a probe so its counterfactual gets observed.
    CHALLENGER = "challenger"
    #: Nobody: the action is drawn uniformly from the safe set.
    #:
    #: This one exists for a reason that is easy to miss and fatal to skip.
    #: The incumbent's action is very nearly a deterministic function of the
    #: incumbent's features — that is what a rule *is*. So in a corpus built
    #: only from incumbent decisions, action and situation are confounded to
    #: the point of being the same variable, and no amount of that data can
    #: say what a different action would have done. Statisticians call the
    #: missing condition positivity: every action needs some probability of
    #: being taken in every region of the situation space, or its effect is
    #: not identifiable from the record at all.
    #:
    #: A thin slice of uniformly random low-stakes actions supplies exactly
    #: that. It is the only part of the corpus that is causally clean, and
    #: without it a learned controller can imitate the rules but can never
    #: discover that a different rule would be better.
    RANDOM = "random"


@dataclass(frozen=True)
class Reservation:
    """The decision about who decides — itself receipted."""

    decider: Decider
    #: Why this decider: ``"authority"``, ``"probe"``, ``"counterfactual"``,
    #: ``"stakes_ceiling"``, ``"budget_exhausted"``, ``"default"``.
    reason: str
    reserved: bool = False

    @property
    def is_probe(self) -> bool:
        return self.reserved and self.decider is Decider.CHALLENGER

    @property
    def is_counterfactual(self) -> bool:
        return self.reserved and self.decider is Decider.INCUMBENT

    @property
    def is_random(self) -> bool:
        return self.decider is Decider.RANDOM


#: One episode in this many goes to the challenger while it is still learning.
#: Low, because a probe means an unvalidated model chose a real action.
PROBE_DENOMINATOR = 32

#: One episode in this many takes a uniformly random action from the safe set.
#: This is the causally clean slice — the only episodes from which the effect
#: of an action can be separated from the situation that provoked it. It runs
#: at every stage, forever, on low-stakes episodes only.
RANDOM_DENOMINATOR = 24

#: One episode in this many goes back to the incumbent after the challenger
#: holds authority. Higher rate than probing: this slice is the only thing
#: standing between a promoted head and unfalsifiable drift, and it is
#: permanent. Twelve percent of decisions is a cheap insurance premium.
COUNTERFACTUAL_DENOMINATOR = 8

#: Stakes above this are never reserved in either direction. Evidence is
#: bought with small change.
DEFAULT_STAKES_CEILING = 0.7

#: Reservations per control point per hour. A burst cannot spend the budget.
DEFAULT_HOURLY_BUDGET = 60


class ExplorationReservation:
    """Decides who decides, deterministically and within budget."""

    def __init__(
        self,
        *,
        probe_denominator: int = PROBE_DENOMINATOR,
        counterfactual_denominator: int = COUNTERFACTUAL_DENOMINATOR,
        random_denominator: int = RANDOM_DENOMINATOR,
        stakes_ceiling: float = DEFAULT_STAKES_CEILING,
        hourly_budget: int = DEFAULT_HOURLY_BUDGET,
    ) -> None:
        self._probe_denom = max(1, int(probe_denominator))
        self._counterfactual_denom = max(1, int(counterfactual_denominator))
        self._random_denom = max(1, int(random_denominator))
        self._stakes_ceiling = float(stakes_ceiling)
        self._hourly_budget = max(0, int(hourly_budget))
        self._lock = checked_lock("ontogeny.reservation", rank=LockRank.LEAF)
        self._spent: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}

    def decide(
        self,
        control_point: str,
        *,
        seed: str,
        stakes: float,
        has_authority: bool,
        challenger_ready: bool,
    ) -> Reservation:
        """Choose the decider for one episode.

        ``challenger_ready`` means a head exists and can produce a decision at
        all — a head with no fitted weights is never allowed to act, probe
        slice or not.
        """
        if stakes > self._stakes_ceiling:
            base = Decider.CHALLENGER if has_authority else Decider.INCUMBENT
            return Reservation(decider=base, reason="stakes_ceiling")

        # The causally clean slice comes first and runs at every stage. It is
        # drawn on its own hash so it never collides with the other slices,
        # and it never stops — an organ that has held authority for a year
        # still needs episodes whose action was not chosen by anything.
        if self._draw(f"{control_point}#random", seed, self._random_denom):
            if self._spend(control_point):
                return Reservation(Decider.RANDOM, "positivity", reserved=True)

        if has_authority:
            # The permanent counterfactual slice. This is the one that never
            # expires, and the one that makes revocation evidence-based.
            if self._draw(control_point, seed, self._counterfactual_denom):
                if self._spend(control_point):
                    return Reservation(Decider.INCUMBENT, "counterfactual", reserved=True)
                return Reservation(Decider.CHALLENGER, "budget_exhausted")
            return Reservation(Decider.CHALLENGER, "authority")

        if challenger_ready and self._draw(control_point, seed, self._probe_denom):
            if self._spend(control_point):
                return Reservation(Decider.CHALLENGER, "probe", reserved=True)
            return Reservation(Decider.INCUMBENT, "budget_exhausted")

        return Reservation(Decider.INCUMBENT, "default")

    @staticmethod
    def _draw(control_point: str, seed: str, denominator: int) -> bool:
        """Deterministic in (control point, seed): a replay reserves identically."""
        digest = hashlib.sha256(f"{control_point}|{seed}".encode()).digest()
        return (int.from_bytes(digest[:4], "big") % denominator) == 0

    def _spend(self, control_point: str) -> bool:
        now = time.time()
        with self._lock:
            spent = self._spent.setdefault(control_point, deque())
            while spent and now - spent[0] > 3600.0:
                spent.popleft()
            if len(spent) >= self._hourly_budget:
                return False
            spent.append(now)
            self._counts[control_point] = self._counts.get(control_point, 0) + 1
            return True

    def report(self) -> dict[str, object]:
        with self._lock:
            recent = {cp: len(times) for cp, times in self._spent.items()}
            total = dict(self._counts)
        return {
            "probe_rate": round(1.0 / self._probe_denom, 4),
            "counterfactual_rate": round(1.0 / self._counterfactual_denom, 4),
            "random_rate": round(1.0 / self._random_denom, 4),
            "stakes_ceiling": self._stakes_ceiling,
            "hourly_budget": self._hourly_budget,
            "reserved_last_hour": recent,
            "reserved_total": total,
        }


_reservation: ExplorationReservation | None = None
_reservation_lock = checked_lock("core.ontogeny.reservation._reservation_lock")


def get_reservation() -> ExplorationReservation:
    global _reservation
    if _reservation is None:
        with _reservation_lock:
            if _reservation is None:
                _reservation = ExplorationReservation()
    return _reservation


__all__ = [
    "COUNTERFACTUAL_DENOMINATOR",
    "DEFAULT_HOURLY_BUDGET",
    "DEFAULT_STAKES_CEILING",
    "PROBE_DENOMINATOR",
    "RANDOM_DENOMINATOR",
    "Decider",
    "ExplorationReservation",
    "Reservation",
    "get_reservation",
]
