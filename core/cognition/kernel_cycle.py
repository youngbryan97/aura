"""core/cognition/kernel_cycle.py — one cycle the loops are configurations of.

Aura has several executive loops. The legacy pipeline serves chat. The kernel
tick serves the continuous mind. Screen pursuit runs its own perceive-act loop.
The planner recurses. Each was written for its own traffic and each reinvented
the same seven steps in a different order with different names, so a fix to one
does not reach the others and a phase that misbehaves in one is hard to
recognise in the next.

Seven steps, and the order carries the meaning:

    perceive -> propose -> elaborate -> prefer -> select -> apply -> verify -> learn

* **Elaborate runs to quiescence.** Proposals may generate proposals, and the
  cycle does not move on while anything is still adding. Selecting from a
  partial field is the defect underneath "she decided before she had finished
  thinking", and a step limit turns a runaway into a reported impasse rather
  than a hang.
* **Prefer is symbolic, select is not.** Preference runs first and removes;
  scoring runs afterward among what survives. A prohibition that a score could
  outbid is not a prohibition.
* **Verify is not optional.** A cycle that applies and does not verify produces
  the receipt-less action the learners are forbidden to train on.

Configuration, not inheritance
------------------------------
A loop is a :class:`CycleConfig`: which phases run, what budget, whether
learning is on. The chat pipeline skips nothing; a fast reflex path skips
elaborate and learn; a dry run skips apply. Five existing loops become five
configurations rather than five implementations, and a phase fixed once is
fixed everywhere it runs.

One transaction per action
--------------------------
Each turn of the cycle opens exactly one selection record and exactly one
verification record, both on the event graph. That is card A12.1's bar, and it
is what lets an action be traced to the decision that produced it rather than
to the log line nearest in time.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.cognition.cognitive_event import Phase, cycle, get_event_graph

__all__ = [
    "Step",
    "CycleConfig",
    "CycleResult",
    "Handlers",
    "CognitiveKernel",
    "CHAT_PIPELINE",
    "REFLEX",
    "DRY_RUN",
    "DELIBERATE",
    "PURSUIT",
]


class Step(StrEnum):
    PERCEIVE = "perceive"
    PROPOSE = "propose"
    ELABORATE = "elaborate"
    PREFER = "prefer"
    SELECT = "select"
    APPLY = "apply"
    VERIFY = "verify"
    LEARN = "learn"


#: Steps that may not be switched off. Selecting without preferring lets a
#: score outbid a prohibition; applying without verifying produces the
#: receipt-less action the learners are forbidden to train on.
REQUIRED = frozenset({Step.SELECT, Step.PREFER})


@dataclass(frozen=True, slots=True)
class CycleConfig:
    """One loop, as a configuration of the same seven steps."""

    name: str
    steps: frozenset[Step]
    max_elaboration: int = 8
    learning: bool = True
    seconds: float = 30.0

    def __post_init__(self) -> None:
        missing = REQUIRED - self.steps
        if missing:
            raise ValueError(
                f"{self.name!r} omits {sorted(s.value for s in missing)}; selecting "
                "without preferring lets a score outbid a prohibition"
            )
        if Step.APPLY in self.steps and Step.VERIFY not in self.steps:
            raise ValueError(
                f"{self.name!r} applies without verifying; an action with no receipt is "
                "one no learner may train on"
            )

    def runs(self, step: Step) -> bool:
        return step in self.steps


CHAT_PIPELINE = CycleConfig("legacy_pipeline", frozenset(Step), max_elaboration=8)
DELIBERATE = CycleConfig("kernel_tick", frozenset(Step), max_elaboration=16, seconds=60.0)
PURSUIT = CycleConfig(
    "screen_pursuit",
    frozenset({Step.PERCEIVE, Step.PROPOSE, Step.PREFER, Step.SELECT, Step.APPLY, Step.VERIFY, Step.LEARN}),
    max_elaboration=2, seconds=10.0,
)
REFLEX = CycleConfig(
    "reflex",
    frozenset({Step.PERCEIVE, Step.PROPOSE, Step.PREFER, Step.SELECT, Step.APPLY, Step.VERIFY}),
    max_elaboration=0, learning=False, seconds=1.0,
)
DRY_RUN = CycleConfig(
    "dry_run",
    frozenset({Step.PERCEIVE, Step.PROPOSE, Step.ELABORATE, Step.PREFER, Step.SELECT}),
    learning=False,
)


@dataclass(frozen=True, slots=True)
class CycleResult:
    """What one turn of the cycle did, and the two records it opened."""

    config: str
    cycle_id: int
    ran: tuple[Step, ...]
    chosen: Any = None
    applied: bool = False
    verified: bool | None = None
    learned: bool = False
    elaboration_rounds: int = 0
    selection_event: int = 0
    verification_event: int = 0
    impasse: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "cycle_id": self.cycle_id,
            "ran": [s.value for s in self.ran],
            "chosen": repr(self.chosen) if self.chosen is not None else None,
            "applied": self.applied,
            "verified": self.verified,
            "learned": self.learned,
            "elaboration_rounds": self.elaboration_rounds,
            "selection_event": self.selection_event,
            "verification_event": self.verification_event,
            "impasse": self.impasse,
            "seconds": self.seconds,
        }


@dataclass
class Handlers:
    """What a caller plugs into the seven steps. Every one is optional."""

    perceive: Callable[[], Mapping[str, Any]] | None = None
    propose: Callable[[Mapping[str, Any]], Sequence[Any]] | None = None
    elaborate: Callable[[Mapping[str, Any], Sequence[Any]], Sequence[Any]] | None = None
    prefer: Callable[[Sequence[Any]], Sequence[Any]] | None = None
    select: Callable[[Sequence[Any]], Any] | None = None
    apply: Callable[[Any], Any] | None = None
    verify: Callable[[Any, Any], bool] | None = None
    learn: Callable[[CycleResult], None] | None = None


class CognitiveKernel:
    """The one cycle. Loops differ by configuration, not by implementation."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = checked_lock("core.cognition.kernel_cycle.CognitiveKernel", reentrant=True)
        self._clock = clock
        self._turns: dict[str, int] = {}
        self._impasses: dict[str, int] = {}
        self._unverified: dict[str, int] = {}

    def turn(self, config: CycleConfig, handlers: Handlers) -> CycleResult:
        """Run one turn. Exactly one selection record and one verification record."""
        started = self._clock()
        graph = get_event_graph()
        ran: list[Step] = []
        situation: Mapping[str, Any] = {}
        candidates: list[Any] = []
        rounds = 0
        impasse = ""
        chosen = None
        applied = False
        verified: bool | None = None
        learned = False
        selection_event = 0
        verification_event = 0

        with cycle(config.name) as scope:
            if config.runs(Step.PERCEIVE) and handlers.perceive:
                situation = dict(handlers.perceive() or {})
                ran.append(Step.PERCEIVE)
                graph.record(Phase.PERCEIVE, config.name, "perceive", loop=config.name)

            if config.runs(Step.PROPOSE) and handlers.propose:
                candidates = list(handlers.propose(situation) or ())
                ran.append(Step.PROPOSE)

            if config.runs(Step.ELABORATE) and handlers.elaborate:
                ran.append(Step.ELABORATE)
                # Run to quiescence: proposals may generate proposals, and
                # selecting from a partial field is deciding before finishing.
                while rounds < config.max_elaboration:
                    added = list(handlers.elaborate(situation, candidates) or ())
                    fresh = [c for c in added if c not in candidates]
                    rounds += 1
                    if not fresh:
                        break
                    candidates.extend(fresh)
                else:
                    impasse = (
                        f"no_change: elaboration did not settle in {config.max_elaboration} "
                        "rounds"
                    )

            if config.runs(Step.PREFER) and handlers.prefer:
                candidates = list(handlers.prefer(candidates) or ())
                ran.append(Step.PREFER)

            if not candidates:
                impasse = impasse or "rejection: nothing survived preference"
            elif config.runs(Step.SELECT) and handlers.select:
                chosen = handlers.select(candidates)
                ran.append(Step.SELECT)

            selection = graph.record(
                Phase.SELECT, config.name, "select", loop=config.name,
                outcome="chosen" if chosen is not None else (impasse or "nothing"),
                detail={"candidates": len(candidates), "elaboration_rounds": rounds},
            )
            selection_event = selection.seq

            if chosen is not None and config.runs(Step.APPLY) and handlers.apply:
                outcome = handlers.apply(chosen)
                applied = True
                ran.append(Step.APPLY)
                if config.runs(Step.VERIFY) and handlers.verify:
                    verified = bool(handlers.verify(chosen, outcome))
                    ran.append(Step.VERIFY)
                verification = graph.record(
                    Phase.VERIFY, config.name, "verify", loop=config.name,
                    parents=[selection_event],
                    outcome="verified" if verified else "unverified",
                )
                verification_event = verification.seq

            result = CycleResult(
                config=config.name, cycle_id=scope.cycle_id, ran=tuple(ran), chosen=chosen,
                applied=applied, verified=verified, learned=False,
                elaboration_rounds=rounds, selection_event=selection_event,
                verification_event=verification_event, impasse=impasse,
                seconds=self._clock() - started,
            )

            if config.learning and config.runs(Step.LEARN) and handlers.learn:
                handlers.learn(result)
                learned = True
                ran.append(Step.LEARN)
                graph.record(Phase.LEARN, config.name, "learn", loop=config.name,
                             parents=[selection_event])

        final = CycleResult(
            config=result.config, cycle_id=result.cycle_id, ran=tuple(ran), chosen=chosen,
            applied=applied, verified=verified, learned=learned,
            elaboration_rounds=rounds, selection_event=selection_event,
            verification_event=verification_event, impasse=impasse,
            seconds=self._clock() - started,
        )
        with self._lock:
            self._turns[config.name] = self._turns.get(config.name, 0) + 1
            if impasse:
                self._impasses[config.name] = self._impasses.get(config.name, 0) + 1
            if applied and verified is not True:
                self._unverified[config.name] = self._unverified.get(config.name, 0) + 1
        return final

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loops": sorted(self._turns),
                "turns": dict(sorted(self._turns.items())),
                "impasses": dict(sorted(self._impasses.items())),
                "applied_without_verification": dict(sorted(self._unverified.items())),
                "configurations_of_one_kernel": len(self._turns),
            }
