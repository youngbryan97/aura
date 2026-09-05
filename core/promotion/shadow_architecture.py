"""core/promotion/shadow_architecture.py — trying a different design safely.

Aura can change her own code. What she cannot do is try a different *design*
and find out whether it is better, because the only way to find out is to run
it, and the only place it can run is where a mistake costs something.

A shadow is the answer to that. An alternative implementation receives every
input the live one receives, produces its own answer, and that answer goes
nowhere. Over enough turns the two disagree somewhere, and the disagreements
are the finding — a shadow that never disagrees is the same design wearing a
different name, and a shadow that disagrees everywhere is not a candidate for
anything.

The output never reaching the caller is structural rather than a rule anybody
follows: :meth:`Shadow.run` returns the live answer and the shadow's is kept
inside. There is no parameter that changes that, and a caller cannot get the
shadow's answer out of the return value, because a shadow that can be switched
on by an argument is not a shadow — it is a feature flag, and those get
flipped.

Migration takes more than being better on average. Three things, and the third
is the one that is usually skipped:

* the shadow satisfies the same behavioural contracts the live one does;
* it has been tried on enough distinct inputs to have been tried at all;
* and it does not regress on anything the live one got right.

That last one matters because an average is a poor summary of a change. A
design that is better on nine cases and destroys the tenth has a good mean and
should not ship, and "no regression on what already worked" is the condition
that says so.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Promotion.Shadow")

#: Distinct inputs before a comparison means anything. A shadow tried twice
#: has not been tried.
MIN_TRIALS = 20

#: Disagreements below which the two designs are the same design. A shadow
#: that never differs is not a candidate; it is a rename.
MIN_DISAGREEMENTS = 3

#: How much of the live behaviour a shadow may change and still be considered
#: a variant of it rather than a different system needing its own case.
MAX_DISAGREEMENT_RATE = 0.5


class Standing(StrEnum):
    """What a shadow has earned."""

    #: Better where they differ, no regression, enough evidence.
    READY = "ready"
    #: Better on average and it broke something that worked.
    REGRESSES = "regresses"
    #: Not better where they differ.
    NO_BETTER = "no_better"
    #: The same design under another name.
    IDENTICAL = "identical"
    #: So different it is not a variant of the live design.
    DIVERGENT = "divergent"
    #: Not tried enough to say.
    UNTRIED = "untried"

    @property
    def may_migrate(self) -> bool:
        return self is Standing.READY


@dataclass(frozen=True)
class Trial:
    """One input, both answers, and how each scored."""

    key: str
    agreed: bool
    live_score: float
    shadow_score: float
    at: float = field(default_factory=time.time)

    @property
    def regression(self) -> bool:
        """The live one got it right and the shadow did not."""
        return self.live_score > self.shadow_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "agreed": self.agreed,
            "live_score": round(self.live_score, 4),
            "shadow_score": round(self.shadow_score, 4),
            "regression": self.regression,
        }


@dataclass(frozen=True)
class MigrationProof:
    """What is known about whether switching would be safe."""

    standing: Standing
    trials: int
    disagreements: int
    regressions: tuple[str, ...]
    mean_gain: float
    contracts_held: bool
    because: str

    @property
    def may_migrate(self) -> bool:
        return self.standing.may_migrate

    def to_dict(self) -> dict[str, Any]:
        return {
            "standing": str(self.standing),
            "may_migrate": self.may_migrate,
            "trials": self.trials,
            "disagreements": self.disagreements,
            "regressions": list(self.regressions),
            "mean_gain": round(self.mean_gain, 4),
            "contracts_held": self.contracts_held,
            "because": self.because,
        }


class Shadow:
    """An alternative design, running where a mistake costs nothing."""

    def __init__(
        self,
        name: str,
        live: Callable[[Any], Any],
        candidate: Callable[[Any], Any],
        *,
        agree: Callable[[Any, Any], bool] | None = None,
        grade: Callable[[Any, Any], float] | None = None,
    ) -> None:
        self.name = str(name)
        self._live = live
        self._candidate = candidate
        self._agree = agree or (lambda a, b: a == b)
        # Without a grader there is no way to say which answer was better, so
        # every disagreement is a tie and the shadow can never be READY. That
        # is the correct behaviour, not a missing feature: a design nobody can
        # score is a design nobody should switch to.
        self._grade = grade
        self._trials: list[Trial] = []
        self._errors = 0

    def run(self, value: Any, *, key: str = "", expected: Any = None) -> Any:
        """Answer with the live design, and record what the shadow would say.

        Returns the live answer. There is no argument that changes that: a
        shadow reachable by a flag is a feature flag, and those get flipped.
        """
        answer = self._live(value)
        try:
            other = self._candidate(value)
        except Exception as exc:  # noqa: BLE001 - see below
            # Every exception, deliberately. A narrow list is the right default
            # everywhere else in this codebase and is wrong here: the shadow's
            # whole purpose is to run where a mistake costs nothing, and a
            # mistake it can make that this does not catch costs the live
            # answer. The first version listed five exception types and a
            # ZeroDivisionError from the candidate took the caller down with
            # it, which is the one thing a shadow must never do.
            self._errors += 1
            logger.debug("Shadow %s raised on %s: %s", self.name, key, exc)
            if self._errors == 1:
                # The first is news. A shadow that fails on every input would
                # otherwise open an incident per turn, which buries the one
                # that mattered under its own repetitions.
                record_degradation(
                    "promotion.shadow",
                    exc,
                    action=f"{self.name} raised; live answer unaffected",
                )
            return answer

        agreed = bool(self._agree(answer, other))
        live_score = shadow_score = 0.0
        if self._grade is not None and expected is not None:
            live_score = float(self._grade(answer, expected))
            shadow_score = float(self._grade(other, expected))
        self._trials.append(
            Trial(
                key=str(key) or f"trial-{len(self._trials)}",
                agreed=agreed,
                live_score=live_score,
                shadow_score=shadow_score,
            )
        )
        return answer

    @property
    def trials(self) -> tuple[Trial, ...]:
        return tuple(self._trials)

    @property
    def errors(self) -> int:
        return self._errors

    def proof(self, *, contracts_held: bool = True) -> MigrationProof:
        """Whether switching to this design would be safe."""
        trials = self._trials
        disagreements = [t for t in trials if not t.agreed]
        regressions = tuple(sorted(t.key for t in trials if t.regression))
        gains = [t.shadow_score - t.live_score for t in disagreements]
        mean_gain = sum(gains) / len(gains) if gains else 0.0

        if len(trials) < MIN_TRIALS:
            return self._proof(
                Standing.UNTRIED, trials, disagreements, regressions, mean_gain,
                contracts_held,
                f"{len(trials)} trials; {MIN_TRIALS} are needed before a "
                "comparison means anything",
            )
        if len(disagreements) < MIN_DISAGREEMENTS:
            return self._proof(
                Standing.IDENTICAL, trials, disagreements, regressions, mean_gain,
                contracts_held,
                f"{len(disagreements)} disagreements in {len(trials)} trials; "
                "this is the live design under another name and migrating to "
                "it buys nothing",
            )
        if len(disagreements) / len(trials) > MAX_DISAGREEMENT_RATE:
            return self._proof(
                Standing.DIVERGENT, trials, disagreements, regressions, mean_gain,
                contracts_held,
                f"they differ on {len(disagreements) / len(trials):.0%} of "
                "inputs; this is not a variant of the live design and needs "
                "its own case rather than a migration",
            )
        if not contracts_held:
            return self._proof(
                Standing.REGRESSES, trials, disagreements, regressions, mean_gain,
                contracts_held,
                "the shadow does not satisfy the behavioural contracts the "
                "live design does",
            )
        if regressions:
            return self._proof(
                Standing.REGRESSES, trials, disagreements, regressions, mean_gain,
                contracts_held,
                f"better by {mean_gain:+.3f} on average and it broke "
                f"{len(regressions)} case(s) that worked. An average is a poor "
                "summary of a change: nine improvements and one destruction "
                "has a good mean",
            )
        if mean_gain <= 0.0:
            return self._proof(
                Standing.NO_BETTER, trials, disagreements, regressions, mean_gain,
                contracts_held,
                f"where they differ the shadow is {mean_gain:+.3f}; it is a "
                "different design and not a better one",
            )
        return self._proof(
            Standing.READY, trials, disagreements, regressions, mean_gain,
            contracts_held,
            f"{len(disagreements)} disagreements over {len(trials)} trials, "
            f"better by {mean_gain:+.3f} where they differ, nothing that "
            "worked broken, contracts held",
        )

    @staticmethod
    def _proof(
        standing: Standing,
        trials: Sequence[Trial],
        disagreements: Sequence[Trial],
        regressions: tuple[str, ...],
        mean_gain: float,
        contracts_held: bool,
        because: str,
    ) -> MigrationProof:
        return MigrationProof(
            standing=standing,
            trials=len(trials),
            disagreements=len(disagreements),
            regressions=regressions,
            mean_gain=mean_gain,
            contracts_held=contracts_held,
            because=because,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trials": len(self._trials),
            "disagreements": sum(1 for t in self._trials if not t.agreed),
            "errors": self._errors,
        }


__all__ = [
    "MAX_DISAGREEMENT_RATE",
    "MIN_DISAGREEMENTS",
    "MIN_TRIALS",
    "MigrationProof",
    "Shadow",
    "Standing",
    "Trial",
]
