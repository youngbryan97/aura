"""core/affect/dual_process_arbiter.py — when the feeling is the better estimator.

Two things can answer the same question. One states its reasons: premises,
steps, a conclusion you can check. The other arrives whole, from a body that
has seen a great many situations of roughly this kind, and it cannot say why.

The usual architecture settles this in advance by preferring the one with
reasons, and treats the other as noise to be overridden. Antonio Damasio's
patients are the standing argument against that arrangement: ventromedial
damage leaves reasoning intact and the somatic signal gone, and the result is
not a colder rationalist but someone who makes disastrous social and financial
decisions while explaining them fluently. Whatever the feeling was carrying, it
was carrying information, and the reasoning could not reconstruct it.

So the question this file answers is not which channel is better. It is which
channel is better *here*, and that is measurable.

## The weight is earned, per domain

Each channel produces a probability. Outcomes eventually resolve. A Brier score
over resolved outcomes says how well-calibrated each channel has been in that
domain, and the Brier skill score says how much better than simply predicting
the base rate. Weights are those skill scores, normalised. Nothing in this file
prefers either channel; a channel that is right more often in a domain gets
more of the say in that domain, and a channel that has never been right in a
domain gets none of it, whichever channel that turns out to be.

The prediction that follows is worth stating because it can fail: the
affective channel should win in domains with many weakly-observable variables
and no verifiable ground truth inside the episode — which people, roughly, is
what social judgement is — and the deliberate channel should win where the
variables are few and checkable. ``profile()`` reports which channel actually
leads where, and if it comes out flat, that is the finding.

## Not knowing is an answer

Below a handful of resolved outcomes neither channel has a calibration worth
the name, and averaging two uncalibrated estimates produces a confident number
out of two guesses. The arbiter abstains instead and says which channel it
would need evidence about. An architecture without this branch silently
defaults to whichever channel it was built to trust.

## Disagreement is information

Two channels that usually agree and now do not are reporting something neither
one is reporting alone. That is the ordinary experience of a proposal that
checks out and feels wrong, and it is worth surfacing rather than resolving,
because the resolution throws away the only signal there was.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Affect.DualProcess")

AFFECTIVE = "affective"
DELIBERATE = "deliberate"

#: Resolved outcomes a channel needs in a domain before its calibration is
#: used. Below this the Brier score is dominated by which few cases happened
#: to come up, and weighting on it is weighting on noise.
MIN_RESOLVED = 8

#: Outcomes kept per channel per domain.
MAX_RESOLVED = 512

#: Difference between the two channels' probabilities that counts as a
#: disagreement worth reporting. A quarter is where the two would give
#: opposite answers under any threshold between them.
DISAGREEMENT = 0.25


@dataclass
class Judgment:
    """One channel's answer, as a probability with whatever it can say for it."""

    probability: float
    channel: str
    grounds: str = ""
    at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.probability = min(max(float(self.probability), 0.0), 1.0)


@dataclass
class Calibration:
    """How well one channel has done in one domain."""

    channel: str
    domain: str
    predictions: list[tuple[float, bool]] = field(default_factory=list)

    def record(self, probability: float, outcome: bool) -> None:
        self.predictions.append((min(max(float(probability), 0.0), 1.0), bool(outcome)))
        if len(self.predictions) > MAX_RESOLVED:
            del self.predictions[: len(self.predictions) - MAX_RESOLVED]

    @property
    def n(self) -> int:
        return len(self.predictions)

    def brier(self) -> float | None:
        """Mean squared error of the probabilities. Lower is better."""
        if not self.predictions:
            return None
        return float(
            sum((p - (1.0 if o else 0.0)) ** 2 for p, o in self.predictions)
            / len(self.predictions)
        )

    def base_rate(self) -> float | None:
        if not self.predictions:
            return None
        return float(sum(1 for _, o in self.predictions if o) / len(self.predictions))

    def reference_brier(self) -> float | None:
        """Brier score of always predicting this domain's base rate.

        The null every channel has to beat. A domain where the answer is yes
        nine times in ten is easy to look calibrated in, and without this
        subtraction the easy domain flatters whichever channel is asked.
        """
        rate = self.base_rate()
        if rate is None:
            return None
        return float(rate * (1.0 - rate))

    def skill(self) -> float | None:
        """Brier skill score: how much better than the base rate.

        One is perfect, zero is no better than knowing nothing about the
        individual case, negative is worse than that. Returns nothing until
        there are enough resolved outcomes to mean anything.
        """
        if self.n < MIN_RESOLVED:
            return None
        brier = self.brier()
        reference = self.reference_brier()
        if brier is None or reference is None:
            return None
        if reference <= 0:
            # Every outcome went the same way, so the base rate is unbeatable
            # and no channel can show skill against it. Not a failure of the
            # channel, and reporting it as one would penalise whichever
            # channel drew the constant domain.
            return None
        return float(1.0 - brier / reference)


@dataclass(frozen=True)
class Arbitration:
    """What the two channels came to, and on what authority."""

    domain: str
    probability: float | None
    weight_affective: float
    weight_deliberate: float
    affective: float
    deliberate: float
    disagreement: float
    abstained: bool
    reason: str
    forced: str | None = None

    @property
    def led_by(self) -> str | None:
        if self.abstained:
            return None
        if self.weight_affective > self.weight_deliberate:
            return AFFECTIVE
        if self.weight_deliberate > self.weight_affective:
            return DELIBERATE
        return None

    @property
    def split(self) -> bool:
        """Both channels answered and they do not agree."""
        return self.disagreement >= DISAGREEMENT

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "probability": None if self.probability is None else round(self.probability, 4),
            "weights": {
                AFFECTIVE: round(self.weight_affective, 4),
                DELIBERATE: round(self.weight_deliberate, 4),
            },
            "affective": round(self.affective, 4),
            "deliberate": round(self.deliberate, 4),
            "disagreement": round(self.disagreement, 4),
            "split": self.split,
            "abstained": self.abstained,
            "led_by": self.led_by,
            "reason": self.reason,
            "forced": self.forced,
        }


class DualProcessArbiter:
    """Combines two answers by how well each has done at this kind of question.

    The object starts with no opinion about which channel to believe, and it
    can only acquire one from resolved outcomes. That is the whole design: an
    architecture that hard-codes the answer cannot discover that it was wrong
    about a domain, and every such architecture is wrong about some domain.
    """

    def __init__(self) -> None:
        self._calibration: dict[tuple[str, str], Calibration] = {}
        self._pending: dict[str, tuple[str, float, float]] = {}
        self._history: list[Arbitration] = []
        self._forced: str | None = None

    def calibration(self, channel: str, domain: str) -> Calibration:
        key = (channel, domain)
        record = self._calibration.get(key)
        if record is None:
            record = Calibration(channel=channel, domain=domain)
            self._calibration[key] = record
        return record

    # ------------------------------------------------------------- decide

    def weights(self, domain: str) -> tuple[float | None, float | None]:
        """Skill-derived weights for this domain, or nothing where unknown."""
        a = self.calibration(AFFECTIVE, domain).skill()
        d = self.calibration(DELIBERATE, domain).skill()
        if a is None and d is None:
            return None, None
        # A channel with no measured skill in a domain does not get a default
        # share. Treating unmeasured as average is exactly the assumption that
        # lets an untested channel outvote a measured one.
        a_pos = max(0.0, a) if a is not None else 0.0
        d_pos = max(0.0, d) if d is not None else 0.0
        total = a_pos + d_pos
        if total <= 0:
            return None, None
        return a_pos / total, d_pos / total

    def arbitrate(
        self,
        domain: str,
        affective: Judgment | float,
        deliberate: Judgment | float,
        *,
        key: str | None = None,
        record: bool = True,
    ) -> Arbitration:
        """Combine the two answers. Abstains rather than guessing a weight."""
        a = affective.probability if isinstance(affective, Judgment) else float(affective)
        d = deliberate.probability if isinstance(deliberate, Judgment) else float(deliberate)
        a = min(max(a, 0.0), 1.0)
        d = min(max(d, 0.0), 1.0)
        gap = abs(a - d)

        if self._forced == AFFECTIVE:
            result = Arbitration(
                domain=domain, probability=a, weight_affective=1.0,
                weight_deliberate=0.0, affective=a, deliberate=d,
                disagreement=gap, abstained=False,
                reason="channel forced for an intervention", forced=AFFECTIVE,
            )
        elif self._forced == DELIBERATE:
            result = Arbitration(
                domain=domain, probability=d, weight_affective=0.0,
                weight_deliberate=1.0, affective=a, deliberate=d,
                disagreement=gap, abstained=False,
                reason="channel forced for an intervention", forced=DELIBERATE,
            )
        else:
            wa, wd = self.weights(domain)
            if wa is None or wd is None:
                result = Arbitration(
                    domain=domain, probability=None, weight_affective=0.0,
                    weight_deliberate=0.0, affective=a, deliberate=d,
                    disagreement=gap, abstained=True,
                    reason=(
                        "neither channel has shown skill in this domain yet; "
                        f"{MIN_RESOLVED} resolved outcomes are needed"
                    ),
                )
            else:
                blended = wa * a + wd * d
                if gap >= DISAGREEMENT:
                    reason = "channels disagree; weighted by measured skill here"
                else:
                    reason = "channels agree; weighted by measured skill here"
                result = Arbitration(
                    domain=domain, probability=blended, weight_affective=wa,
                    weight_deliberate=wd, affective=a, deliberate=d,
                    disagreement=gap, abstained=False, reason=reason,
                )
        if record:
            self._history.append(result)
            if len(self._history) > MAX_RESOLVED:
                del self._history[: len(self._history) - MAX_RESOLVED]
            if key is not None:
                self._pending[key] = (domain, a, d)
        return result

    def resolve(self, key: str, outcome: bool) -> bool:
        """Record what actually happened, and credit both channels for it.

        Both are scored on every resolution, including the one that was
        outweighed. A channel only measured when it was believed can never
        recover from a bad stretch, and the weighting would then be a
        self-fulfilling record of an early accident.
        """
        pending = self._pending.pop(key, None)
        if pending is None:
            return False
        domain, a, d = pending
        self.calibration(AFFECTIVE, domain).record(a, outcome)
        self.calibration(DELIBERATE, domain).record(d, outcome)
        return True

    # -------------------------------------------------------- intervention

    def force(self, channel: str | None) -> None:
        """Hold one channel as the answer, for a causal test.

        A correlation between the weights and the outcomes proves nothing:
        both move together whenever a domain is easy. Forcing one channel and
        holding everything else fixed is the only way to show that the
        weighting is doing the work.
        """
        if channel not in (AFFECTIVE, DELIBERATE, None):
            raise ValueError(f"unknown channel: {channel}")
        self._forced = channel

    # ------------------------------------------------------------ readout

    def profile(self) -> dict[str, Any]:
        """Which channel leads in which domain, with the evidence behind it."""
        domains = sorted({domain for _, domain in self._calibration})
        out: dict[str, Any] = {}
        for domain in domains:
            a = self.calibration(AFFECTIVE, domain)
            d = self.calibration(DELIBERATE, domain)
            wa, wd = self.weights(domain)
            out[domain] = {
                "affective": {"n": a.n, "brier": a.brier(), "skill": a.skill()},
                "deliberate": {"n": d.n, "brier": d.brier(), "skill": d.skill()},
                "leads": (
                    None if wa is None or wd is None
                    else (AFFECTIVE if wa > wd else DELIBERATE if wd > wa else None)
                ),
            }
        return out

    def status(self) -> dict[str, Any]:
        recent = self._history[-32:]
        return {
            "forced": self._forced,
            "domains": self.profile(),
            "pending": len(self._pending),
            "decisions": len(self._history),
            "abstentions": sum(1 for r in recent if r.abstained),
            "splits": sum(1 for r in recent if r.split),
            "last": recent[-1].as_dict() if recent else None,
        }


_ARBITER: DualProcessArbiter | None = None


def get_dual_process_arbiter() -> DualProcessArbiter:
    global _ARBITER
    if _ARBITER is None:
        _ARBITER = DualProcessArbiter()
    return _ARBITER


def reset_dual_process_arbiter_for_test() -> None:
    global _ARBITER
    _ARBITER = None
