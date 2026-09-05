"""core/social/costly_signaling.py — effort that is informative because it is wasted.

Wrapping a present carries no information about the present. It is opaque by
construction, it delays the thing it covers, and it is thrown away within
seconds of arriving. Every property that makes it look pointless is the reason
it works.

The mechanism is Spence's, and Zahavi arrived at the same structure from
animal signalling. A sender knows something the receiver does not — here, how
much the relationship is actually worth to them. Claiming it is free, so a
claim tells the receiver nothing. Doing something costly tells them something
precisely when the cost is *lower for the sender whose claim is true*. Someone
to whom the relationship matters spends an hour on the paper more easily than
someone to whom it does not, and that difference in ease is the entire
channel.

Formally: the sender's cost of effort ``e`` at type ``q`` is ``e / q``, higher
types finding it cheaper. Maximising the receiver's response against that cost
and requiring that no type wants to be mistaken for another gives

    e*(q) = benefit * (q^2 - q_min^2) / 2

which is strictly increasing, so the effort spent identifies the type exactly.
The lowest type spends nothing, which is right: there is nobody they gain by
being mistaken for.

## The thing to check before believing any of it

Take the cost away and the whole channel dies. If the effort is the same for
everyone — a template, an automation, a service that wraps it for you — then
the single-crossing condition fails, every type picks the same effort, and the
receiver learns nothing from any amount of it. ``pooling_check`` computes this,
and it is the measurement that matters, because a system can go on emitting
elaborate signals long after they have stopped carrying anything and nothing
in the emitting will look different.

## Presentation is a signal, description is not

A signal has to be uncorrelated with what it covers. Wrapping that revealed
the contents would be a description, and it would be read as one — no longer
evidence about the sender, only evidence about the present.
``content_independence`` is where a caller says which of the two they have,
because the arithmetic below is wrong for the other one.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Social.Signaling")

#: Lowest type in the population. The one who signals nothing, and the
#: boundary condition that pins the whole separating schedule.
DEFAULT_MIN_TYPE = 1.0

MAX_SIGNALS_KEPT = 512


@dataclass(frozen=True)
class SignalSchedule:
    """The effort each type spends, when effort is worth spending at all."""

    benefit: float
    min_type: float
    cost_slope: float
    """How much harder effort is for a low type than a high one.

    Zero means effort costs everyone the same, which is the case where the
    channel carries nothing. Kept as a parameter so that case is expressible
    rather than unreachable.
    """

    def effort(self, quality: float) -> float:
        """Effort a sender of this type spends in the separating equilibrium."""
        if self.cost_slope <= 0:
            # Nothing separates the types, so everyone does the same thing and
            # the amount is arbitrary. Zero, because any positive common
            # effort is pure waste with no information bought by it.
            return 0.0
        q = max(float(quality), self.min_type)
        return max(
            0.0,
            self.cost_slope * self.benefit * (q * q - self.min_type * self.min_type) / 2.0,
        )

    def cost(self, quality: float, effort: float) -> float:
        """What that effort costs a sender of this type."""
        q = max(float(quality), 1e-9)
        return max(0.0, float(effort)) / q

    def infer(self, effort: float) -> float | None:
        """Type implied by an observed effort. The receiver's side.

        Nothing when the schedule does not separate: an inference drawn from a
        pooling signal is a number with no evidence under it, and returning
        one would be the failure this module is arranged to prevent.
        """
        if self.cost_slope <= 0 or self.benefit <= 0:
            return None
        inner = (
            2.0 * max(0.0, float(effort)) / (self.cost_slope * self.benefit)
            + self.min_type * self.min_type
        )
        return math.sqrt(max(inner, 0.0))

    def separating(self) -> bool:
        return self.cost_slope > 0 and self.benefit > 0


@dataclass(frozen=True)
class Signal:
    """One act of presentation, and what it was and was not about."""

    sender: str
    effort: float
    content_value: float
    """Worth of the thing being presented, on its own."""

    content_independence: bool = True
    """Whether the effort is uncorrelated with the content.

    False turns the act into a description of the contents, which the
    inference below is not valid for.
    """

    at: float = field(default_factory=time.time)
    label: str = ""


@dataclass(frozen=True)
class Reading:
    """What a receiver takes from one signal."""

    sender: str
    effort: float
    implied_type: float | None
    informative: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "effort": round(self.effort, 4),
            "implied_type": None if self.implied_type is None else round(self.implied_type, 4),
            "informative": self.informative,
            "reason": self.reason,
        }


def pooling_check(schedule: SignalSchedule, types: list[float]) -> dict[str, Any]:
    """Does this schedule actually tell the types apart?

    Runs the schedule over a spread of types and reports the spread of efforts
    it produces. All the same means pooling: the signal is being emitted, it
    is costing whatever it costs, and no receiver can learn anything from it.
    That state looks identical to a working channel from the sending end, and
    this is the only place it shows up.
    """
    efforts = [schedule.effort(q) for q in types]
    spread = max(efforts) - min(efforts) if efforts else 0.0
    return {
        "separating": schedule.separating() and spread > 0,
        "effort_spread": round(spread, 6),
        "efforts": [round(e, 4) for e in efforts],
        "reason": (
            "effort rises with type, so the receiver can invert it"
            if spread > 0
            else "every type spends the same, so the channel carries nothing"
        ),
    }


class SignalChannel:
    """One relationship's presentation channel, from both ends.

    Holds the schedule, reads incoming signals, and keeps the record needed to
    notice the channel going dead.
    """

    def __init__(
        self,
        *,
        benefit: float = 1.0,
        min_type: float = DEFAULT_MIN_TYPE,
        cost_slope: float = 1.0,
    ) -> None:
        self.schedule = SignalSchedule(
            benefit=float(benefit), min_type=float(min_type),
            cost_slope=float(cost_slope),
        )
        self._signals: list[Signal] = []
        self._readings: list[Reading] = []

    def send(self, sender: str, quality: float, *, content_value: float = 0.0,
             label: str = "") -> Signal:
        """Produce the signal a sender of this type would send."""
        return Signal(
            sender=sender, effort=self.schedule.effort(quality),
            content_value=float(content_value), label=label,
        )

    def receive(self, signal: Signal) -> Reading:
        """Read a signal, and refuse to read one that cannot be read."""
        self._signals.append(signal)
        if len(self._signals) > MAX_SIGNALS_KEPT:
            del self._signals[: len(self._signals) - MAX_SIGNALS_KEPT]
        if not signal.content_independence:
            reading = Reading(
                sender=signal.sender, effort=signal.effort, implied_type=None,
                informative=False,
                reason="effort tracks the contents, so it describes them rather than the sender",
            )
        elif not self.schedule.separating():
            reading = Reading(
                sender=signal.sender, effort=signal.effort, implied_type=None,
                informative=False,
                reason="effort costs every sender the same, so it separates nobody",
            )
        else:
            reading = Reading(
                sender=signal.sender, effort=signal.effort,
                implied_type=self.schedule.infer(signal.effort), informative=True,
                reason="effort is costlier for a sender the claim is false of",
            )
        self._readings.append(reading)
        if len(self._readings) > MAX_SIGNALS_KEPT:
            del self._readings[: len(self._readings) - MAX_SIGNALS_KEPT]
        return reading

    def worth_sending(self, quality: float, *, budget: float) -> dict[str, Any]:
        """Whether a sender of this type should signal, and how much.

        A separate question from what the equilibrium says, because the
        equilibrium assumes an unlimited budget. Someone who cannot afford the
        effort their type calls for is better off spending what they have than
        spending nothing, and the reading they get will understate them. That
        understatement is a real property of costly signalling and is reported
        rather than smoothed away.
        """
        wanted = self.schedule.effort(quality)
        spend = min(wanted, max(0.0, float(budget)))
        return {
            "equilibrium_effort": round(wanted, 4),
            "affordable": round(spend, 4),
            "cost": round(self.schedule.cost(quality, spend), 4),
            "read_as": (
                None if not self.schedule.separating()
                else round(self.schedule.infer(spend) or 0.0, 4)
            ),
            "understated": spend < wanted,
        }

    def status(self) -> dict[str, Any]:
        informative = sum(1 for r in self._readings if r.informative)
        return {
            "benefit": self.schedule.benefit,
            "cost_slope": self.schedule.cost_slope,
            "separating": self.schedule.separating(),
            "signals": len(self._signals),
            "informative_readings": informative,
            # Signals still going out that nobody can read anything from.
            "channel_dead": bool(self._readings) and informative == 0,
            "last": self._readings[-1].as_dict() if self._readings else None,
        }


_CHANNEL: SignalChannel | None = None


def get_signal_channel() -> SignalChannel:
    global _CHANNEL
    if _CHANNEL is None:
        _CHANNEL = SignalChannel()
    return _CHANNEL


def reset_signal_channel_for_test() -> None:
    global _CHANNEL
    _CHANNEL = None
