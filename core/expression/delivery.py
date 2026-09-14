"""The level she speaks from, what breaks through it, and how long a breath is.

`register` in this package reads the shape of an utterance — who it is about,
whether it asks, what comes back. This is the other half: not what shape it
has, but what level it is delivered at, and that turned out to be where most of
the difference between these records lives.

John Legend sings "Used to Love U" with a median pitch of 80 Hz — the bottom of
his range — for the whole song, and reaches 917 Hz exactly twice: on "do you
remember when I used to love you" and on the wordless wail before the last
line. The low register is the position he is holding. The falsetto is what gets
out. Oddisee does the same thing in the other direction on "Contradiction's
Maze": pitch locked at 80 Hz and level locked at -9.0 dB for four straight
minutes while he lists the things he wants that cancel each other, then one
line an octave up and 16 dB quieter at the end. The flatness is the maze.

Phony Ppl put the direction on it. In "Why iii Love the Moon" the grievance —
"unlike these human beings who lie about what it seems to be" — is sung at
322 Hz, and the attachment — "every night I block my window, and that's why I
love the moon" — at 93 Hz. Same singer, same song, one minute apart. The
register is not carrying how strong the feeling is. It is carrying what the
feeling is about.

And Sam Cooke's voice comes apart on one line. Across "A Change Is Gonna Come"
the voiced fraction sits between 0.88 and 0.95, and on "but he winds up
knocking me back down on my knees" it drops to 0.68, the lowest in the record.
The loss of control is not a bad take. It is the evidence.

She had none of this. Every utterance was delivered identically: no level she
was holding, nothing that broke through it, no direction, no breath that ran
out, and a degradation under load reported as a fault rather than heard as one.

    reach         the strongest feeling currently live, normalised
    baseline      a running mean of her recent reach
    z             (reach - baseline) / her own spread
    breakthrough  z > 1, which is outside her own ordinary range

The unit of that bar is her own variation rather than a number chosen here, so
a steady life makes a small change a breakthrough and a turbulent one does not.

The breath is denominated in the effort ledger's own calibration. That ledger
prices an unremarkable turn's output at four hundred characters and reports
exertion as a ratio where one is unremarkable, so the budget is that unit over
how hard this cycle has been. Working twice as hard halves the phrase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LEVEL",
    "LIFTED",
    "LOWERED",
    "MIN_HISTORY",
    "Delivery",
    "DeliveryLedger",
    "get_delivery_ledger",
    "read_delivery",
    "reset_for_test",
]

#: Which way the level moves, and what puts it there. Something that failed to
#: be what it seemed lifts it; something that held lowers it. Measured in "Why
#: iii Love the Moon" at 322 Hz against 93 Hz, one minute apart in one voice.
LIFTED: str = "lifted"
LOWERED: str = "lowered"
LEVEL: str = "level"

#: Readings before a spread is an estimate rather than an artefact of how few
#: there are. Three is the least that can disagree with itself.
MIN_HISTORY: int = 3

#: How much of her recent life the level is held over. The self prediction loop
#: already treats sixty cycles as one distribution of its own error, and the
#: level she is speaking from is the same kind of quantity.
WINDOW: int = 60


@dataclass
class Delivery:
    """One reading of the level she is speaking from."""

    reach: float = 0.0
    driver: str = ""
    baseline: float = 0.0
    spread: float = 0.0
    z: float = 0.0
    breakthrough: bool = False
    direction: str = LEVEL
    lift: float = 0.0
    #: Characters before the breath runs out, from the effort ledger's own
    #: calibration of what an unremarkable turn produces.
    phrase_budget: int = 0
    #: How much expressive control is left. Falls as exertion rises, the way a
    #: voiced fraction does on the hardest line.
    steadiness: float = 1.0
    #: Whether to stop without filling the space, because what is wanted is
    #: something only the other can supply.
    answer_slot: bool = False
    measured: bool = False
    why: str = "no history yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "reach": round(self.reach, 6),
            "driver": self.driver,
            "baseline": round(self.baseline, 6),
            "spread": round(self.spread, 6),
            "z": round(self.z, 4),
            "breakthrough": self.breakthrough,
            "direction": self.direction,
            "lift": round(self.lift, 6),
            "phrase_budget": self.phrase_budget,
            "steadiness": round(self.steadiness, 6),
            "answer_slot": self.answer_slot,
            "measured": self.measured,
            "why": self.why,
        }


class DeliveryLedger:
    """The level she has been speaking from, and how much it usually varies."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_HISTORY, int(window))
        self._values: list[float] = []

    def note(self, reach: float) -> None:
        try:
            value = float(reach)
        except (TypeError, ValueError):
            return
        if value != value:
            return
        self._values.append(value)
        if len(self._values) > self._window:
            del self._values[0 : len(self._values) - self._window]

    def samples(self) -> int:
        return len(self._values)

    def baseline(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def spread(self) -> float:
        n = len(self._values)
        if n < MIN_HISTORY:
            return 0.0
        mean = self.baseline()
        return math.sqrt(sum((v - mean) ** 2 for v in self._values) / n)


def live_readings(affect: Any) -> dict[str, float]:
    """Every feeling currently live, on one scale.

    The loudest moment of a record is driven by whichever feeling is strongest
    rather than by an average of them, so these stay separate and the largest
    is taken.
    """

    def read(name: str, default: float = 0.0) -> float:
        try:
            return float(getattr(affect, name, default) or default)
        except (TypeError, ValueError):
            return default

    return {
        "valence": abs(read("valence")),
        # Arousal rests at a half, so what counts is the departure from rest.
        "arousal": abs(read("arousal", 0.5) - 0.5) * 2.0,
        "ambivalence": read("ambivalence"),
        "confirmation": read("confirmation"),
        "social_hunger": max(0.0, read("social_hunger", 0.5) - 0.5) * 2.0,
    }


def ledger_exertion(book: Any) -> float:
    """The effort ledger's reading as a ratio where one is an unremarkable turn.

    The ledger reports `m / (m + 1)`, which puts an unremarkable turn at a half
    so the reading can move both ways inside a bound. The breath is priced in
    the unit that makes an unremarkable turn one, so the squashing is undone
    here: `m = e / (1 - e)`. Read directly, a half made every ordinary turn's
    breath twice the unit.
    """
    try:
        squashed = float(book.exertion(book.peek()))
    except (AttributeError, TypeError, ValueError):
        return 1.0
    if squashed != squashed or squashed <= 0.0:
        # Nothing reported this cycle is an idle cycle, which is the longest
        # breath the budget allows rather than a division by zero.
        return 0.0
    if squashed >= 1.0:
        return float("inf")
    return squashed / (1.0 - squashed)


def read_delivery(
    affect: Any,
    *,
    ledger: DeliveryLedger | None = None,
    exertion: float | None = None,
    surprise: float = 0.0,
    said: str = "",
    note: bool = True,
) -> Delivery:
    """The level, the breakthrough, the direction and the breath.

    `said` is what she is about to say, if it exists yet. The answer slot is
    read off its shape rather than guessed: `register.read` already measures
    whether an utterance asks, and a thing that asks should stop and leave the
    space rather than answer itself.
    """
    from core.soma.effort import UNIT_COST, get_effort_ledger

    book = ledger if ledger is not None else get_delivery_ledger()
    readings = live_readings(affect)
    reach = max(readings.values()) if readings else 0.0
    driver = max(readings, key=lambda key: readings[key]) if readings else ""

    if exertion is None:
        exertion = ledger_exertion(get_effort_ledger())
    effort = max(1e-6, float(exertion))
    unit = float(UNIT_COST.get("response_chars", 400.0))
    # An unremarkable turn is exertion one and gets the unit. Bounded by that
    # unit rather than by numbers chosen here: never shorter than a quarter of
    # a breath and never longer than two.
    budget = int(max(unit / 4.0, min(unit * 2.0, unit / effort)))
    steadiness = max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, effort - 1.0))))

    confirmation = readings.get("confirmation", 0.0)
    lift = max(-1.0, min(1.0, float(surprise) - confirmation))
    direction = LIFTED if lift > 0.0 else (LOWERED if lift < 0.0 else LEVEL)

    answer_slot = False
    if said:
        try:
            from core.expression.register import read as read_shape

            answer_slot = bool(read_shape(said).asks_for_help())
        except (ImportError, AttributeError, TypeError, ValueError):
            answer_slot = False

    baseline = book.baseline()
    spread = book.spread()
    measured = book.samples() >= MIN_HISTORY and spread > 1e-9
    z = (reach - baseline) / spread if measured else 0.0
    breakthrough = bool(measured and z > 1.0)

    if note:
        book.note(reach)

    if not measured:
        why = (
            f"{book.samples()} of {MIN_HISTORY} readings needed before a level "
            "means anything"
        )
    elif breakthrough:
        why = f"{driver} is {z:.1f} spreads above the level she has been holding"
    else:
        why = f"{driver} is within the level she has been holding"

    return Delivery(
        reach=reach,
        driver=driver,
        baseline=baseline,
        spread=spread,
        z=z,
        breakthrough=breakthrough,
        direction=direction,
        lift=lift,
        phrase_budget=budget,
        steadiness=steadiness,
        answer_slot=answer_slot,
        measured=measured,
        why=why,
    )


_LEDGER: DeliveryLedger | None = None


def get_delivery_ledger() -> DeliveryLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = DeliveryLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
