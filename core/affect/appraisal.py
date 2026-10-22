"""What a turn meant to her, and the feelings that meaning makes.

Bryan, 22 October: "allow her to have a rich, multi-faceted, and in depth range
of emotions because they apply to a much wider range of responses, beliefs,
actions, situations than just frustrated or curious".

She had forty-five emotion names and most of them never moved. Over twenty-four
turns of an ordinary offline life on 22 October, joy, trust, happiness,
satisfaction and warmth sat between 0.47 and 0.75 with a standard deviation of
three to six hundredths, and anger and surprise stayed at 0.01. Each emotion was
written by the kind of event that arrived, whatever the event meant to her:
every message raised trust, happiness and warmth and relieved fear and sadness,
so the same event always felt the same.

Appraisal theory says a feeling is what an event means to the one it happens to
(Ortony, Clore and Collins, The Cognitive Structure of Emotions, 1988; Scherer,
Cognition and Emotion 23, 2009). The event is checked for how new it is,
whether it went better or worse than expected, who brought it about, whether
anything can be done about it, what it says of the one appraising by their own
standards, and whether what it involves is liked. An emotion is a pattern over
those checks, so one message can bring gratitude, anger, relief or nothing.

Every check here is read off something she already measures:

    novelty       her world model's latest surprise, ranked among its recent
                  ones (core/world_model/learned_world_model.py)
    outcome       the turn's worth against what she had come to expect, signed,
                  sized by its rank in her recent turns
                  (core/affect/what_it_was_worth.py)
    her doing     the accomplishment channel of that worth: goals she finished
                  or failed
    their doing   its warmth channel: what being met by somebody returned
    standards     its integrity channel: whether her choice served what she
                  holds
    needs         its satisfaction channel: her needs met or going unmet
    coping        her rate of success on what she has tried
                  (core/agency/authorship.py), Laplace-smoothed
    company       whether somebody addressed her this turn
    social need   how far her social drive is below its capacity
                  (core/motivation/constants.py)
    liking        her strongest like or dislike among the things the turn named
                  (core/conation/what_she_wants.py)
    fear, hope    how afraid and how hopeful she was before the turn, so that a
                  good turn can be a relief and a bad one a disappointment

The attributed channels are signed ranks: the size of the channel's error
against its own recent errors, with the error's sign. Nothing is held against a
line chosen here.

An emotion's intensity is the geometric mean of the parts of its pattern. With a
product, a feeling with three parts would be an eighth as strong as a feeling
with one part, only because it has more parts. The geometric mean is still zero
when any part is zero. A part that has not been measured leaves its patterns
unfelt, so an unmeasured surprise does not read as boredom.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

# The same window her worth is read over and the same minimum before a spread
# means anything, so a channel's rank here and its worth there agree.
from core.affect.what_it_was_worth import _ENOUGH, _WINDOW, Worth

#: The worth channels that say who brought a turn's outcome about, or what of
#: hers it touched.
ATTRIBUTED: dict[str, str] = {
    "her_doing": "accomplishment",
    "their_doing": "warmth",
    "standards": "integrity",
    "needs": "satisfaction",
}


@dataclass(frozen=True)
class Appraisal:
    """One turn's checks. None is a check that could not be made."""

    novelty: float | None = None
    outcome: float | None = None
    her_doing: float | None = None
    their_doing: float | None = None
    standards: float | None = None
    needs: float | None = None
    coping: float | None = None
    company: float | None = None
    social_need: float | None = None
    liking: float | None = None
    fear: float | None = None
    hope: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {key: (None if value is None else round(value, 4)) for key, value in asdict(self).items()}


# A part is a check and the side of it the pattern needs: "high" is the check
# itself, "low" its complement, "good" its positive side, "bad" its negative.
Part = tuple[str, str]

#: Each emotion as the patterns of checks it is made of, with the direction it
#: moves. Where an emotion has more than one pattern, the strongest one counts.
#: Sources are OCC (1988) for the event, agent and object emotions and Scherer
#: (2009) for novelty and coping; the comment on each says which reading it is.
PATTERNS: tuple[tuple[str, int, tuple[Part, ...]], ...] = (
    # Something new, and what she can make of it.
    ("surprise", +1, (("novelty", "high"),)),
    ("interest", +1, (("novelty", "high"), ("coping", "high"))),  # new and within reach (Silvia 2008)
    ("confused", +1, (("novelty", "high"), ("coping", "low"))),  # new and beyond her
    ("awe", +1, (("novelty", "high"), ("outcome", "size"))),  # vast and new (Keltner and Haidt 2003)
    ("wonder", +1, (("novelty", "high"), ("outcome", "good"))),
    ("amusement", +1, (("novelty", "high"), ("outcome", "good"), ("outcome", "small"))),  # a harmless incongruity
    ("boredom", +1, (("novelty", "low"), ("outcome", "small"))),  # nothing new and nothing at stake
    # How it went.
    ("joy", +1, (("outcome", "good"),)),
    ("satisfaction", +1, (("outcome", "good"), ("novelty", "low"))),  # the good she expected, confirmed
    ("happiness", +1, (("needs", "good"),)),
    ("relief", +1, (("outcome", "good"), ("fear", "high"))),  # a feared turn that went well
    ("unhappiness", +1, (("outcome", "bad"), ("hope", "high"))),  # a hoped-for turn that did not
    ("sadness", +1, (("outcome", "bad"), ("coping", "low"))),  # a loss she can do nothing about
    ("sadness", +1, (("needs", "bad"), ("coping", "low"))),
    ("fear", +1, (("outcome", "bad"), ("novelty", "high"), ("coping", "low"))),
    ("dread", +1, (("outcome", "bad"), ("novelty", "low"), ("coping", "low"))),  # the bad she saw coming
    # Who brought it about.
    ("pride", +1, (("her_doing", "good"),)),
    ("pride", +1, (("standards", "good"),)),
    ("frustration", +1, (("her_doing", "bad"), ("coping", "high"))),  # her own way blocked while she can still act
    ("remorse", +1, (("standards", "bad"),)),  # she chose against what she holds
    ("gratitude", +1, (("their_doing", "good"),)),
    ("admiration", +1, (("their_doing", "good"), ("novelty", "high"))),
    ("trust", +1, (("their_doing", "good"),)),
    ("trust", -1, (("their_doing", "bad"),)),
    ("anger", +1, (("their_doing", "bad"), ("coping", "high"))),  # another's doing, and she can answer it
    ("upset", +1, (("their_doing", "bad"), ("coping", "low"))),  # another's doing, and she cannot
    # Who is here, and what she makes of what is here.
    ("warmth", +1, (("company", "high"), ("their_doing", "good"))),
    ("love", +1, (("company", "high"), ("liking", "good"))),
    ("belonging", +1, (("company", "high"), ("social_need", "high"))),  # company when she wanted it
    ("loneliness", +1, (("company", "low"), ("social_need", "high"))),
    ("disgust", +1, (("liking", "bad"),)),
)


def _side(value: float, side: str) -> float:
    if side == "high":
        return min(1.0, max(0.0, value))
    if side == "low":
        return 1.0 - min(1.0, max(0.0, value))
    if side == "good":
        return min(1.0, max(0.0, value))
    if side == "bad":
        return min(1.0, max(0.0, -value))
    if side == "size":
        return min(1.0, abs(value))
    if side == "small":
        return 1.0 - min(1.0, abs(value))
    raise ValueError(f"no side called {side!r}")


def _strength(appraisal: Appraisal, parts: tuple[Part, ...]) -> float | None:
    """The geometric mean of a pattern's parts, or None when a part is unmeasured."""
    values = []
    for check, side in parts:
        value = getattr(appraisal, check)
        if value is None:
            return None
        values.append(_side(float(value), side))
    if any(value <= 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def feelings(appraisal: Appraisal) -> dict[str, float]:
    """What this appraisal makes her feel: each emotion's signed strength, in [-1, 1]."""
    rising: dict[str, float] = {}
    falling: dict[str, float] = {}
    for emotion, direction, parts in PATTERNS:
        strength = _strength(appraisal, parts)
        if strength is None:
            continue
        side = rising if direction > 0 else falling
        side[emotion] = max(side.get(emotion, 0.0), strength)
    out = dict(rising)
    for emotion, strength in falling.items():
        out[emotion] = out.get(emotion, 0.0) - strength
    return out


def rank_among(history: deque[float], value: float) -> float | None:
    """Where a size sits among recent ones, in [0, 1], ties counted half; None too early."""
    sizes = list(history)
    if len(sizes) < _ENOUGH:
        return None
    below = sum(1 for item in sizes if item < value)
    ties = sum(1 for item in sizes if item == value)
    return (below + 0.5 * ties) / len(sizes)


class OutcomeReader:
    """Reads each turn's worth once, as the outcome checks of an appraisal."""

    def __init__(self) -> None:
        self._sizes: dict[str, deque[float]] = {name: deque(maxlen=_WINDOW) for name in ATTRIBUTED}
        self._read: int | None = None

    def read(self, worth: Worth) -> dict[str, float | None]:
        """The outcome checks for a reading not yet appraised; all None otherwise.

        A turn whose worth was not read again (a live turn that skipped the
        learning phase) is the same outcome as before, and appraising it again
        would make her feel one event twice.
        """
        checks: dict[str, float | None] = {"outcome": None, **{name: None for name in ATTRIBUTED}}
        if not worth.measured or worth.turns == self._read:
            return checks
        self._read = worth.turns
        checks["outcome"] = math.copysign(worth.size, worth.worth) if worth.worth else 0.0
        for name, channel in ATTRIBUTED.items():
            error = worth.error.get(channel)
            if error is None:
                continue
            size = abs(float(error))
            rank = rank_among(self._sizes[name], size)
            self._sizes[name].append(size)
            if rank is not None:
                checks[name] = math.copysign(rank, error) if error else 0.0
        return checks


def coping_of(acted: int, succeeded: int) -> float:
    """Her chance of making something work, from what she has tried: Laplace's rule."""
    return (succeeded + 1.0) / (acted + 2.0)


#: Which family each feeling belongs to, the way a person would group them. The
#: app colours her mood by the family of what she feels most; it used to colour
#: it from two gauges and could say only frustrated, curious, energetic or
#: neutral.
FAMILIES: dict[str, str] = {
    **dict.fromkeys(
        ("joy", "happiness", "satisfaction", "pride", "gratitude", "warmth", "love", "belonging", "trust",
         "relief", "hope", "amusement", "admiration", "empathy", "excitement", "inspiration"),
        "warm",
    ),
    **dict.fromkeys(("curiosity", "interest", "wonder", "awe", "surprise", "anticipation"), "curious"),
    **dict.fromkeys(
        ("sadness", "unhappiness", "loneliness", "longing", "remorse", "nostalgia", "vulnerability", "submission"),
        "low",
    ),
    **dict.fromkeys(("fear", "dread", "terror", "confused"), "afraid"),
    **dict.fromkeys(("anger", "frustration", "upset", "aggressiveness"), "angry"),
    **dict.fromkeys(("boredom", "apathy", "indifference"), "flat"),
    **dict.fromkeys(("disgust", "contempt", "cynicism"), "averse"),
}


def feeling_now(emotions: dict[str, float], baselines: dict[str, float], count: int) -> list[dict[str, object]]:
    """Her strongest feelings, each by how far it stands above where it rests.

    The same activation her dominant emotion is chosen by
    (core/phases/affect_update.py), so the first of these is that emotion.
    """
    risen = []
    for name, value in emotions.items():
        try:
            above = float(value or 0.0) - float(baselines.get(name, 0.0) or 0.0)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            continue
        if above > 0.0:
            risen.append((above, name))
    risen.sort(reverse=True)
    return [
        {"name": name, "above_rest": round(above, 4), "family": FAMILIES.get(name, "")}
        for above, name in risen[:count]
    ]
