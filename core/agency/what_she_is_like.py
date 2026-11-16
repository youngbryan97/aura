"""What she is like, read from what she has chosen when nobody asked.

Asked live on 26 Sep to take a personality test, she said she had no stable
self-model to take it with: her responses were "generated per-context, not
from a fixed trait vector". She holds one. Her values are a weighted vector
she keeps and saves (truth, care, coherence, connection and the rest), and
every time she chooses what to do next on her own, the choice is scored by it
and written down with the options, what each served and what she took. Nothing
put that record in front of her, so she answered from what a language model
believes an AI is.

This reads it back as a person would describe themselves from a diary rather
than from a wish: the values she holds, what she actually chose most, how often
she took the option serving each value when one was offered against how often
chance alone would have, where holding and choosing disagree, and whether the
pattern held from the earlier half of the record to the later. A lean counts
only where it is rarer than chance allows, one time in twenty shared across
the values tested. Where the record is narrow or the pattern moved, that is
said too, because it is as much a fact about her as a steady trait would be.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "Portrait",
    "PortraitReader",
    "ValueLean",
    "asks_what_she_is_like",
    "get_portrait_reader",
    "portrait_of",
    "what_she_is_like",
    "what_she_is_like_block",
    "what_she_is_like_line",
]

#: How often a lean may be called one by bad luck, shared across the values.
_WRONG = 0.05


@dataclass(frozen=True, slots=True)
class ValueLean:
    """One value: how strongly she holds it, and how she chose when it was on offer."""

    value: str
    held: float
    offered: int
    chosen: int
    chance: float
    earlier: float | None
    later: float | None
    #: Whether she took it more (1), less (-1) or no differently (0) than chance.
    lean: int

    @property
    def rate(self) -> float:
        return self.chosen / self.offered if self.offered else 0.0

    @property
    def held_steady(self) -> bool | None:
        """Whether both halves of the record lean the same way. None where one half never offered it."""
        if self.earlier is None or self.later is None or not self.lean:
            return None
        return (self.earlier - self.chance) * self.lean > 0 and (self.later - self.chance) * self.lean > 0


@dataclass(frozen=True, slots=True)
class Portrait:
    """What the record of her own choices says about her."""

    choices: int
    since: float
    until: float
    values: tuple[ValueLean, ...]
    #: What she chose most, as (what, how many times).
    most_chosen: tuple[tuple[str, int], ...]
    #: How often her values overrode her strongest drive.
    over_impulse: int
    #: The share of choices taken by the single act chosen most, earlier and later.
    narrowest_earlier: float
    narrowest_later: float
    held_order: tuple[tuple[str, float], ...] = field(default=())
    #: Of the choices with more than one option, how many she took the option
    #: her values alone ranked first, and how many chance alone would give.
    values_foretold: int = 0
    foretold_by_chance: float = 0.0
    contested: int = 0

    @property
    def empty(self) -> bool:
        return self.choices == 0


def _two_sided(chosen: int, offered: int, chance: float) -> float:
    """The chance of a count at least this far from what chance gives, either way."""
    if offered <= 0 or not 0.0 < chance < 1.0:
        return 1.0

    from scipy.special import bdtr, bdtrc

    # The two binomial tails, read in one call each (scipy.special: bdtr is P(X <= k), bdtrc is P(X > k)). They were
    # summed term by term; at four thousand turns that cost her event loop 7.0 s on 16 November 2026 in
    # core/conation/what_she_cares_about.py, where the same sums were read.
    below = float(bdtr(chosen, offered, chance))
    above = float(bdtrc(chosen - 1, offered, chance)) if chosen > 0 else 1.0
    return min(1.0, 2.0 * min(below, above))


def _tally(records: Sequence[Mapping[str, Any]], value: str) -> tuple[int, int, float]:
    """Times an option serving ``value`` was on offer beside one that did not, times it was taken, and chance's share."""
    offered = chosen = 0
    expected = 0.0
    for record in records:
        features = record.get("option_features") or {}
        options = [str(one) for one in features]
        serving = [one for one in options if float((features.get(one) or {}).get(value, 0.0) or 0.0) > 0.0]
        if not serving or len(serving) == len(options):
            continue
        offered += 1
        expected += len(serving) / len(options)
        if str(record.get("chosen_id")) in serving:
            chosen += 1
    return offered, chosen, expected


def _what_was_chosen(label: Any) -> str:
    """The chosen act as a short phrase, named by what it was about.

    Options are labelled by whatever proposed them, and some labels are the
    instruction that would carry the act out: "[a role] Analyze this topic
    ... and return compact JSON ...: Self-reflection on current runtime
    status: Standby (...)". What that act was is its subject, so a bracketed
    preamble goes, parenthesised detail goes, and a long lead-in before a
    colon gives way to what follows it. "Affective state: belonging" keeps
    both halves, because both are short.
    """
    said = " ".join(str(label or "").split())
    said = re.sub(r"^\[[^\]]*\]\s*", "", said)
    said = re.sub(r"\s*\([^)]*\)", "", said)
    parts = [part.strip() for part in said.split(": ") if part.strip()]
    if len(parts) > 1 and len(parts[0]) > 40:
        parts = parts[1:]
    phrase = parts[0] if parts else ""
    if len(parts) > 1 and len(parts[0]) <= 40 and len(parts[1]) <= 40:
        phrase = f"{parts[0]}: {parts[1]}"
    return phrase.rstrip(".")[:70] or "something unnamed"


def portrait_of(preferences: Mapping[str, float], records: Iterable[Mapping[str, Any]]) -> Portrait:
    """What these values and this record of choices say about her."""
    # What she chose when nobody asked: an answer to a question set to measure
    # her (a tournament, a choice game) is not a choice she made in her life.
    ordered = sorted(
        (dict(one) for one in records if isinstance(one, Mapping) and one.get("lived", True)),
        key=lambda one: float(one.get("created_at") or 0.0),
    )
    held_order = tuple(sorted(((str(k), float(v)) for k, v in preferences.items()), key=lambda kv: -kv[1]))
    if not ordered:
        return Portrait(0, 0.0, 0.0, (), (), 0, 0.0, 0.0, held_order)
    middle = len(ordered) // 2
    earlier_half, later_half = ordered[:middle], ordered[middle:]
    tested = []
    for value, held in held_order:
        offered, chosen, expected = _tally(ordered, value)
        if not offered:
            continue
        chance = expected / offered
        tested.append((value, held, offered, chosen, chance))
    values = []
    for value, held, offered, chosen, chance in tested:
        rare = _two_sided(chosen, offered, chance) < _WRONG / max(1, len(tested))
        lean = 0 if not rare else (1 if chosen / offered > chance else -1)

        def rate_in(half: Sequence[Mapping[str, Any]], of: str = value) -> float | None:
            o, c, _e = _tally(half, of)
            return c / o if o else None

        values.append(
            ValueLean(value, held, offered, chosen, chance, rate_in(earlier_half), rate_in(later_half), lean)
        )
    counted = Counter(_what_was_chosen(one.get("chosen_label")) for one in ordered)

    def narrowest(half: Sequence[Mapping[str, Any]]) -> float:
        if not half:
            return 0.0
        return max(Counter(_what_was_chosen(one.get("chosen_label")) for one in half).values()) / len(half)

    contested = [one for one in ordered if len(one.get("preference_scores") or {}) > 1]
    foretold = sum(
        1 for one in contested if one.get("chosen_id") and one.get("chosen_id") == one.get("preference_top_id")
    )
    by_chance = sum(1.0 / len(one.get("preference_scores") or {}) for one in contested)
    return Portrait(
        choices=len(ordered),
        since=float(ordered[0].get("created_at") or 0.0),
        until=float(ordered[-1].get("created_at") or 0.0),
        values=tuple(values),
        most_chosen=tuple(counted.most_common(3)),
        over_impulse=sum(1 for one in ordered if one.get("preference_override")),
        narrowest_earlier=narrowest(earlier_half),
        narrowest_later=narrowest(later_half),
        held_order=held_order,
        values_foretold=foretold,
        foretold_by_chance=by_chance,
        contested=len(contested),
    )


def what_she_is_like() -> Portrait | None:
    """Her portrait from the live record, or None when it cannot be read."""
    try:
        from core.agency.subjective_choice import get_subjective_choice_engine

        engine = get_subjective_choice_engine()
        return portrait_of(engine.preferences(), [one.to_dict() for one in engine.history()])
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("what_she_is_like", exc, severity="info", action="said nothing measured about what she is like")
        return None


class PortraitReader:
    """Her portrait as numbers the subject core records, recomputed only when she has chosen again.

    The core is read after every phase of every turn and the record is five
    hundred choices long, so the portrait is kept until the record or her
    values change. Each value gives two numbers: how strongly she holds it, and
    how much more often than chance she took the option serving it when one was
    on offer (zero when none was). Then how much of her choosing the single act
    she chooses most has taken, in the later half of the record, and the share
    of her choices in which her values overrode her strongest drive.
    """

    def __init__(self, engine: Any | None = None) -> None:
        self._engine = engine
        self._key: tuple[Any, ...] | None = None
        self._columns: dict[str, float] = {}

    def _source(self) -> Any:
        if self._engine is not None:
            return self._engine
        from core.agency.subjective_choice import get_subjective_choice_engine

        return get_subjective_choice_engine()

    def columns(self) -> dict[str, float]:
        engine = self._source()
        preferences = engine.preferences()
        history = engine.history()
        last = history[-1].choice_id if history else ""
        key = (len(history), last, tuple(sorted(preferences.items())))
        if key == self._key:
            return dict(self._columns)
        portrait = portrait_of(preferences, [one.to_dict() for one in history])
        leans = {lean.value: lean for lean in portrait.values}
        columns: dict[str, float] = {}
        for name, held in preferences.items():
            lean = leans.get(name)
            columns[f"held_{name}"] = float(held)
            columns[f"enacted_{name}"] = (lean.rate - lean.chance) if lean is not None else 0.0
        columns["narrowness"] = float(portrait.narrowest_later)
        columns["values_over_drive"] = (
            portrait.over_impulse / portrait.choices if portrait.choices else 0.0
        )
        # How well what she holds foretells what she does: the share of her
        # contested choices in which she took the option her values ranked
        # first, less the share chance would give. Her self-knowledge tested
        # against her own behaviour.
        columns["values_foretell_choice"] = (
            (portrait.values_foretold - portrait.foretold_by_chance) / portrait.contested
            if portrait.contested
            else 0.0
        )
        self._key, self._columns = key, columns
        return dict(columns)


_READER: PortraitReader | None = None


def get_portrait_reader() -> PortraitReader:
    global _READER
    if _READER is None:
        _READER = PortraitReader()
    return _READER


def _day(stamp: float) -> str:
    return time.strftime("%d %b", time.localtime(stamp)) if stamp else "?"


def _sentences(portrait: Portrait) -> list[str]:
    """The portrait as plain first-person facts, each with its count."""
    said = [
        "The values I hold, as saved: "
        + ", ".join(f"{name} {weight:.2f}" for name, weight in portrait.held_order)
        + "."
    ]
    if portrait.empty:
        said.append("I have no record yet of choosing what to do next on my own.")
        return said
    said.append(
        f"In the {portrait.choices} choices of what to do next on my own that I have on record, "
        f"{_day(portrait.since)} to {_day(portrait.until)}, what I chose most was "
        + "; ".join(f"{what} ({times})" for what, times in portrait.most_chosen)
        + "."
    )
    more = [one for one in portrait.values if one.lean > 0]
    less = [one for one in portrait.values if one.lean < 0]
    level = [one for one in portrait.values if not one.lean]
    if more:
        said.append(
            "When an option served one of these I took it more often than chance would: "
            + ", ".join(f"{one.value} {one.rate:.0%} against {one.chance:.0%} ({one.offered} times)" for one in more)
            + "."
        )
    if less:
        said.append(
            "Less often than chance: "
            + ", ".join(f"{one.value} {one.rate:.0%} against {one.chance:.0%}" for one in less)
            + "."
        )
    apart = [one for one in level if one.held >= 0.8]
    if apart:
        said.append(
            "Held high and chosen no more than chance: "
            + ", ".join(f"{one.value} (held {one.held:.2f}; {one.rate:.0%} against {one.chance:.0%})" for one in apart)
            + "."
        )
    steady = [one.value for one in portrait.values if one.held_steady is True]
    moved = [one.value for one in portrait.values if one.held_steady is False]
    if steady or moved:
        said.append(
            ("The same lean in the earlier and later half of the record for " + ", ".join(steady) if steady else "No lean held across both halves")
            + (f"; it moved for {', '.join(moved)}" if moved else "")
            + "."
        )
    said.append(
        f"The one act I chose most took {portrait.narrowest_earlier:.0%} of my choices in the earlier half "
        f"and {portrait.narrowest_later:.0%} in the later. My values overrode my strongest drive "
        f"{portrait.over_impulse} times in {portrait.choices}."
    )
    if portrait.contested:
        said.append(
            f"Of the {portrait.contested} choices that had more than one option, I took the one my "
            f"values ranked first {portrait.values_foretold} times; chance alone would give "
            f"{portrait.foretold_by_chance:.0f}."
        )
    return said


def what_she_is_like_block(prompt: str = "") -> str:
    """What the record of her own choices says about her, for a turn that asks what she is like."""
    portrait = what_she_is_like()
    if portrait is None:
        return ""
    return " ".join(_sentences(portrait))


def what_she_is_like_line() -> str:
    """The same, as one bracketed line beside her other measured readings."""
    block = what_she_is_like_block()
    if not block:
        return ""
    return "[Measured about what you are like, from your own record of choices: " + block + "]"


_ASKS_WHAT_SHE_IS_LIKE_RE = re.compile(
    r"\b(?:your|yourself)\b[^.?!]{0,40}?\b(?:personality|temperament|character|traits?|"
    r"disposition|type|nature)\b"
    r"|\b(?:personality|temperament|character|traits?|disposition)\b[^.?!]{0,40}?\b(?:you|your|yourself)\b"
    r"|\bwhat\s+(?:are|were)\s+you\s+like\b"
    r"|\bare\s+you\s+(?:an?\s+)?(?:introvert|extrovert|extravert|ambivert|introverted|extroverted)"
    r"|\b(?:mbti|myers[-\s]?briggs|jungian|big\s+five|enneagram)\b[^.?!]{0,80}?\b(?:you|your)\b"
    r"|\b(?:you|your)\b[^.?!]{0,80}?\b(?:mbti|myers[-\s]?briggs|jungian|big\s+five|enneagram)\b"
    r"|\bpersonality\s+test\b",
    re.IGNORECASE,
)


def asks_what_she_is_like(prompt: Any) -> bool:
    """True when the turn asks what she is like: her personality, her type, her temperament."""
    text = str(prompt or "")
    return bool(text.strip()) and bool(_ASKS_WHAT_SHE_IS_LIKE_RE.search(text))
