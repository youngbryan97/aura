"""What counts as done, for anything a person asks to be carried through.

Whatever its domain, a thing asked of her has one of three shapes:

- an END the thing itself declares: a game won, a level cleared, a story
  finished, a puzzle solved, a form accepted. It is done when that is reached.
- a MEASURE it keeps and never closes: a score, a best time, a count. No run
  reaches an end; a run is bettered, not finished.
- a thing MADE, with no end and no measure: a character built, a picture drawn,
  a playlist put together. It is done once it is made.

A person says what to do with each shape in the request, as conditions: "if it
can be won, play until you win it; if it only keeps a score, play it three times
and tell me your best; if it has neither, do it once and tell me what you made
and why." Those conditions are read here into terms, so how long to go on, how
many runs, and what to say at the end come from the person's words and not from
a number written into the code. Which shape a particular thing has is read from
the thing itself (`shape_in`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["END", "MADE", "MEASURE", "Terms", "conditions_as_said", "shape_in", "terms_asked", "without_conditions"]

END: Final = "end"
MEASURE: Final = "measure"
MADE: Final = "made"

_NUMBERS: Final = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_A_COUNT: Final = r"(\d{1,3}|" + "|".join(_NUMBERS) + r")"
#: A count of runs in whatever the runs are called: "three times", "two drafts", "best of five".
_RUNS: Final = re.compile(r"\b" + _A_COUNT + r"\s+(?:more\s+)?(?:times|\w+s)\b|\bbest\s+(?:of|out\s+of)\s+" + _A_COUNT + r"\b", re.I)

#: Where a condition opens: "if it ...", "where there is ...", "for anything that ...".
_CONDITION: Final = re.compile(r"^\s*(?:and\s+|or\s+|but\s+)?(?:if|where|when|whenever|wherever|in case|for (?:anything|any one|one|those|games?|things?) that)\b", re.I)

#: A condition about a thing with neither an end nor a measure: a thing for making.
_NEITHER: Final = re.compile(
    r"\b(neither|nothing to (?:win|score|beat)\b.*\b(?:or|nor) (?:score|measure)|no (?:score|measure)\b|not even a (?:score|measure|high score)|"
    r"(?:just |only )?for fun|for (?:making|creating|building)|(?:make|create|build|design|draw|dress)\w*\b|creator|maker|sandbox|no (?:agenda|goal) at all)",
    re.I,
)
#: A condition about a thing that keeps a measure and has no end.
_A_MEASURE: Final = re.compile(
    r"\b(score|points|high score|\w*\s?time|record|measure|count|rating|rank|total|arcade|no way to win|no (?:win|winner|winning)|"
    r"(?:can ?not|can'?t) (?:be )?(?:won|win|beaten)|(?:isn'?t|is not|not) winnable|nothing to win|no (?:end|ending|objective|goal))\b",
    re.I,
)
#: A condition about a thing with an end of its own.
_AN_END: Final = re.compile(
    r"\b(can be (?:won|beaten|finished|completed|solved|cleared)|winnable|has (?:a|an) (?:win|winning|end|ending|objective|goal|story|finish)|"
    r"(?:a|an|its) (?:win condition|objective|goal|story|ending|way to win)|to win|beat|finish|complete|solve|clear)\b",
    re.I,
)

_UNTIL: Final = re.compile(
    r"\b(until (?:you|she|it|they)?\s*(?:actually |really |finally )?(?:win|beat|finish|complete|solve|clear|reach|get there|have won|'?ve won|is won|is beaten|is done)|"
    r"keep (?:going|playing|at it|trying|on)|don'?t (?:stop|give up))",
    re.I,
)
_BEST: Final = re.compile(r"\b(best|highest|top|record|high score)\b", re.I)
_ONCE: Final = re.compile(r"\b(once|one time|a single time|just the one)\b", re.I)
_TELL_MADE: Final = re.compile(r"\b(?:tell|say|explain|describe|show)\b.{0,40}\b(?:made|make|created|built|chose|why)\b", re.I)


@dataclass(frozen=True)
class Terms:
    """What the person asked to be done with each shape of thing; empty where they said nothing of it."""

    until_the_end: bool = False
    measure_runs: int | None = None
    tell_the_best: bool = False
    made_once: bool = False
    tell_what_was_made: bool = False
    said: tuple[str, ...] = ()

    def says_anything(self) -> bool:
        return bool(self.said)


def _clauses(request: str) -> list[str]:
    """The request's clauses: its sentences, each cut again at semicolons, colons, and a new condition after a comma."""
    text = " ".join(str(request or "").split())
    parts: list[str] = []
    for sentence in re.split(r"(?<=[.!?;:])\s+", text):
        parts += [p for p in re.split(r",\s*(?=(?:and\s+|or\s+|but\s+)?(?:if|where|when|whenever|wherever)\b)", sentence, flags=re.I) if p.strip()]
    return [p.strip() for p in parts]


def _shape_of_condition(condition: str) -> str:
    if _NEITHER.search(condition):
        return MADE
    if _A_MEASURE.search(condition):
        return MEASURE
    if _AN_END.search(condition):
        return END
    return ""


def _a_count(text: str) -> int | None:
    found = _RUNS.search(text)
    if not found:
        return None
    word = (found.group(1) or found.group(2) or "").casefold()
    value = _NUMBERS.get(word) if word in _NUMBERS else int(word)
    return value if value and value > 0 else None


def _conditions(request: str) -> list[tuple[str, str, str]]:
    """Each condition the request states, as (shape, condition, consequence)."""
    found = []
    for clause in _clauses(request):
        if not _CONDITION.search(clause):
            continue
        condition, _, consequence = clause.partition(",")
        if not consequence:
            condition, _, consequence = clause.partition(" then ")
        shape = _shape_of_condition(condition)
        if shape:
            found.append((shape, condition.strip(), consequence.strip()))
    return found


def terms_asked(request: str) -> Terms:
    """What the request says to do with a thing that has an end, a measure, or neither."""
    until = runs = None
    best = once = tell_made = False
    said = []
    for shape, condition, consequence in _conditions(request):
        said.append(f"{condition}, {consequence}".strip(" ,"))
        if shape == END:
            until = until or bool(_UNTIL.search(consequence))
        elif shape == MEASURE:
            runs = runs or _a_count(consequence)
            best = best or bool(_BEST.search(consequence))
        elif shape == MADE:
            once = once or bool(_ONCE.search(consequence)) or _a_count(consequence) == 1
            tell_made = tell_made or bool(_TELL_MADE.search(consequence))
    return Terms(until_the_end=bool(until), measure_runs=runs, tell_the_best=best, made_once=once,
                 tell_what_was_made=tell_made, said=tuple(said))


def conditions_as_said(request: str) -> str:
    """The request's conditions on what counts as done, in the person's own words, joined as one passage."""
    said = terms_asked(request).said
    return " ".join(s[:1].upper() + s[1:].rstrip(" .;:") + "." for s in said)


def without_conditions(request: str) -> str:
    """The request with its conditions on what counts as done taken out, so a count said of one shape is not taken for every thing."""
    kept = [c for c in _clauses(request) if not (_CONDITION.search(c) and _shape_of_condition(c.partition(",")[0]))]
    return " ".join(kept)


def shape_in(words: str, *, a_measure_seen: bool = False) -> str:
    """Which shape a thing is, from its own words (and whether a measure of it has been seen): END, MEASURE, MADE, or "".

    What it says it sets out to do comes first: something to make, an
    objective, a score to run up. Then a win it names. Then a measure it
    keeps: a thing that ends each run with a score is played for the score,
    levels or not, as an arcade game is. Levels or stages with no measure
    shown are steps to an end.
    """
    from core.language.how_a_game_ended import (
        declares_a_win,
        offers_a_win,
        what_it_asks_of_a_player,
    )

    asked = what_it_asks_of_a_player(words)
    if asked == "make":
        return MADE
    if asked == "win":
        return END
    if asked == "score":
        return MEASURE
    if declares_a_win(words):
        return END
    if a_measure_seen:
        return MEASURE
    if offers_a_win(words):
        return END
    return ""
