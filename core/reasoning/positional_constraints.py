"""Who sits where, worked out rather than described.

LIVE, 2026-08-20. Six people, a round table, four constraints, one
arrangement. Asked twice, she narrated twice and got it wrong both times —
once naming Emil correctly and Dara's neighbours wrongly, once stating a
layout in which Dara was "opposite Ada" one line after Boris was.

Offering her a Python sandbox did not help: the tool set reached the turn and
the model answered directly anyway. Prompting it to use the sandbox is not
available and would not be a fix. Where the answer follows from the
constraints, the runtime computes it, the same way it already computes a
column of a spreadsheet and the product of two numbers.

The family is narrow and common: entities at positions, in a ring or a row,
related by adjacency, distance, opposition and immediate order. Seating,
queues, shelves, finishing order, house-order puzzles. What makes it
tractable is that the domain is finite and small, so enumeration is not a
heuristic — it IS the answer.

Reported only when every arrangement that satisfies the constraints agrees.
A round table has no absolute positions, so the arrangements always differ by
rotation and often by reflection; what has to be unique is the ANSWER, not
the seating.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import permutations

__all__ = [
    "PositionalAnswer",
    "PositionalProblem",
    "answer_positional_problem",
    "describe_positional_answer",
    "parse_positional_problem",
]

#: Small enough to enumerate exhaustively, large enough for any problem a
#: person states in a sentence.
MAX_ENTITIES = 9

#: An occupant the problem counts but never names.
_UNNAMED = "\u2014unnamed-"


def _spoken(name: str) -> str:
    """How an occupant is referred to in an answer."""
    return "someone the problem does not name" if name.startswith(_UNNAMED) else name

_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
}

_RING_HINT = re.compile(
    r"\b(?:round|circular|circle|ring|carousel|roundtable)\b", re.IGNORECASE
)
_ROW_HINT = re.compile(
    r"\b(?:in\s+a\s+(?:row|line|queue)|side\s+by\s+side|shelf|bench|lined\s+up)\b",
    re.IGNORECASE,
)

#: A capitalised word. Sentence position says nothing here: in problems of
#: this shape the names are usually the subjects, so "Boris sits directly
#: opposite Ada." opens with one. Excluding sentence-initial capitals found
#: three names of five and the parse declined a problem it could solve.
_NAME_RE = re.compile(r"\b([A-Z][a-z]{1,15})\b")

#: Capitalised words that are English rather than names. Everything here is a
#: word a person writes at the start of a sentence, or the vocabulary of the
#: problem itself.
_NOT_A_NAME = frozenset(
    {
        "a", "all", "also", "an", "and", "another", "any", "are", "around",
        "assume", "at", "because", "bench", "beside", "between", "both",
        "but", "by", "chair", "chairs", "circle", "clockwise", "consider",
        "directly", "each", "eight", "every", "exactly", "finally", "first",
        "five", "for", "four", "friends", "from", "given", "guests", "he",
        "her", "here", "his", "how", "however", "if", "immediately", "in",
        "is", "it", "last", "left", "line", "next", "nine", "no", "not",
        "note", "now", "of", "on", "one", "opposite", "or", "people",
        "person", "persons", "place", "places", "players", "position",
        "positions", "queue", "right", "round", "row", "seat", "seats",
        "second", "seven", "shelf", "she", "since", "sits", "six", "so",
        "students", "table", "ten", "that", "the", "their", "then", "there",
        "therefore", "these", "they", "third", "this", "those", "three",
        "to", "two", "we", "what", "when", "where", "which", "who", "whom",
        "why", "with", "you",
    }
)

#: How many take part. "places" is deliberately absent: in these problems a
#: place is a POSITION, so "exactly two places from Rosa" read as a
#: population of two and the parse gave up on a five-runner problem.
_COUNT_RE = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|\d{1,2})\s+"
    r"(?:people|persons?|friends?|students?|guests?|players?|seats?|chairs?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Constraint:
    """One stated relation, and the test it imposes on an arrangement."""

    text: str
    test: Callable[[dict[str, int], int], bool]


@dataclass(frozen=True, slots=True)
class Question:
    """One thing the person asked, and how to read it off an arrangement."""

    text: str
    subject: str
    kind: str
    read: Callable[[dict[str, int], int], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class PositionalProblem:
    entities: tuple[str, ...]
    seats: int
    cyclic: bool
    constraints: tuple[Constraint, ...]
    questions: tuple[Question, ...]


@dataclass(frozen=True, slots=True)
class PositionalAnswer:
    """What every valid arrangement agrees on, and what they do not.

    A puzzle with two consistent seatings has a real answer — "these two fit"
    — and reporting only unanimous findings returned nothing at all for it.
    Silence is the wrong answer to ambiguity: it reads as inability where the
    truth is that the clues do not decide.
    """

    findings: tuple[tuple[str, tuple[str, ...]], ...]
    arrangements: int
    entities: tuple[str, ...]
    seats: int
    cyclic: bool
    alternatives: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = ()


def _gap(a: int, b: int, seats: int, cyclic: bool) -> int:
    distance = abs(a - b)
    return min(distance, seats - distance) if cyclic else distance


def _names(text: str) -> list[str]:
    """Capitalised words that behave like names in this problem."""
    found: list[str] = []
    for match in _NAME_RE.finditer(text):
        name = match.group(1)
        if name.lower() in _NOT_A_NAME:
            continue
        if name not in found:
            found.append(name)
    # A possessive written without its apostrophe is the same person.
    #
    # LIVE, 2026-08-20. "who are Daras two neighbours" put a sixth person at
    # a six-seat table and the answer came out wrong; "Rosas neighbours" made
    # a sixth runner and the parse declined a problem it could settle. People
    # type it this way constantly, and a name that is another name plus an s
    # is not a new name.
    base = {name.lower() for name in found}
    return [
        name
        for name in found
        if not (name.lower().endswith("s") and name[:-1].lower() in base)
    ]


def _seat_count(text: str, names: list[str]) -> int:
    match = _COUNT_RE.search(text)
    if match:
        token = match.group(1).lower()
        stated = _NUMBER_WORDS.get(token)
        if stated is None:
            try:
                stated = int(token)
            except ValueError:
                stated = 0
        if stated:
            return stated
    return len(names)


def _relations(text: str, names: Iterable[str], seats: int, cyclic: bool) -> list[Constraint]:
    known = {name.lower(): name for name in names}
    if not known:
        return []
    alternation = "|".join(re.escape(name) for name in known.values())
    found: list[Constraint] = []

    def add(sentence: str, test: Callable[[dict[str, int], int], bool]) -> None:
        found.append(Constraint(" ".join(sentence.split()), test))

    for sentence in re.split(r"(?<=[.!?])\s+|;", text):
        # What is asked is not also given. "Who sits opposite Chen" states no
        # premise, and read as one it over-constrained the problem until
        # nothing satisfied it.
        if "?" in sentence or re.match(r"\s*(?:who|which|where)\b", sentence, re.IGNORECASE):
            continue
        clause_text = " ".join(sentence.split())
        lowered = clause_text.lower()
        sentence_negated = bool(re.search(r"\b(?:not|never|n't|no)\b", lowered))

        # A constraint about ONE person.
        #
        # This required two names in a sentence, so "Tomas sits at one of the
        # two ends" and "Wen is not at either end" were dropped without a
        # trace — and a puzzle missing two of its five premises has more
        # solutions than it should, or none.
        solo = re.findall(rf"\b({alternation})\b", sentence, re.IGNORECASE)
        if not cyclic and len(set(name.lower() for name in solo)) == 1:
            who = known[solo[0].lower()]
            if re.search(r"\b(?:either\s+end|one\s+of\s+the\s+(?:two\s+)?ends|an?\s+end)\b", lowered):
                if sentence_negated:
                    add(sentence, lambda s, n, a=who: 0 < s[a] < n - 1)
                else:
                    add(sentence, lambda s, n, a=who: s[a] in (0, n - 1))
                continue
            if re.search(r"\b(?:far\s+left|leftmost|first\s+(?:seat|chair))\b", lowered):
                add(sentence, lambda s, n, a=who: s[a] == 0)
                continue
            if re.search(r"\b(?:far\s+right|rightmost|last\s+(?:seat|chair))\b", lowered):
                add(sentence, lambda s, n, a=who: s[a] == n - 1)
                continue
            if re.search(r"\b(?:middle|centre|center)\b", lowered):
                add(sentence, lambda s, n, a=who: s[a] == (n - 1) // 2)
                continue

        pair = re.search(
            rf"\b({alternation})(?:'s|\u2019s|s'|s)?\b"
            rf"(?P<mid>[^.?!]{{0,60}}?)"
            rf"\b({alternation})(?:'s|\u2019s|s'|s)?\b",
            sentence,
            re.IGNORECASE,
        )
        if not pair:
            continue
        first = known[pair.group(1).lower()]
        second = known[pair.group(3).lower()]
        # The relation is read from the WHOLE clause, not the span between the
        # two names: "Chen and Dara sit next to each other" states adjacency
        # after naming both, and reading only the middle found " and " and
        # dropped the constraint. Ordering-sensitive relations still consult
        # the middle, because "A clockwise of B" and "B clockwise of A" differ.
        clause = " ".join(sentence.split()).lower()
        middle = pair.group("mid").lower()
        negated = bool(re.search(r"\b(?:not|never|n't|no)\b", clause))

        if re.search(r"\bopposite\b", clause):
            if seats % 2 == 0 and cyclic:
                half = seats // 2
                add(sentence, lambda s, n, a=first, b=second, h=half: (
                    _gap(s[a], s[b], n, True) == h
                ))
            continue
        clockwise = re.search(r"\b(?:immediately|directly)?\s*clockwise\b", middle)
        anticlockwise = re.search(
            r"\b(?:anti[- ]?clockwise|counter[- ]?clockwise)\b", middle
        )
        if anticlockwise:
            add(sentence, lambda s, n, a=first, b=second: (s[a] - 1) % n == s[b] % n)
            continue
        if clockwise:
            add(sentence, lambda s, n, a=first, b=second: (s[b] + 1) % n == s[a] % n)
            continue
        exact = re.search(
            r"\b(two|three|four|five|six|seven|eight|nine|\d{1,2})\s+"
            r"(?:seats?|places?|chairs?|positions?)\s+(?:from|away|apart)",
            clause,
        )
        if exact:
            token = exact.group(1).lower()
            distance = _NUMBER_WORDS.get(token) or int(token)
            add(sentence, lambda s, n, a=first, b=second, d=distance, c=cyclic: (
                _gap(s[a], s[b], n, c) == d
            ))
            continue
        # X sits between A and B — a three-name relation, so the pair match
        # above found only the first two and the premise was lost.
        if not cyclic and re.search(r"\bbetween\b", clause):
            trio = re.findall(rf"\b({alternation})\b", sentence, re.IGNORECASE)
            unique = list(dict.fromkeys(name.lower() for name in trio))
            if len(unique) == 3:
                middle_name = known[unique[0]]
                left_name, right_name = known[unique[1]], known[unique[2]]
                directly = bool(re.search(r"\b(?:directly|immediately|right)\b", clause))
                if directly:
                    add(sentence, lambda s, n, m=middle_name, a=left_name, b=right_name: (
                        abs(s[m] - s[a]) == 1
                        and abs(s[m] - s[b]) == 1
                        and min(s[a], s[b]) < s[m] < max(s[a], s[b])
                    ))
                else:
                    add(sentence, lambda s, n, m=middle_name, a=left_name, b=right_name: (
                        min(s[a], s[b]) < s[m] < max(s[a], s[b])
                    ))
                continue

        # Which side of whom, in a row. "somewhere to the left of" is an
        # ordering; "directly to the left of" is adjacency with a direction.
        if not cyclic:
            left_of = re.search(r"\bto\s+the\s+left\s+of\b|\bleft\s+of\b", middle)
            right_of = re.search(r"\bto\s+the\s+right\s+of\b|\bright\s+of\b", middle)
            immediate = bool(re.search(r"\b(?:directly|immediately)\b", middle))
            if left_of or right_of:
                if left_of:
                    if immediate:
                        add(sentence, lambda s, n, a=first, b=second: s[a] + 1 == s[b])
                    else:
                        add(sentence, lambda s, n, a=first, b=second: s[a] < s[b])
                else:
                    if immediate:
                        add(sentence, lambda s, n, a=first, b=second: s[a] - 1 == s[b])
                    else:
                        add(sentence, lambda s, n, a=first, b=second: s[a] > s[b])
                continue

        if re.search(r"\b(?:next\s+to|beside|adjacent|alongside)\b", clause):
            if negated:
                add(sentence, lambda s, n, a=first, b=second, c=cyclic: (
                    _gap(s[a], s[b], n, c) != 1
                ))
            else:
                add(sentence, lambda s, n, a=first, b=second, c=cyclic: (
                    _gap(s[a], s[b], n, c) == 1
                ))
    return found


def _questions(text: str, names: Iterable[str], seats: int, cyclic: bool) -> list[Question]:
    known = {name.lower(): name for name in names}
    if not known:
        return []
    alternation = "|".join(re.escape(name) for name in known.values())
    asked: list[Question] = []

    # Only from what was ASKED. Reading the whole text found "opposite Ada"
    # inside "Boris sits directly opposite Ada" and answered a question
    # nobody put, restating a premise as a finding.
    interrogative = " ".join(
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or ""))
        if "?" in sentence or re.match(r"\s*(?:who|which|where)\b", sentence, re.IGNORECASE)
    )
    if not interrogative.strip():
        return []
    text = interrogative

    for match in re.finditer(
        rf"\bopposite\s+({alternation})(?:'s|\u2019s|s'|s)?\b", text, re.IGNORECASE
    ):
        subject = known[match.group(1).lower()]
        if not cyclic or seats % 2:
            continue
        asked.append(
            Question(
                f"opposite {subject}",
                subject,
                "opposite",
                lambda s, n, a=subject, h=seats // 2: tuple(
                    sorted(name for name, seat in s.items() if _gap(seat, s[a], n, True) == h)
                ),
            )
        )

    for match in re.finditer(
        rf"\b({alternation})(?:'s|\u2019s|s'|s)?\s+(?:two\s+)?neighbou?rs?\b"
        rf"|\bnext\s+to\s+({alternation})(?:'s|\u2019s|s'|s)?\b"
        rf"|\bneighbou?rs?\s+of\s+({alternation})(?:'s|\u2019s|s'|s)?\b",
        text,
        re.IGNORECASE,
    ):
        token = match.group(1) or match.group(2) or match.group(3)
        subject = known[str(token).lower()]
        asked.append(
            Question(
                f"{subject}'s neighbours",
                subject,
                "neighbours",
                lambda s, n, a=subject, c=cyclic: tuple(
                    sorted(name for name, seat in s.items() if _gap(seat, s[a], n, c) == 1)
                ),
            )
        )

    # The whole arrangement — the plainest question a seating puzzle has, and
    # the one form that was missing. "Who sits where?" parsed no question at
    # all, so a problem whose premises all read correctly was still rejected
    # as unparseable and went to the model, which spent 103 seconds on it and
    # was cut off mid-reasoning.
    if re.search(
        r"\bwho\s+sits\s+where\b"
        r"|\bwhat\s+is\s+the\s+(?:order|arrangement|seating)\b"
        r"|\bwhere\s+(?:does|do)\s+(?:each|everyone|everybody)\b"
        r"|\bwork\s+out\s+the\s+(?:order|arrangement|seating)\b"
        r"|\b(?:give|tell)\s+me\s+the\s+(?:order|arrangement|seating)\b",
        text,
        re.IGNORECASE,
    ):
        asked.append(
            Question(
                "the seating order",
                "",
                "arrangement",
                lambda s, n: tuple(
                    name for name, _seat in sorted(s.items(), key=lambda item: item[1])
                ),
            )
        )

    deduped: list[Question] = []
    for question in asked:
        if all(question.text != existing.text for existing in deduped):
            deduped.append(question)
    return deduped


def parse_positional_problem(text: object) -> PositionalProblem | None:
    """The problem stated in this text, or None when it is not one.

    Conservative at every step: no names, no seats, no relations or no
    question and it declines, because a wrong arrangement served with
    authority is worse than leaving the turn to the model.
    """
    body = str(text or "")
    if len(body) < 40:
        return None
    names = _names(body)
    if not 3 <= len(names) <= MAX_ENTITIES:
        return None
    seats = _seat_count(body, names)
    if not len(names) <= seats <= MAX_ENTITIES or seats < 3:
        return None
    # A problem may state more seats than it names people: "six people sit
    # around a round table" and then name five. The sixth is still an
    # occupant and still takes a seat, so the arrangement is only right if it
    # is placed — and an answer that lands on them says so.
    occupants = list(names) + [
        f"{_UNNAMED}{index}" for index in range(1, seats - len(names) + 1)
    ]
    cyclic = bool(_RING_HINT.search(body)) and not _ROW_HINT.search(body)
    constraints = _relations(body, names, seats, cyclic)
    if len(constraints) < 2:
        return None
    questions = _questions(body, names, seats, cyclic)
    if not questions:
        return None
    return PositionalProblem(
        entities=tuple(occupants),
        seats=seats,
        cyclic=cyclic,
        constraints=tuple(constraints),
        questions=tuple(questions),
    )


def answer_positional_problem(text: object) -> PositionalAnswer | None:
    """Everything every valid arrangement agrees on, or None."""
    problem = parse_positional_problem(text)
    if problem is None:
        return None

    entities = list(problem.entities)
    seats = problem.seats
    # A ring has no absolute positions, so one entity is pinned and the
    # rotations it stands for are counted once. Every relation asked about is
    # rotation-invariant, which is what makes that sound.
    if problem.cyclic:
        head, rest = entities[0], entities[1:]
    else:
        head, rest = None, entities

    solutions: list[dict[str, int]] = []
    for order in permutations(rest):
        arrangement: dict[str, int] = {}
        if head is not None:
            arrangement[head] = 0
            for index, name in enumerate(order, start=1):
                arrangement[name] = index
        else:
            for index, name in enumerate(order):
                arrangement[name] = index
        if all(rule.test(arrangement, seats) for rule in problem.constraints):
            solutions.append(arrangement)
        if len(solutions) > 5000:
            return None

    if not solutions:
        return None

    findings: list[tuple[str, tuple[str, ...]]] = []
    alternatives: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    for question in problem.questions:
        readings = {question.read(arrangement, seats) for arrangement in solutions}
        if len(readings) == 1:
            answer = readings.pop()
            if answer:
                findings.append((question.text, answer))
            continue
        # Few enough to state. Past that the honest report is the count, which
        # the answer already carries.
        if len(readings) <= _MOST_ALTERNATIVES_WORTH_STATING:
            ordered = tuple(sorted(reading for reading in readings if reading))
            if ordered:
                alternatives.append((question.text, ordered))
    if not findings and not alternatives:
        return None
    return PositionalAnswer(
        findings=tuple(findings),
        alternatives=tuple(alternatives),
        arrangements=len(solutions),
        entities=problem.entities,
        seats=seats,
        cyclic=problem.cyclic,
    )


#: More than this and listing them says less than counting them.
_MOST_ALTERNATIVES_WORTH_STATING = 4


def describe_positional_answer(answer: PositionalAnswer | None) -> str:
    """The findings as sentences, or "" when there is nothing settled."""
    if answer is None or (not answer.findings and not answer.alternatives):
        return ""
    lines: list[str] = []
    for asked, names in answer.findings:
        spoken = [_spoken(name) for name in names]
        # Only the first letter: capitalize() lowercases the rest, and the
        # rest is a name.
        label = asked[:1].upper() + asked[1:]
        if len(spoken) == 1:
            lines.append(f"{label}: {spoken[0]}.")
        else:
            lines.append(f"{label}: {' and '.join(spoken)}.")
    for asked, readings in answer.alternatives:
        label = asked[:1].upper() + asked[1:]
        shown = "; ".join(
            ", ".join(_spoken(name) for name in reading) for reading in readings
        )
        lines.append(
            f"{label} is not settled by these clues — {len(readings)} fit: {shown}."
        )
    return " ".join(lines)
