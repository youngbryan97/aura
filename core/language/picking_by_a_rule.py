"""Picking from a list by a rule a person states: read as the steps it is, and worked out exactly.

"Number the games on the list from 0. Take the current minute of the hour,
divide it by how many games there are, and play the game whose number is the
remainder. When that game is over, add 19 to the number, take the remainder
again, and play that game. Then add 19 once more for the third game." That is
a procedure: a starting number (from the clock, a date, or a number said), the
steps done to it, how the list is counted, and how each next pick comes from
the last. Left to her model, it was not followed: LIVE 2026-10-05 she opened the
first game on the list whatever the minute was. Here the rule is read from the
words into those parts, and each pick is worked out from the list as it is on
the page and the clock as it is when she starts, with a line for a person
watching that shows the working.

Nothing here knows what is being picked: games, songs, recipes, rows.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

__all__ = ["PickingRule", "Pick", "a_picking_rule"]

_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "twice": 2, "once": 1}

#: Where a starting number comes from, and how it is said.
_STARTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("last digit of the minute", re.compile(r"\b(?:last|final|ones?|terminal)\s+digit\s+of\s+(?:the\s+)?(?:current\s+)?(?:time|minute|clock)\b", re.I)),
    ("minute", re.compile(r"\b(?:current\s+minute|minute\s+of\s+the\s+hour|the\s+minutes?\s+(?:it\s+is|now|on\s+the\s+clock)|minutes?\s+past\s+the\s+hour)\b", re.I)),
    ("second", re.compile(r"\bcurrent\s+second\b|\bseconds?\s+(?:on\s+the\s+clock|now)\b", re.I)),
    ("hour", re.compile(r"\bcurrent\s+hour\b|\bhour\s+of\s+the\s+day\b", re.I)),
    ("day of the month", re.compile(r"\b(?:today'?s\s+date|day\s+of\s+the\s+month|date\s+today)\b", re.I)),
    ("day of the year", re.compile(r"\bday\s+of\s+the\s+year\b", re.I)),
)

#: Steps done to the number, in the order they are said.
_STEPS = re.compile(
    r"(?P<add>\b(?:add|plus|adding)\s+(?P<a>\d+|[a-z]+)(?:\s+to\s+(?:it|the\s+number|that))?)"
    r"|(?P<sub>\b(?:subtract|minus|take\s+away)\s+(?P<s>\d+|[a-z]+))"
    r"|(?P<mul>\b(?:multiply\s+(?:it\s+|that\s+)?by|times)\s+(?P<m>\d+|[a-z]+))"
    r"|(?P<mod>\b(?:divide\w*\s+(?:it\s+|that\s+)?by\s+(?:how\s+many|the\s+number\s+of)|remainder|modulo|mod\b|wrap\w*\s+around|count\w*\s+(?:that\s+many|it)\s+\w*\s*forward))",
    re.I,
)

#: Words that begin the next pick.
_NEXT = re.compile(r"\b(?:when\s+(?:that|it|the\s+\w+)\s+(?:\w+\s+)?(?:is\s+)?(?:over|done|finished|ends)|go\s+back\s+to\s+the\s+list|then\b|after\s+that|next\b|for\s+the\s+(?:second|third|fourth|fifth|next))", re.I)
_AGAIN = re.compile(r"\b(?:once\s+more|again|the\s+same\s+way)\b", re.I)


def _number(said: str) -> int | None:
    said = said.lower()
    return int(said) if said.isdigit() else _NUMBERS.get(said)


@dataclass
class Pick:
    number: int
    index: int
    item: str
    working: str


@dataclass
class PickingRule:
    """How a person said to pick from a list: where the number starts, the steps for the first pick and each next one."""

    start: str
    first: list[tuple[str, int]] = field(default_factory=list)
    then: list[list[tuple[str, int]]] = field(default_factory=list)
    counted_from: int = 1
    picks: int = 1

    def worked_out(self, items: list[str], now: datetime.datetime | None = None) -> list[Pick]:
        """Each pick from ``items`` (as listed, in order), with the working shown; none where the list is empty."""
        if not items:
            return []
        now = now or datetime.datetime.now()
        number, said = _starting(self.start, now)
        out: list[Pick] = []
        for n in range(self.picks):
            steps = self.first if n == 0 else (self.then[min(n - 1, len(self.then) - 1)] if self.then else self.first)
            out.append(self._stepped(items, number, steps, said if n == 0 else f"from {number}"))
            number = out[-1].number
        return out

    def going_on(self, items: list[str], picks: list[Pick]) -> Pick | None:
        """The next pick the rule makes after ``picks``, by its last step, passing over items already picked; None once it only comes back round.

        Where a pick cannot be had at all, a person asked for three keeps to three by
        going on the way the rule goes: "add 19 once more".
        """
        if not items or not picks:
            return None
        steps = self.then[-1] if self.then else self.first
        taken = {p.index for p in picks}
        number = picks[-1].number
        for _ in range(len(items)):
            pick = self._stepped(items, number, steps, f"from {number}")
            if pick.number == number and pick.index in taken:
                return None  # the step does not move it: there is no going on
            if pick.index not in taken:
                return pick
            number = pick.number
        return None

    def _stepped(self, items: list[str], number: int, steps: list[tuple[str, int]], said: str) -> Pick:
        count = len(items)
        working = [said]
        for op, by in steps:
            before = number
            if op == "add":
                number += by
                working.append(f"{before} + {by} = {number}")
            elif op == "sub":
                number -= by
                working.append(f"{before} - {by} = {number}")
            elif op == "mul":
                number *= by
                working.append(f"{before} × {by} = {number}")
            elif op == "mod":
                number %= count
                working.append(f"{before} divided by {count} leaves {number}")
        index = (number - self.counted_from) % count
        return Pick(number, index, items[index], "; ".join(working))

    def says(self) -> str:
        return (f"start from the {self.start}, counting the list from {self.counted_from}; "
                f"{self.picks} pick{'s' if self.picks != 1 else ''}")


def _starting(start: str, now: datetime.datetime) -> tuple[int, str]:
    clock = now.strftime("%-I:%M %p").lower() if hasattr(now, "strftime") else ""
    if start == "minute":
        return now.minute, f"it is {clock}, so the minute is {now.minute}"
    if start == "last digit of the minute":
        return now.minute % 10, f"it is {clock}, so the last digit is {now.minute % 10}"
    if start == "second":
        return now.second, f"the second is {now.second}"
    if start == "hour":
        return now.hour, f"the hour is {now.hour}"
    if start == "day of the month":
        return now.day, f"today is the {now.day}{_th(now.day)}"
    if start == "day of the year":
        day = now.timetuple().tm_yday
        return day, f"today is day {day} of the year"
    said = re.match(r"number (\d+)", start)
    return (int(said.group(1)), f"the number is {said.group(1)}") if said else (0, "starting from 0")


def _th(n: int) -> str:
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _steps(text: str) -> list[tuple[str, int]]:
    steps: list[tuple[str, int]] = []
    for m in _STEPS.finditer(text):
        if m.group("add") and _number(m.group("a")) is not None:
            steps.append(("add", _number(m.group("a")) or 0))
        elif m.group("sub") and _number(m.group("s")) is not None:
            steps.append(("sub", _number(m.group("s")) or 0))
        elif m.group("mul") and _number(m.group("m")) is not None:
            steps.append(("mul", _number(m.group("m")) or 1))
        elif m.group("mod") and not (steps and steps[-1][0] == "mod"):
            steps.append(("mod", 0))
    return steps


def a_picking_rule(words: str) -> PickingRule | None:
    """The rule ``words`` give for picking from a list, or None where they give none.

    A rule has a starting number from something that changes (the clock, the
    date) or a number said, and at least one step or count done to it.
    """
    text = " ".join(str(words or "").split())
    start, at = "", len(text)
    for name, said in _STARTS:
        found = said.search(text)
        if found and found.start() < at:
            start, at = name, found.start()
    if not start:
        said_number = re.search(r"\bstart(?:ing)?\s+(?:at|from|with)\s+(?:number\s+)?(\d+)\b", text, re.I)
        if not said_number:
            return None
        start, at = f"number {said_number.group(1)}", said_number.start()
    counted = re.search(r"\bnumber\w*\s+(?:the\s+)?[\w\s]{0,40}?\bfrom\s+(0|zero|1|one)\b", text, re.I)
    counted_from = 0 if counted and counted.group(1).lower() in ("0", "zero") else 1
    if re.search(r"\bcount\w*\s+(?:that\s+many|it)\s+\w*\s*forward\s+from\s+the\s+first\b", text, re.I):
        counted_from = 0  # counting N forward from the first lands on the item numbered N from 0
    # The first pick's steps run from the starting number to the first word that begins another pick.
    rest = text[at:]
    cut = [m.start() for m in _NEXT.finditer(rest) if m.start() > 0]
    groups = [rest[a:b] for a, b in zip([0, *cut], [*cut, len(rest)], strict=False)]
    first = _steps(groups[0])
    if not first:
        return None
    then: list[list[tuple[str, int]]] = []
    for group in groups[1:]:
        steps = _steps(group)
        before = then[-1] if then else first
        if steps and _AGAIN.search(group) and all(step in before for step in steps):
            then.append(list(before))  # "add 19 once more": the same steps as last time, not only the one said
        elif steps:
            then.append(steps)
        elif _AGAIN.search(group) and then:
            then.append(list(then[-1]))
    asked = re.search(r"\b(?:play|pick|open|choose|do|try|visit|read|watch|cook|listen\s+to)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", text, re.I)
    more = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+more\b", text, re.I)
    picks = max(1 + len(then), _number(asked.group(1)) or 1 if asked else 1, 1 + (_number(more.group(1)) or 0) if more else 1)
    # Each next pick the person said nothing new for is taken the way the one before was.
    if then and len(then) < picks - 1:
        then += [list(then[-1])] * (picks - 1 - len(then))
    return PickingRule(start, first, then, counted_from, picks)
