"""The sums an answer does on its own numbers, recomputed.

``arithmetic_check`` recomputes the arithmetic a PERSON asked for — "what is
7919 times 6421" — and serves the computed value. That covers the question and
not the answer: the sums Aura performs inside a worked reply have never been
checked by anything.

LIVE, 2026-09-08. Asked for the daylight lost at 45°N between the solstice and
the equinox, she wrote the hour-angle formula correctly and then read the wrong
number off it three times running — 15h02m, 15h08m, 15h10m, where the formula
she had just stated gives 15h26m. One of those runs subtracted correctly from a
wrong input; another subtracted 926 − 720 and announced 105.

This never rejects a reply and never rewrites one. It recomputes the steps the
reply states, and hands back the ones that do not hold so the delivery can say
so in a line of its own. A wrong sum inside a good answer is worth a sentence,
not a refusal — and Bryan asked for exactly that: "shouldnt reject the whole
response ever."

Deliberately narrow. It checks arithmetic the reply spells out with both
operands and a result, in the units it spelled them out in. It does not check
a physical claim ("day length at 45°N is 15h10m"), because recomputing that
means owning a model of the world, and a checker that guesses is worse than
the error it is guessing about.
"""

from __future__ import annotations

import re
from typing import NamedTuple

#: `926 - 720 = 206`, `15.43 x 60 = 926`, `2 x 115.7 / 15 = 15.43`.
#:
#: The WHOLE expression, however many operands it has, and then the result.
#: A two-operand pattern read `115.7/15` out of `2 x 115.7/15 = 15.4` and
#: reported that a correct line was wrong by a factor of two — which is worse
#: than not checking, because a checker that fires on good arithmetic teaches
#: everyone to ignore it.
_AN_OPERAND = r"-?\d[\d,]*(?:\.\d+)?"
_AN_OPERATOR = r"[-+*/x×÷−–]"
_A_STATED_SUM = re.compile(
    r"(?<![\w.,])"
    rf"({_AN_OPERAND}(?:\s*{_AN_OPERATOR}\s*{_AN_OPERAND})+)\s*"
    r"(?:=|is|gives|comes to|equals|→|->)\s*"
    r"(?:about\s+|roughly\s+|approximately\s+|~\s*|≈\s*)?"
    rf"({_AN_OPERAND})(?![\d.]*\s*{_AN_OPERATOR}\s*\d)",
    re.IGNORECASE,
)

#: An expression this will evaluate: digits, the four operations, spaces.
#: Nothing else is admitted, so nothing else can be executed.
_ONLY_ARITHMETIC = re.compile(r"\A[\d\s.+*/x×÷−–()-]+\Z")

#: `15 h 10 m`, `15 hours 10 minutes`, `12h 0m`, `3h 8m`.
_A_DURATION = re.compile(
    r"(?<![\w.])(\d{1,3})\s*(?:h|hr|hrs|hour|hours)\s*"
    r"(?:and\s*)?(\d{1,2})?\s*(?:m|min|mins|minute|minutes)?(?![\w.])",
    re.IGNORECASE,
)

#: `15h 10m - 12h 0m = 3h 10m`, written across the units.
_A_DURATION_SUM = re.compile(
    r"(\d{1,3}\s*(?:h|hr|hrs|hour|hours)[^=\n]{0,24}?)"
    r"\s*(?:-|−|–|minus|less)\s*"
    r"(\d{1,3}\s*(?:h|hr|hrs|hour|hours)[^=\n]{0,24}?)"
    r"\s*(?:=|is|gives|comes to|equals|→|->)\s*"
    r"(?:about\s+|roughly\s+|approximately\s+|~\s*|≈\s*)?"
    r"(\d{1,3}\s*(?:h|hr|hrs|hour|hours)[^.\n]{0,24}|\d[\d,]*(?:\.\d+)?\s*(?:minutes|mins|min))",
    re.IGNORECASE,
)

#: How the reply writes an operator, and what Python calls it.
_AS_PYTHON_WRITES_IT = {"x": "*", "×": "*", "÷": "/", "−": "-", "–": "-"}

#: How far a stated result may sit from the computed one before it is wrong.
#: A reply that says "roughly" has said it is rounding; one that does not has
#: still rounded its own intermediate values, so the floor is not zero.
_CLOSE_ENOUGH = 0.005
_CLOSE_ENOUGH_WHEN_HEDGED = 0.05
_A_HEDGE = re.compile(
    r"\b(?:roughly|approximately|about|around|circa|order of|ballpark)\b|~|≈",
    re.IGNORECASE,
)


class ASumThatDoesNotHold(NamedTuple):
    stated: str
    expected: str
    said: str


def _as_number(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _as_minutes(text: str) -> float | None:
    plain = _as_number(re.sub(r"[^\d.,-]", "", str(text))) if "h" not in str(text).lower() else None
    if plain is not None and re.search(r"\b(?:minutes|mins|min)\b", str(text), re.I):
        return plain
    match = _A_DURATION.search(str(text))
    if not match:
        return None
    hours = _as_number(match.group(1))
    minutes = _as_number(match.group(2)) if match.group(2) else 0.0
    if hours is None:
        return None
    return hours * 60.0 + (minutes or 0.0)


def _value_of(expression: str) -> float | None:
    """What the reply's own expression comes to, with ordinary precedence.

    Parsed as an arithmetic literal and nothing else: the admitted characters
    are digits, spaces, brackets and the four operations, checked before the
    string is compiled, and the compiled tree is walked to confirm it holds
    only numbers and arithmetic operators. A reply is untrusted text.
    """

    import ast

    plain = str(expression or "").replace(",", "")
    for written, python in _AS_PYTHON_WRITES_IT.items():
        plain = plain.replace(written, python)
    plain = plain.strip()
    if not plain or not _ONLY_ARITHMETIC.match(plain):
        return None
    try:
        tree = ast.parse(plain, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        value = eval(compile(tree, "<the reply's own sum>", "eval"))  # noqa: S307
    except (ArithmeticError, ValueError, TypeError, OverflowError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _is_wrong(expected: float, said: float, *, hedged: bool) -> bool:
    scale = max(abs(expected), abs(said), 1.0)
    tolerance = (_CLOSE_ENOUGH_WHEN_HEDGED if hedged else _CLOSE_ENOUGH) * scale
    return abs(expected - said) > tolerance


def _pretty(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.4g}"


def sums_that_do_not_hold(text: object) -> tuple[ASumThatDoesNotHold, ...]:
    """Every arithmetic step this reply states that its own numbers refute."""

    body = str(text or "")
    if not body.strip():
        return ()
    hedged = bool(_A_HEDGE.search(body))
    wrong: list[ASumThatDoesNotHold] = []

    for match in _A_STATED_SUM.finditer(body):
        expression, said_text = match.groups()
        expected = _value_of(expression)
        said = _as_number(said_text)
        if expected is None or said is None:
            continue
        if _is_wrong(expected, said, hedged=hedged):
            wrong.append(
                ASumThatDoesNotHold(
                    stated=match.group(0).strip(),
                    expected=_pretty(expected),
                    said=_pretty(said),
                )
            )

    for match in _A_DURATION_SUM.finditer(body):
        first, second, said_text = match.groups()
        a, b, said = _as_minutes(first), _as_minutes(second), _as_minutes(said_text)
        if a is None or b is None or said is None:
            continue
        expected = a - b
        if _is_wrong(expected, said, hedged=hedged):
            wrong.append(
                ASumThatDoesNotHold(
                    stated=match.group(0).strip(),
                    expected=f"{_pretty(expected)} minutes",
                    said=f"{_pretty(said)} minutes",
                )
            )

    # One line, not a list of twenty: a reply that restates the same step in
    # three notations should be reported once.
    seen: set[tuple[str, str]] = set()
    unique: list[ASumThatDoesNotHold] = []
    for item in wrong:
        key = (item.expected, item.said)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def a_note_about_the_arithmetic(text: object) -> str:
    """One sentence naming the steps that do not hold, or empty when they do.

    Appended to a reply; never a replacement for one.
    """

    wrong = sums_that_do_not_hold(text)
    if not wrong:
        return ""
    if len(wrong) == 1:
        one = wrong[0]
        return (
            f"Checking my own arithmetic: “{one.stated}” does not hold — "
            f"that comes to {one.expected}, not {one.said}."
        )
    steps = "; ".join(f"“{item.stated}” comes to {item.expected}" for item in wrong[:3])
    return f"Checking my own arithmetic, {len(wrong)} steps do not hold: {steps}."
