"""The rating scale a message asks to be answered on, and whether a reply answers on it.

A question that names its scale also names the shape of its answer. Asked how
she feels "from -1 (very bad) to 1 (very good)", a number on that scale is the
answer, and a check that wants the answer in words throws it away.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["answers_the_scale_it_was_asked_for", "asked_scale"]

#: A rating scale the person names: "from -1 (very bad) to 1 (very good)",
#: "on a scale of 1 to 10", "between 0 and 5", "a 0-10 scale".
_ASKED_SCALE_RE = re.compile(
    r"(?:\bfrom|\bscale\s+of|\bbetween)\s+([+\-−]?\d+(?:\.\d+)?)\s*(?:\([^)]{0,40}\)\s*)?"
    r"(?:to|and|-|–|—)\s*([+\-−]?\d+(?:\.\d+)?)"
    r"|\b([+\-−]?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*([+\-−]?\d+(?:\.\d+)?)\s+scale\b",
    re.IGNORECASE,
)
_ASKED_OUT_OF_RE = re.compile(r"\bout\s+of\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_REPLY_NUMBER_RE = re.compile(r"(?<![\w.])([+\-−]?(?:\d+(?:\.\d+)?|\.\d+))")


def asked_scale(prompt: Any) -> tuple[float, float] | None:
    """The rating scale a message asks to be answered on, low end first, or None."""
    text = str(prompt or "")
    found = _ASKED_SCALE_RE.search(text)
    if found:
        ends = [value for value in found.groups() if value is not None]
        try:
            low, high = sorted(float(value.replace("−", "-")) for value in ends[:2])
        except ValueError:
            return None
        return (low, high) if low < high else None
    out_of = _ASKED_OUT_OF_RE.search(text)
    if out_of:
        high = float(out_of.group(1))
        return (0.0, high) if high > 0.0 else None
    return None


def answers_the_scale_it_was_asked_for(prompt: Any, reply_text: Any) -> bool:
    """Whether the reply gives a number on the scale the message asked for.

    Asked how she feels "from -1 (very bad) to 1 (very good)", her cortex
    answered in six or seven tokens on every arm of the reports experiment of
    24 September, and every answer was thrown away as a self-condition reply
    with no self-condition in it: the check wanted a first-person pronoun and a
    word such as "steady". The repair then put a sentence composed from her
    telemetry in its place, and that sentence was read as her report. A
    number on the scale the person named is the answer to the question.
    """
    scale = asked_scale(prompt)
    if scale is None:
        return False
    low, high = scale
    for match in _REPLY_NUMBER_RE.finditer(str(reply_text or "")):
        try:
            value = float(match.group(1).replace("−", "-"))
        except ValueError:
            continue
        if low <= value <= high:
            return True
    return False
