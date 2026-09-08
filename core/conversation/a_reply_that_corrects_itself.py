"""A reply that retracts its own opening must not still lead with it.

LIVE, 2026-09-08. Asked how much daylight 45°N loses between the solstice and
the equinox, Aura published:

    Roughly **105 minutes**, or about 1 hour and 45 minutes.
    ...
    My previous "105" was a mental slip (halving the difference
    incorrectly). The correct rough figure is **210 minutes**.

Both sentences were hers and the second one is right. A reader who reads the
first line and stops — which is most readers, and every audience watching a
demonstration — takes away a number she had already withdrawn.

This is not about the words. The reply states a fact, states that the fact was
wrong, and states the replacement, and the delivered text carries all three
with the withdrawn one at the top. The repair strikes the withdrawn opening
through and changes nothing else: no value is rewritten, because rewriting one
means recomputing the rest of her sentence, and that is inventing an answer
rather than delivering hers. A reply that retracts nothing is untouched.

Deliberately narrow. It fires only when every part is present — a retraction
that names a value, that value standing earlier in the reply, and a
replacement given in the same breath — because a repair that guesses is worse
than the contradiction it is guessing about.
"""

from __future__ import annotations

import re
from typing import NamedTuple

#: How a writer marks a retraction of something they just said. Each of these
#: has to be followed by the structure as well: a value that was already
#: stated, and a value offered in its place. The words alone decide nothing.
_A_RETRACTION = re.compile(
    r"\b(?:"
    r"was a (?:mental )?(?:slip|mistake|error)"
    r"|i (?:mis-?(?:calculated|stated|spoke|read)|made an error|got that wrong)"
    r"|(?:that|this|my previous|my earlier|the) (?:figure|number|answer|value|estimate)"
    r"\s+(?:above\s+)?(?:was|is) (?:wrong|incorrect|a mistake|off)"
    r"|correction:"
    r"|scratch that"
    r"|(?:the )?correct(?:ed)? (?:rough )?(?:figure|number|answer|value|total)"
    r"|revised (?:conclusion|answer|figure|total)"
    r")\b",
    re.IGNORECASE,
)

#: A number as a reader meets it: 210, 1,024, 15.5, optionally signed.
_A_VALUE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![\w.])")

#: The opening claim: everything up to the first blank line, capped. Not a
#: fixed slice of the front — a 400-character window swallowed the start of
#: the working, so a value from the working was read as a value from the
#: headline and the repair rewrote the wrong number.
_THE_OPENING_CLAIM = re.compile(r"\A.*?(?=\n\s*\n|\Z)", re.DOTALL)

#: The most of an opening claim this will look at or rewrite.
_AS_FAR_AS_A_HEADLINE_GOES = 320

#: From a retraction marker to the replacement it offers. A writer names the
#: old value and the new one in the same breath; two sentences later is
#: somebody else's sentence.
_THE_SAME_BREATH = 260


class ACorrectedReply(NamedTuple):
    text: str
    superseded: str
    corrected: str
    reason: str


def _as_number(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except (TypeError, ValueError):
        return None


def the_reply_corrects_its_own_headline(text: object) -> ACorrectedReply | None:
    """The correction this reply makes to its own opening, if it makes one.

    Returns the reply with its withdrawn opening struck through, naming the
    value it withdrew and the one it gave instead. None when the reply never
    retracts anything it led with.
    """

    body = str(text or "")
    if not body.strip():
        return None
    opening = _THE_OPENING_CLAIM.match(body)
    headline = (opening.group(0) if opening else "")[:_AS_FAR_AS_A_HEADLINE_GOES]
    if not headline.strip():
        return None
    stated_up_front = {m.group(0) for m in _A_VALUE.finditer(headline)}
    if not stated_up_front:
        return None

    for marker in _A_RETRACTION.finditer(body, len(headline)):
        # Back to the start of the marker's own sentence, and no further. The
        # withdrawn value is named before the marker — `My previous "105" was
        # a slip` — but a fixed look-back reached into the working above it
        # and read "45 degrees north" as the value being withdrawn.
        opened_at = max(
            (
                body.rfind(mark, len(headline), marker.start())
                for mark in (". ", ".\n", "\n", "! ", "? ")
            ),
            default=-1,
        )
        breath = body[
            (opened_at + 1 if opened_at > 0 else len(headline)) : (
                marker.start() + _THE_SAME_BREATH
            )
        ]
        values = [m.group(0) for m in _A_VALUE.finditer(breath)]
        superseded = next((v for v in values if v in stated_up_front), "")
        if not superseded:
            continue
        was = _as_number(superseded)
        if was is None:
            continue
        replacing = ""
        for candidate in values:
            now = _as_number(candidate)
            if now is None or candidate == superseded or now == was:
                continue
            replacing = candidate
            break
        if not replacing:
            continue
        # Struck through, not rewritten.
        #
        # Substituting the new value into the opening sentence was the first
        # version and it was wrong twice over: it picked "3.5" out of "(3.5
        # hours)" as the replacement, and even with the right number the rest
        # of the sentence — "or about 1 hour and 45 minutes" — still restated
        # the withdrawn one. Repairing that means recomputing her answer,
        # which is inventing one.
        #
        # So nothing is removed and nothing is added. The claim she withdrew
        # is marked as withdrawn, in her own words, and her correction stands
        # where she put it.
        return ACorrectedReply(
            text=f"~~{headline.strip()}~~" + body[len(headline) :],
            superseded=superseded,
            corrected=replacing,
            reason="the_reply_withdrew_its_own_opening_value",
        )
    return None
