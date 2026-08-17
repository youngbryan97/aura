"""A number about herself has to come from a channel, or it does not get said.

LIVE 2026-08-17. Asked how she was doing, she answered:

    "My memory stores are at 87% capacity and my computational resources are
    underutilized, with only minimal f..."

There is no memory-store capacity metric in this codebase. Not a stale one, not
a renamed one — ``memory_capacity``, ``memory_stores``, ``store_capacity`` and
``memory_utilization`` appear nowhere. The 87% was invented, stated flatly, and
sat beside a real reading in the same sentence where nothing marked one from
the other.

Her *action* claims already have a guard (``claimed_effect``), because the
vocabulary of effects is finite and closable. Her *sensory* claims have one.
Quantitative claims about her own state had none, and they are the easiest kind
to invent: a plausible percentage costs nothing to generate and reads as
instrumentation.

The closure argument that makes this tractable is the same one that made
EffectClaim work, pointed at a different set. The phrasings she can produce are
open — "87% capacity", "about seven eighths full", "nearly at my limit". The
CHANNELS that could ground any of them are closed: a number about herself is
either traceable to something the system actually samples, or it is not. So
this does not try to recognise every way of saying a number. It extracts the
number and the thing it is predicated of, then asks the evidence layer whether
any channel could have produced it.

What it deliberately does NOT do:

  * ban her from discussing her state. An ungrounded number is replaced by the
    honest typed absence, not by silence, and qualitative description is
    untouched — "I feel flat" is not a measurement and needs no channel.
  * treat an unmatched subject as a lie. ABSENT_NOT_INSTRUMENTED is a real
    answer: it says the channel does not exist, which is exactly what was true
    of the memory stores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SelfMetricClaim",
    "self_metric_claims",
    "unsourced_self_metric_claims",
]

# Tokens that carry no discriminating power when matching a spoken subject
# against a channel name. "memory stores" must not match a channel merely
# because both contain "of".
_STOPWORDS = frozenset(
    {
        "a", "all", "and", "are", "at", "available", "current", "currently",
        "is", "my", "of", "own", "reading", "readings", "state", "the", "to",
        "total", "value", "values", "with",
    }
)

# A number that is being predicated of something belonging to her. The subject
# is captured lazily backwards from the number so "my memory stores are at 87%"
# yields ("memory stores", "87%") and not the whole clause.
_CLAIM_RE = re.compile(
    r"\bmy\s+(?P<subject>[a-z][a-z \-]{2,48}?)\s+"
    r"(?:is|are|sits?|sit|stands?|runs?|reads?)\s+"
    r"(?:at\s+|around\s+|about\s+|roughly\s+|nearly\s+)?"
    r"(?P<quantity>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|gb|mb|tb|kb)?",
    re.IGNORECASE,
)

# The inverted order: "87% of my memory", "12GB of my working set".
_INVERTED_RE = re.compile(
    r"(?P<quantity>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|gb|mb|tb|kb)?\s+"
    r"(?:capacity\s+)?(?:of\s+)?my\s+(?P<subject>[a-z][a-z \-]{2,48}?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SelfMetricClaim:
    """One quantity she stated about herself, and what she predicated it of."""

    subject: str
    quantity: str
    unit: str
    span: str

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(
            t for t in re.split(r"[^a-z0-9]+", self.subject.lower()) if t and t not in _STOPWORDS
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.subject} = {self.quantity}{self.unit}"


def _normalise(text: Any) -> str:
    return str(text or "")


def self_metric_claims(reply: Any) -> tuple[SelfMetricClaim, ...]:
    """Every numeric claim the reply makes about her own state."""

    text = _normalise(reply)
    if not text:
        return ()
    found: list[SelfMetricClaim] = []
    seen: set[tuple[str, str]] = set()
    for pattern in (_CLAIM_RE, _INVERTED_RE):
        for match in pattern.finditer(text):
            subject = " ".join(match.group("subject").split()).strip(" -")
            quantity = match.group("quantity")
            unit = (match.group("unit") or "").strip()
            if not subject or not quantity:
                continue
            key = (subject.lower(), quantity)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                SelfMetricClaim(
                    subject=subject,
                    quantity=quantity,
                    unit="%" if unit.lower() in {"%", "percent"} else unit,
                    span=match.group(0).strip(),
                )
            )
    return tuple(found)


def _channel_tokens(channel: Any) -> frozenset[str]:
    return frozenset(
        t
        for t in re.split(r"[^a-z0-9]+", str(channel or "").lower())
        if t and t not in _STOPWORDS
    )


def unsourced_self_metric_claims(
    reply: Any,
    readings: Any = None,
) -> tuple[SelfMetricClaim, ...]:
    """Claims whose subject matches no channel that actually produced a value.

    `readings` is any iterable of objects carrying `.channel` and `.present`
    (an EvidenceBundle's `.readings`, or the bundle itself). A claim survives
    only if some PRESENT channel shares a discriminating token with its
    subject: a channel that exists but read nothing cannot source a number
    either, which is the ABSENT_NEVER_SAMPLED case.
    """

    claims = self_metric_claims(reply)
    if not claims:
        return ()

    candidates = getattr(readings, "readings", readings) or ()
    grounded: list[frozenset[str]] = []
    for reading in candidates:
        try:
            if not bool(getattr(reading, "present", False)):
                continue
            tokens = _channel_tokens(getattr(reading, "channel", ""))
        except (AttributeError, TypeError, ValueError):
            continue
        if tokens:
            grounded.append(tokens)

    unsourced: list[SelfMetricClaim] = []
    for claim in claims:
        subject_tokens = claim.tokens
        if not subject_tokens:
            continue
        if any(subject_tokens & channel for channel in grounded):
            continue
        unsourced.append(claim)
    return tuple(unsourced)
