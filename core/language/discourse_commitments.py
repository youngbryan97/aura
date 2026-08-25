"""Mechanical commitments a generated answer makes about its remaining shape.

This is part of Aura's existing language substrate. It does not judge whether
an answer is good; it records structure the answer itself declared. Once a
reply says that two reasons follow, emitting only item 1 cannot be a terminal
state even when that item ends with a period.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["DiscourseCommitment", "discourse_commitments", "unfulfilled_commitments"]


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_COUNT_PATTERN = r"(?:[1-9]\d?|" + "|".join(_NUMBER_WORDS) + r")"
_KIND_PATTERN = (
    r"(?:answers?|causes?|cases?|checks?|conditions?|explanations?|factors?|"
    r"issues?|items?|options?|outcomes?|parts?|points?|possibilities|problems?|"
    r"reasons?|scenarios?|sources?|steps?|things?|ways?)"
)
_MODIFIER_PATTERN = (
    r"(?:(?:main|likely|possible|plausible|distinct|specific|independent|"
    r"remaining|common|candidate|potential|primary|major|key|separate)\s+)*"
)

# A declaration needs an explicit discourse cue and a colon. That boundary is
# what separates "there are two reasons:" from ordinary prose such as "I
# compared two reasons and chose one", which does not promise a list.
_COMMITMENT_RE = re.compile(
    rf"\b(?:"
    rf"(?:(?:one|any|all)\s+of\s+)(?P<of_count>{_COUNT_PATTERN})\s+"
    rf"{_MODIFIER_PATTERN}(?P<of_kind>{_KIND_PATTERN})"
    rf"|there\s+(?:are|were|remain)\s+(?P<there_count>{_COUNT_PATTERN})\s+"
    rf"{_MODIFIER_PATTERN}(?P<there_kind>{_KIND_PATTERN})"
    rf"|the\s+following\s+(?P<following_count>{_COUNT_PATTERN})\s+"
    rf"{_MODIFIER_PATTERN}(?P<following_kind>{_KIND_PATTERN})"
    rf")\s*:",
    re.IGNORECASE,
)
_NUMBERED_ITEM_RE = re.compile(
    r"(?m)(?:^|(?<=:))\s*(?:\*\*)?(?P<index>[1-9]\d?)[.)](?:\*\*)?\s+\S"
)
_BULLET_ITEM_RE = re.compile(r"(?m)^\s*[-*+]\s+\S")
_ORDINAL_ITEM_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth)(?:\*\*)?\s*[:,.)-]\s+\S"
)


@dataclass(frozen=True)
class DiscourseCommitment:
    """A declared answer shape and the items visible after its declaration."""

    expected_count: int
    observed_count: int
    kind: str
    declaration: str
    start: int

    @property
    def fulfilled(self) -> bool:
        return self.observed_count >= self.expected_count


def _count(value: str) -> int:
    lowered = str(value or "").casefold()
    if lowered.isdigit():
        return int(lowered)
    return _NUMBER_WORDS.get(lowered, 0)


def _observed_item_count(tail: str) -> int:
    numbered = [int(match.group("index")) for match in _NUMBERED_ITEM_RE.finditer(tail)]
    if numbered:
        # A duplicated item number is still one item. Gaps do not get credited
        # as invisible content: 1 then 3 demonstrates two authored items, not
        # three completed obligations.
        return len(set(numbered))
    bullets = len(_BULLET_ITEM_RE.findall(tail))
    if bullets:
        return bullets
    return len(_ORDINAL_ITEM_RE.findall(tail))


def discourse_commitments(text: object) -> tuple[DiscourseCommitment, ...]:
    """Return list-shaped commitments declared by ``text``.

    Prose without an actual item marker is left alone. The language may mention
    a finite choice inline ("one of two values: yes or no") without promising a
    multi-line answer shape; the presence of item markers makes the promise
    mechanically observable.
    """

    body = str(text or "")
    commitments: list[DiscourseCommitment] = []
    matches = list(_COMMITMENT_RE.finditer(body))
    for position, match in enumerate(matches):
        count_text = next(
            (
                match.group(name)
                for name in ("of_count", "there_count", "following_count")
                if match.group(name)
            ),
            "",
        )
        kind = next(
            (
                match.group(name)
                for name in ("of_kind", "there_kind", "following_kind")
                if match.group(name)
            ),
            "",
        )
        expected = _count(count_text)
        tail_end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        tail = body[match.end() : tail_end]
        observed = _observed_item_count(tail)
        if expected <= 1 or observed <= 0:
            continue
        commitments.append(
            DiscourseCommitment(
                expected_count=expected,
                observed_count=observed,
                kind=kind.casefold(),
                declaration=match.group(0).strip(),
                start=match.start(),
            )
        )
    return tuple(commitments)


def unfulfilled_commitments(text: object) -> tuple[DiscourseCommitment, ...]:
    """Return answer-shape declarations whose items have not all arrived."""

    return tuple(commitment for commitment in discourse_commitments(text) if not commitment.fulfilled)
