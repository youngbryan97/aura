"""A contraction is not a quotation.

LIVE, 2026-08-18. Asked to remember a number — "remember the number 7419 for
me. i'm going to ask you about it later in this conversation, and in between
i'll talk about other things" — she dispatched a program-reconstruction skill
whose target was:

    m going to ask you about it later in this conversation, and in between i

That is the sentence between the apostrophe in "i'm" and the one in "i'll".
The extractor's character class was `["“”']`, so a straight apostrophe opened
a quote and the next one closed it. Any English containing two contractions
produces a bogus quoted argument, and possessives do the same.

`core/conversation/screen_reading_claim.py` already had the right form — the
delimiters guarded by `(?<![A-Za-z])` and `(?![A-Za-z])`. Four other sites did
not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.agency.autonomous_task_engine import AutonomousTaskEngine

ROOT = Path(__file__).resolve().parents[1]

#: The message that produced the mangled target.
LIVE_TURN = (
    "i want to test something about time. remember the number 7419 for me. "
    "i'm going to ask you about it later in this conversation, and in between "
    "i'll talk about other things."
)


def test_the_turn_that_produced_a_mangled_target_yields_nothing():
    assert AutonomousTaskEngine._extract_quoted_text(LIVE_TURN) == ""


@pytest.mark.parametrize(
    "text",
    [
        "i'm going to ask and i'll wait",
        "don't do that, it won't help",
        "open Bryan's file and Dana's note",
        "it's fine, she'd rather we didn't",
    ],
)
def test_contractions_and_possessives_are_not_quotes(text):
    assert AutonomousTaskEngine._extract_quoted_text(text) == "", text


@pytest.mark.parametrize(
    "text,expected",
    [
        ('type "hello world" into the box', "hello world"),
        ("type 'hello world' into the box", "hello world"),
        ("write “the report” now", "the report"),
        ("i'm typing 'save now' because i'll forget", "save now"),
    ],
)
def test_real_quotations_still_extract(text, expected):
    """Including a genuine quote sitting between two contractions."""
    assert AutonomousTaskEngine._extract_quoted_text(text) == expected


def test_no_extractor_treats_a_bare_apostrophe_as_a_delimiter():
    """The same character class appeared at five sites; one was already right.

    A fix applied to the site that happened to be reported, and not to the
    four identical ones, is how the next report gets filed against a different
    file for the same reason.
    """
    offenders: list[str] = []
    for path in ROOT.joinpath("core").rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        for line in source.splitlines():
            if "“”'" not in line:
                continue
            # Guarded forms are fine; an unguarded class is not.
            if "(?<![A-Za-z])" in line or "(?![A-Za-z])" in line:
                continue
            if re.search(r"\[[^\]]*\\?\"[^\]]*“”'[^\]]*\]", line):
                offenders.append(f"{path.relative_to(ROOT)}: {line.strip()[:90]}")
    assert not offenders, "apostrophe treated as a quote delimiter:\n" + "\n".join(offenders)
