"""A scorer that names the demo's words cannot evidence the demo.

The recall scorer carried three literal bonuses:

    if "fox" in lowered:    score += 4.0
    if "3:14" in lowered:   score += 2.5
    if "bryan" in lowered:  score += 1.5

Those were not arbitrary. The general overlap rule underneath them required
``len(token) > 3``, so "fox" scored NOTHING through the general path and
somebody made the demo work by naming it.

The cost was that the exact examples used to show memory working were the
ones the production scorer privileged. "What was that thing about the fox?"
answered correctly was not evidence of general retrieval — it was evidence
that the scorer had been told about foxes.

These tests pin two things: that no word is privileged, and that the DEFECT
which motivated the literals is actually fixed, since deleting them without
fixing it would just make short distinctive tokens invisible again.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.phases.response_generation_unitary import UnitaryResponsePhase

ROOT = Path(__file__).resolve().parents[1]
#: The scorer's own module. It left `response_generation_unitary` when that
#: file was split for size, and splitting the old source on a marker it no
#: longer holds raised IndexError instead of failing the check.
SOURCE = ROOT / "core" / "phases" / "unitary_memory_recall.py"


def _score(candidate: str, objective: str) -> float:
    return UnitaryResponsePhase._score_memory_candidate(candidate, objective)


# ─────────────────────────────────────────────── no word is privileged


@pytest.mark.parametrize(
    "word,rival",
    [
        ("fox", "otter"),
        ("fox", "elk"),
        ("bryan", "morgan"),
        ("bryan", "sam"),
    ],
)
def test_a_previously_named_word_scores_like_an_unnamed_peer(word, rival):
    """Swap the word for one nobody hardcoded; the score must not move.

    This is the property the literals broke. If "fox" outscores "otter" for
    the same sentence shape, the scorer is answering about foxes rather than
    about retrieval.
    """
    named = _score(f"remember the {word} at the door", f"what about the {word}?")
    unnamed = _score(f"remember the {rival} at the door", f"what about the {rival}?")

    assert named == pytest.approx(unnamed, abs=0.01), (
        f"{word!r} scored {named} but {rival!r} scored {unnamed}; the scorer "
        "privileges a specific word and cannot evidence general recall"
    )


def test_the_time_literal_is_not_special_either():
    named = _score("meet at 3:14 sharp", "what time was it?")
    unnamed = _score("meet at 9:42 sharp", "what time was it?")

    assert named == pytest.approx(unnamed, abs=0.01)


def test_no_literal_recall_bonus_survives_in_the_source():
    """Structural: a re-added literal must fail, not quietly work.

    Deleting the three was easy; the thing worth defending is that a fourth
    is never added under deadline. The check looks for a string-literal
    membership test that adds to the score, which is the shape all three had.
    """
    body = SOURCE.read_text("utf-8")
    scorer = body.split("def _score_memory_candidate", 1)[1].split("\n    @", 1)[0]

    offenders = re.findall(
        r'if\s+"([^"]{1,24})"\s+in\s+lowered\s*:\s*\n\s*score\s*\+=', scorer
    )
    # A small curated vocabulary of INTENT markers is legitimate — "remember"
    # and "forever" describe the kind of memory, not its subject. Words that
    # name a SUBJECT are the contamination.
    allowed = {"remember", "forever", "exact phrase", "phrase"}
    contaminating = [word for word in offenders if word not in allowed]

    assert not contaminating, (
        f"literal subject bonuses are back: {contaminating}. A scorer that "
        "names the words in the demo cannot be evidence for the demo."
    )


# ────────────────────────── the defect the literals were papering over


def test_a_short_distinctive_token_is_no_longer_invisible():
    """The reason "fox" was hardcoded: the length floor excluded it."""
    distinct = UnitaryResponsePhase._token_distinctiveness("fox")

    assert distinct > 0.0, (
        "a short, specific token still scores nothing through the general "
        "path — which is exactly the hole the hardcoded bonus was filling"
    )


def test_a_long_empty_word_scores_nothing():
    """Length was never the signal. "something" is 9 characters of nothing."""
    assert UnitaryResponsePhase._token_distinctiveness("something") == 0.0
    assert UnitaryResponsePhase._token_distinctiveness("about") == 0.0


def test_a_structured_token_outscores_an_ordinary_word():
    """"3:14" and "v2" are what people quote back verbatim."""
    structured = UnitaryResponsePhase._token_distinctiveness("3:14")
    ordinary = UnitaryResponsePhase._token_distinctiveness("meeting")

    assert structured > ordinary


def test_content_words_are_weighted_equally_regardless_of_length():
    """The first fix graded by length and a test caught the contradiction.

    Length is not distinctiveness — that is the whole argument for deleting
    the literals — so a gradient that pays more for longer words reintroduces
    an arbitrary ranking with no corpus behind it.
    """
    weights = {
        word: UnitaryResponsePhase._token_distinctiveness(word)
        for word in ("fox", "elk", "otter", "bryan", "extraordinary")
    }

    assert len(set(weights.values())) == 1, (
        f"content words are weighted unequally by length: {weights}"
    )


def test_no_single_token_can_dominate_the_score():
    """The old +4.0 let one word outweigh every structural signal."""
    worst = max(
        UnitaryResponsePhase._token_distinctiveness(token)
        for token in ("fox", "3:14", "bryan", "extraordinary", "v2", "412")
    )

    assert worst <= 4.0
    assert worst < 4.0, "a token bonus is as large as the literal it replaced"


def test_matching_more_distinctive_tokens_still_scores_higher():
    """The general mechanism has to actually work, not just be unbiased."""
    one = _score("the meeting was moved", "what about the meeting?")
    several = _score(
        "the meeting was moved to 9:42 in room 300b",
        "what about the meeting at 9:42 in room 300b?",
    )

    assert several > one
