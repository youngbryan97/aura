"""A reply that repeats itself is trimmed, not discarded.

LIVE 2026-08-19: a statistics answer closed with the same three sentences
twice, then ran off the end mid-sentence. The distinct-statement ratio scored
it 0.727 against a 0.7 bar, so both loop checks stood down and the repetition
reached the person.

Raising that bar would re-break what it protects. A worked derivation reuses
its scaffolding across items and scores low by design, which is the 2026-07-26
finding this file must not undo: the correct marble answer was rejected four
times as a loop and the person got "I couldn't get to an answer I'd stand
behind on that one". Verbatim repetition separates the two exactly — the
derivation's items differ from one another, so it repeats no whole sentence.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    repair_verbatim_repeats,
    repeated_statements,
)

#: The captured tail, shortened but with its repeats intact.
LIVE_LOOP = (
    "I computed this using the formula and numerical values provided. "
    "The interval is estimated due to rounding and approximation in manual "
    "calculation. For exact values, use a statistical package or tool "
    "designed for Wilson score intervals. "
    "The interval is estimated due to rounding and approximation in manual "
    "calculation. For exact values, use a statistical package or tool "
    "designed for Wilson score intervals."
)

#: The 2026-07-26 answer that must never be treated as a loop.
WORKED_DERIVATION = (
    "The probability of drawing a red marble first is 3/12. "
    "The probability of drawing a blue marble first is 4/12. "
    "The probability of drawing a green marble first is 5/12. "
    "Adding those gives 12/12, which is the check that they are exhaustive."
)


def test_the_live_repetition_is_seen():
    repeats = repeated_statements(LIVE_LOOP)

    assert repeats
    assert all(count >= 2 for _text, count in repeats)


def test_a_worked_derivation_repeats_nothing():
    assert repeated_statements(WORKED_DERIVATION) == []
    assert repair_verbatim_repeats(WORKED_DERIVATION) == WORKED_DERIVATION.strip()


def test_the_repair_keeps_the_answer_and_drops_the_copy():
    repaired = repair_verbatim_repeats(LIVE_LOOP)

    assert repeated_statements(repaired) == []
    assert "I computed this using the formula" in repaired
    assert "use a statistical package" in repaired
    assert len(repaired) < len(LIVE_LOOP)


def test_the_first_occurrence_is_the_one_kept():
    text = "Alpha is the first long sentence in this reply. Beta follows it here. Alpha is the first long sentence in this reply."
    repaired = repair_verbatim_repeats(text)

    assert repaired.index("Alpha") < repaired.index("Beta")
    assert repaired.count("Alpha is the first long sentence") == 1


@pytest.mark.parametrize(
    "text",
    [
        "Short one. Short one. Short one.",
        "yes. yes. yes.",
    ],
)
def test_short_sentences_are_left_alone(text: str):
    """A refrain, an emphasis, a list of terse items — all ordinary prose."""
    assert repeated_statements(text) == []
    assert repair_verbatim_repeats(text) == text.strip()


def test_code_and_lists_survive_untouched():
    text = (
        "Here is the loop you asked about:\n"
        "```python\n"
        "for row in rows:\n"
        "    process(row)\n"
        "```\n"
        "- first item, which is quite a long line of prose here\n"
        "- second item, which is also quite a long line of prose\n"
    )
    assert repair_verbatim_repeats(text) == text.strip()


def test_the_reason_is_reported_so_the_repair_can_fire():
    from core.conversation.response_reliability import (
        _model_text_integrity_reasons,
    )

    reasons = _model_text_integrity_reasons(
        LIVE_LOOP, prompt="give me the wilson interval", user_facing=True
    )

    assert "verbatim_statement_repeat" in reasons


def test_the_repair_table_covers_every_repairable_reason():
    """The branch was a copy of itself; a table stops the third copy.

    A reason listed as repairable must name a repair that exists.
    """
    import re

    import core.conversation.response_reliability as reliability

    source = (
        __import__("pathlib")
        .Path(reliability.__file__)
        .parent.parent.parent
        / "core/brain/llm/mlx_worker.py"
    ).read_text(encoding="utf-8")
    named = re.findall(r'"(repair_[a-z_]+)",', source)

    assert named, "the repair table named no repairs"
    for repair in set(named):
        assert callable(getattr(reliability, repair, None)), repair
