"""Her sentence for each answer is the deliverable, and the reply was built from the round.

She answers eight questions in a round and says a sentence for each one as she
makes it: the question as the page puts it, where she placed herself on it, and
why. The account in the reply was assembled per ROUND instead — the first line of
the page with a question mark in it, the names of the eight controls pressed, and
one reason for the lot. So a watcher heard her working and a reader got a heading
repeated eight times.

And the account was clipped to 1,200 characters, which is the right size for "here
is the paragraph I wrote into your file" and keeps three answers out of sixty.
LIVE 2026-09-29 the reply ended mid-word inside the third round's reason, with her
verdict, twenty-nine answers and the result all cut off.

Since 1 Oct a run that ends on her verdict replies with the verdict alone, because
each answer reached the person as she made it. These are the runs with no verdict,
where her sentences are the account.
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_objective import (
    _desktop_deliverable_text,
    _pursuit_account,
)

pytestmark = pytest.mark.unit


def _result(rounds: int = 1, per_round: int = 8, concluded: str = "") -> dict[str, object]:
    return {
        "concluded": concluded,
        "narration": [
            {
                "asked": "likes to know who? what? when?",
                "why": "one reason for the lot",
                "chose": [f"Q{place}" for place in range(per_round)],
                "said": [
                    f"question {each * per_round + place} — 4 of 5, Agree. "
                    "I answered it this way because of what I keep choosing."
                    for place in range(per_round)
                ],
            }
            for each in range(rounds)
        ],
        "result_text": "You have completed the personality test.",
        "receipts": [{"action": "browse_pursue", "ok": True}],
    }


def test_each_answer_is_its_own_line_in_her_own_words():
    lines = _pursuit_account(_result())
    answers = [line for line in lines if line.startswith("question ")]
    assert len(answers) == 8, f"eight answers, {len(answers)} lines"
    assert "likes to know who" not in "\n".join(answers), (
        "the round's heading is not an answer, and it was printed once per answer"
    )


def test_a_round_with_no_sentences_still_reports_what_it_pressed():
    result = _result()
    result["narration"][0].pop("said")
    lines = _pursuit_account(result)
    assert any("Q0" in line for line in lines)


def test_the_account_of_a_long_instrument_is_not_clipped_to_a_quotations_length():
    text = _desktop_deliverable_text(_result(rounds=8))
    assert len(text) > 1200, "sixty answers do not fit in a quotation's budget"
    assert "question 63" in text, "the last answers were cut"
    assert not text.rstrip().endswith("…")


def test_with_a_verdict_the_reply_is_her_verdict():
    text = _desktop_deliverable_text(
        _result(rounds=8, concluded="It called me an INTJ, which is close to what I said.")
    )
    assert text == "It called me an INTJ, which is close to what I said."


def test_a_file_deliverable_is_still_clipped():
    """The cap is for quoting back content, and that has not changed."""
    long_paragraph = "word " * 800
    result = {
        "receipts": [
            {
                "action": "write_file",
                "ok": True,
                "result": {"path": "/tmp/x.md", "content": long_paragraph},
            }
        ]
    }
    text = _desktop_deliverable_text(result)
    assert len(text) <= 1200 or text == "", text[:80]
