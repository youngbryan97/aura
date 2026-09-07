"""Missing bookkeeping is not evidence the answer is wrong.

LIVE 2026-09-07: "what was the first thing I said to you in this conversation?"
The cortex answered. `quality_metrics` recorded confidence=high, off_topic=False,
stale=False, reply_len=175. The turn was then failed closed and the person got
"I couldn't get my full attention onto that one, and I'd rather tell you that
than answer from half of it."

The proof that was missing was `authored_answer_incomplete:retry_exhausted` — a
fact about a retry counter, not about the text. `chat_turn_contract` already
separates the proofs that say the ANSWER is unfinished from the ones that say a
receipt was never bound, in a comment written after the same shape appeared on
2026-08-29, and nothing acted on the difference.

A gate nobody can pass takes a working system and gates it into a coma. This is
the other direction: the answer is served, disclosed as bounded, and the
withholding is kept for the case where the text itself is unfinished.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _a_proof_that_says_the_answer_is_unfinished


@pytest.mark.parametrize(
    "proof",
    [
        "authored_answer_incomplete:generation_cut_off",
        "authored_answer_incomplete:semantically_short",
        "authored_answer_incomplete:semantic_contract_unmet",
    ],
)
def test_a_proof_about_the_text_still_withholds(proof: str) -> None:
    assert _a_proof_that_says_the_answer_is_unfinished((proof,)) is True


@pytest.mark.parametrize(
    "proof",
    [
        "authored_answer_incomplete:retry_exhausted",
        "authored_answer_incomplete:nobody_checked",
        "live_mind_controls_unbound",
        "architecture_context_unbound",
        "live_mind_snapshot_unbound",
    ],
)
def test_a_proof_about_the_receipts_does_not(proof: str) -> None:
    assert _a_proof_that_says_the_answer_is_unfinished((proof,)) is False


def test_one_proof_about_the_text_outranks_any_number_about_receipts() -> None:
    assert _a_proof_that_says_the_answer_is_unfinished(
        (
            "authored_answer_incomplete:retry_exhausted",
            "live_mind_controls_unbound",
            "authored_answer_incomplete:generation_cut_off",
        )
    ) is True


def test_an_unclassified_proof_is_treated_as_being_about_the_answer() -> None:
    """A new proof nobody has classified must not silently serve something."""
    assert _a_proof_that_says_the_answer_is_unfinished(("something_new",)) is True


def test_no_missing_proof_at_all_is_not_a_reason_to_serve() -> None:
    """This helper is only consulted when the contract is already unproven."""
    assert _a_proof_that_says_the_answer_is_unfinished(()) is True


def test_the_route_consults_it_before_replacing_the_reply() -> None:
    from pathlib import Path

    source = Path("interface/routes/chat.py").read_text()
    start = source.index("def _fail_closed_on_an_unproven_full_mind_contract(")
    end = source.index("async def _serve_the_bounded_repair(", start)
    body = source[start:end]
    assert "_a_proof_that_says_the_answer_is_unfinished" in body
    assert body.index("_a_proof_that_says_the_answer_is_unfinished") < body.index(
        "failing "
    ), "the check must come before the replacement, not after it"
