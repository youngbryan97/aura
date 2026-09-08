"""A branch that decoded nothing has no prior candidate, and is not an error.

`advance_post_adaptation_candidate` accepts `None` for "there was no prior
candidate" and rejects anything else that is not non-empty text. The winning
branch's probe can come back empty, which is an ordinary outcome, and the
caller passed the empty string — so the validator raised a type error at it.

What that cost: the latent episode died, its receipt came back without a
`kv_state_tree`, a `decode_incumbent`, a terminal disposition, a causal
receipt or branch isolation, all five contract checks failed together, the
cortex was declared unavailable, and a 9B answered a question the 27B was
resident for.

LIVE, 2026-09-07, one turn: `ValueError: post-adaptation prior candidate is
invalid` then `receipt_contract_failed:kv_state_tree_unproven,
decode_incumbent_unproven, terminal_disposition_unproven,
causal_receipt_unproven, branch_isolation_unproven` then `🪜 Fallback ladder
answered while the cortex was unavailable`.
"""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.post_adaptation_candidate import (
    advance_post_adaptation_candidate,
)

_EVIDENCE = {"latent_opt_attempts": 1, "latent_opt_accepted_steps": 0}


def _advance(prior):
    return advance_post_adaptation_candidate(
        selected_branch=0,
        prior_candidate=prior,
        observed_candidate="an answer",
        stage="post_final_adaptation",
        strict_answer_contract=False,
        adaptation_evidence=_EVIDENCE,
    )


def test_no_prior_candidate_is_allowed() -> None:
    receipt, admitted = _advance(None)
    assert isinstance(receipt, dict)
    assert admitted == "an answer"


def test_a_real_prior_candidate_is_allowed() -> None:
    receipt, admitted = _advance("what the branch said")
    assert isinstance(receipt, dict)
    assert admitted == "an answer"


def test_an_empty_string_is_still_rejected_by_the_validator() -> None:
    """The validator's rule stays a real rule; the caller says what it means."""

    with pytest.raises(ValueError, match="prior candidate is invalid"):
        _advance("")


def test_the_caller_turns_an_empty_probe_into_no_candidate() -> None:
    from pathlib import Path

    engine = Path(__file__).resolve().parents[1] / "core/brain/llm/latent_cortex/engine.py"
    body = engine.read_text("utf-8")
    marker = body.index("prior_candidate = (")
    window = body[marker : marker + 1800]
    assert "isinstance(prior_candidate, str) and not prior_candidate" in window
    assert "prior_candidate = None" in window
    # And it is recorded rather than silently swallowed.
    assert "post_adaptation_prior_candidate_empty" in window
