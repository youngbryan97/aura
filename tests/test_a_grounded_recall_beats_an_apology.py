"""A correct answer in hand is served, not thrown away for an apology.

The desktop chat path builds `expected_recall_reply` from the transcript of
the conversation when the person asks what was said. On a required full-mind
turn, when the model's own draft missed the recall contract, this branch
returned None on the grounds that a deterministic reconstruction is not her
answer.

What the caller serves instead is "I couldn't get my full attention onto that
one, and I'd rather tell you that than answer from half of it." That is not
her answer either. It carries nothing, and it is false about what happened —
the attention was there, and a gate rejected a draft.

The branch immediately above it, for self-condition grounding, already served
its canonical projection marked `replaced_by_runtime`. The two contracts
differed only in which was written second.

LIVE, 2026-09-07: "What did I just ask you?", asked one turn after the
question it was recalling. 411 seconds, two rejected drafts, and the apology,
with a correct answer in hand the whole time.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "interface/routes/chat.py"


def _branch_source() -> str:
    body = CHAT.read_text("utf-8")
    start = body.index("                if expected_recall_reply:")
    end = body.index("                if context_challenge_context", start)
    return body[start:end]


def test_the_branch_still_exists() -> None:
    assert "expected_recall_reply" in _branch_source()


def test_a_grounded_recall_is_returned_rather_than_refused() -> None:
    branch = _branch_source()
    assert "return expected_recall_reply" in branch, (
        "the recall branch has a correct answer and does not serve it"
    )


def test_it_is_marked_as_the_runtime_speaking_not_her() -> None:
    """Serving it is right; passing it off as generation is not."""

    branch = _branch_source()
    assert 'authorship_effect="replaced_by_runtime"' in branch
    assert "deterministic=True" in branch
    assert "bounded_contract_used=True" in branch


def test_it_still_has_to_survive_the_reply_assessment() -> None:
    """Bounded is not exempt: a reconstruction that reads wrong is refused."""

    branch = _branch_source()
    assert "assess_user_facing_reply(" in branch
    assert "_reply_assessment_requires_repair_with_memory_evidence(" in branch
    assert "return None" in branch, (
        "there must still be a path that refuses, for a reconstruction that "
        "does not survive assessment"
    )


def test_the_two_recall_paths_agree_with_each_other() -> None:
    """The sibling branch's shape, on the branch that used to differ."""

    body = CHAT.read_text("utf-8")
    sibling = body[
        body.index("chat.self_condition_bounded_projection") - 3000 :
        body.index("chat.self_condition_bounded_projection") + 1500
    ]
    branch = _branch_source()
    for marker in (
        'authorship_effect="replaced_by_runtime"',
        "deterministic=True",
        "bounded_contract_used=True",
        "post_generation_repair_applied=True",
    ):
        assert marker in sibling, f"the sibling stopped doing {marker}"
        assert marker in branch, f"the recall branch does not do {marker}"


def test_the_file_still_parses() -> None:
    ast.parse(CHAT.read_text("utf-8"))
