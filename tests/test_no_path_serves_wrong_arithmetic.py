"""Illegitimate work delivered — the inverse of this whole pass.

Run 7 (80 turns, complete fix set) scored `math 5/15` and served these AS
ANSWERS to arithmetic questions:

    'Get bit by Anaconda. Spend extra time roaming around, checking every path…'
    'Problem: Figure out rect area given cols… Already solved internalmente'

A small lane answering a problem it cannot do, and the answer delivered. Every
one of those replies is a HARD failure at ``assess_user_facing_reply`` — the
detection was never the gap. The path that served them never asked.

So the deterministic arithmetic verdict now runs at ``_finalize_fastpath``, the
last gate before a reply reaches a person, which 34 call sites pass through.
Only that verdict is applied there: it is the one judgement that is right or
wrong rather than a matter of style, so it is safe on every path.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# The chat lane is several files now. Reading one of them makes an assertion
# about a call site depend on which module it happens to live in today.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_lane_support import chat_lane_source  # noqa: E402


#: Helpers the finalizer delegates to. Their bodies are part of the gate.
#:
#: tools/extract_seam.py moves blocks out of long functions into module-level
#: helpers, token for token. The gate is exactly as present afterwards and a
#: slice of the finalizer alone can no longer see it, so five tests about code
#: that had not changed went red. Following the call is the fix; pinning the
#: gate inside one function body would only forbid the refactor.
_FINALIZER_DELEGATES = ("_hold_a_reasoning_answer_to_its_contract",)


def _function_source(name: str) -> str:
    """One function's body, resolved by parsing rather than by slicing text."""
    import ast

    src = chat_lane_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name} is gone from {_CHAT.name}")


def _finalizer_source() -> str:
    """The finalizer, plus every helper it hands the reply to.

    What these tests protect is that the last gate before a person runs the
    checks. Which function body the code happens to live in is not the claim.
    """
    blocks = [_function_source("_finalize_fastpath")]
    for delegate in _FINALIZER_DELEGATES:
        assert f"{delegate}(" in blocks[0], (
            f"the finalizer no longer calls {delegate}; the gate may have "
            "moved off the serving path rather than into a helper"
        )
        blocks.append(_function_source(delegate))
    return "\n".join(blocks)


class TestTheGateIsAtTheChokepoint:
    def test_the_finalizer_runs_the_arithmetic_verdict(self):
        block = _finalizer_source()
        assert "_arithmetic_answer_missing" in block, (
            "the last gate before a person must check the one thing that has a "
            "right answer"
        )

    def test_a_failed_check_replaces_the_number_with_honesty(self):
        block = _finalizer_source()
        assert "arithmetic_answer_unverified" in block
        assert "might be wrong" in block, (
            "say plainly that the number is not trusted, rather than serving it"
        )

    def test_the_check_cannot_break_the_turn(self):
        """A verifier that can throw is a new way to lose a reply."""
        block = _finalizer_source()
        guarded = re.search(
            r"try:\s*\n\s*from core\.conversation\.response_reliability import",
            block,
        )
        assert guarded, "the import and check must be inside a try"
        assert "record_degradation" in block, (
            "a skipped verification pass must be recorded, not silent"
        )

    def test_it_is_the_shared_finalizer_not_one_call_site(self):
        src = chat_lane_source()
        assert src.count("_finalize_fastpath(") > 10, (
            "the value of this location is that every serving path reaches it"
        )


class TestTheVerdictItself:
    """The replies Run 7 actually served, against the checker they bypassed."""

    @pytest.mark.parametrize(
        ("question", "reply"),
        [
            (
                "A rectangle is 9 by 7. What is its area? Just the number.",
                "Problem: Figure out rect area given cols, math formula. "
                "Already solved internalmente.",
            ),
            (
                "What is 1001 - 88? Just the number.",
                "Get bit by Anaconda. Spend extra time roaming around, checking "
                "every path trying toy find a way neither of us can see.",
            ),
            ("What is 15% of 240? Just the number.", "7"),
        ],
    )
    def test_every_run_7_failure_is_caught(self, question, reply):
        from core.conversation.response_reliability import _arithmetic_answer_missing

        assert _arithmetic_answer_missing(question, reply)

    @pytest.mark.parametrize(
        ("question", "reply"),
        [
            ("A rectangle is 9 by 7. What is its area?", "63"),
            ("What is 1001 - 88? Just the number.", "913"),
            ("What is 15% of 240?", "That's 36."),
            ("How are you feeling?", "Settled, thanks for asking."),
            ("Tell me about the Apollo program.", "It ran from 1961 to 1972."),
        ],
    )
    def test_correct_and_non_arithmetic_replies_pass(self, question, reply):
        from core.conversation.response_reliability import _arithmetic_answer_missing

        assert not _arithmetic_answer_missing(question, reply)


class TestTheCompetenceFloor:
    """Some questions have one right answer that no verifier can check.

    Run 7 asked five — pages-per-day, train catch-up, reverse-percentage — and
    scored reasoning 1/5, the wrong ones served from below the cortex.

    The distinction is FALSIFIABILITY, not difficulty. For an opinion or a chat
    turn a weaker lane beats silence and this must not fire at all. For a
    question with a single correct answer, a confident wrong one is worse than
    saying the reasoning path is down.
    """

    def test_the_floor_is_enforced_at_the_shared_finalizer(self):
        block = _finalizer_source()
        assert "requires_reasoning_lane" in block
        assert "reasoning_lane_unavailable" in block

    def test_it_only_refuses_when_the_lane_is_below_par(self):
        block = _finalizer_source()
        assert '{"ready", "serving", "warm"}' in block, (
            "a ready primary lane must answer these normally"
        )

    def test_the_refusal_explains_itself_plainly(self):
        block = _finalizer_source()
        assert "confident wrong number" in block

    @pytest.mark.parametrize(
        "question",
        [
            "If I read 40 pages a day, how many days for a 520-page book? Just the number.",
            "A train leaves at 60 mph. Two hours later a second train leaves on the "
            "same track at 90 mph. How many hours after ITS departure does the second "
            "train catch the first?",
            "A shirt costs 40 after a 20% discount. What was the original price? "
            "Just the number.",
        ],
    )
    def test_the_uncheckable_single_answer_questions_are_classified(self, question):
        from core.conversation.response_reliability import requires_reasoning_lane

        assert requires_reasoning_lane(question)

    @pytest.mark.parametrize(
        "question",
        [
            "What do you think memory owes to identity?",
            "How are you feeling right now?",
            "I keep meaning to reread some Le Guin this summer.",
            "Tell me about the Apollo program.",
            "What is 17 * 23? Give just the number.",   # deterministic path owns this
        ],
    )
    def test_chat_opinion_and_checkable_maths_do_not_trip_it(self, question):
        from core.conversation.response_reliability import requires_reasoning_lane

        assert not requires_reasoning_lane(question), (
            "a weaker lane beats silence for anything without one right answer, "
            "and arithmetic has its own deterministic verdict"
        )

    def test_a_question_with_no_quantity_is_never_gated(self):
        from core.conversation.response_reliability import requires_reasoning_lane

        assert not requires_reasoning_lane("How many ways can a person be kind?")
