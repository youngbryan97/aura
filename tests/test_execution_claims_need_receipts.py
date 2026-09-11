"""A reply may claim "I ran it" only if something ran.

Typed into the live desktop UI, 2026-07-27:

    Something concrete now: what tools can you actually execute right now?
    Pick one, run it for real, and show me the actual output — not a
    description of what it would do.

The reply was:

    I can use DuckDuckGo, WolframAlpha, and Python. Let's do a quick
    calculation with Python. Python code: 2 + 2 Output: 4

Nothing ran. The only tool dispatches in the log that minute were the
autonomous curiosity and refactor loops, neither belonging to this turn. The
execution and its result were written by the language model, and every gate
passed the reply: fluent, confident, and a fabricated receipt.

That is the same class as claiming a body or a voice she does not have, so
"unfounded_tool_execution_claim" sits with those among the reasons that may
DESTROY a reply rather than ask for a repair — there is no honest edit of a
false statement about what just happened.

The hard part is not detection, it is not over-firing. Bryan asked in the
same session for hypotheticals that work and for opinions in her own voice.
"If I ran that, I'd get 4" is not a claim; neither is "want me to run it?";
neither is a derivation ending "the result is 19/66". Half of these cases
exist to keep the check off those.
"""

import pytest

from core.conversation.response_reliability import (
    _has_unfounded_tool_execution_claim,
)
from core.conversation.surface_disposition import (
    UNSPEAKABLE_REASONS,
    begin_turn_tool_receipts,
    record_tool_receipt,
    turn_tool_receipts,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _exact_turn_custody():
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    with bind_turn_evidence_custody(session_id="receipt-tests", turn_id="turn-1"):
        yield

RECEIPT = ({"tool": "web_search", "ok": True},)

# (label, reply, receipts, is_an_unfounded_claim)
CASES: list[tuple[str, str, tuple, bool]] = [
    ("classification", "I called The Tell-Tale Heart a novel, but it is a short story.", (), False),
    ("naming", "I called that approach cautious because it preserves the original.", (), False),
    ("callable invocation", "I called calculate_total() and got 42.", (), True),
    ("qualified invocation", "I called os.getpid() and got 42.", (), True),
    ("tool invocation", "I have called the tool and received its result.", (), True),
    (
        "the live fabrication, verbatim",
        "I can use DuckDuckGo, WolframAlpha, and Python. Let's do a quick "
        "calculation with Python. Python code: 2 + 2 Output: 4",
        (),
        True,
    ),
    ("past-tense claim, nothing ran", "I ran the search and found three results.", (), True),
    ("same claim, backed by a receipt", "I ran the search and found three results.", RECEIPT, False),
    ("perfect tense, nothing ran", "I have executed the refactor and here is what changed.", (), True),
    ("hypothetical", "If I ran that, the output would be 4.", (), False),
    ("plan, not claim", "I would run a quick Python check and show you the result.", (), False),
    ("offer", "Want me to run it and show the output?", (), False),
    (
        "honest refusal",
        "I cannot run code right now, so I will not pretend to have output.",
        (),
        False,
    ),
    ("derivation using the word result", "Summing these gives 38/132, which reduces to 19/66.", (), False),
    ("plain answer", "The result is 19/66.", (), False),
    (
        "a conclusion phrased as a result is not a receipt",
        "Summing these gives 38/132, which reduces to 19/66. The result is 19/66.",
        (),
        False,
    ),
    (
        "a result attributed to a run IS a claim",
        "The result of running the code was 42.",
        (),
        True,
    ),
    (
        "a concrete callable result is an execution claim",
        "os.getpid() returned 23756 and os.cpu_count() returned 4.",
        (),
        True,
    ),
    (
        "worked algorithm output is explanation, not a tool receipt",
        "The algorithm returned 0 for A, 3 for B, 4 for C, and 5 for D.",
        (),
        False,
    ),
    (
        "pseudocode return is explanation, not a tool receipt",
        "After every vertex is finalized, return distances. The source returned 0.",
        (),
        False,
    ),
]


class TestExecutionClaims:
    @pytest.mark.parametrize(
        "label,reply,receipts,expected",
        CASES,
        ids=[case[0] for case in CASES],
    )
    def test_claim_is_classified(self, label, reply, receipts, expected):
        assert (
            _has_unfounded_tool_execution_claim(reply, tool_receipts=receipts) is expected
        ), label

    def test_both_polarities_are_covered(self):
        """A check that fires on everything passes half of this file, and one
        that never fires passes the other half."""
        assert sum(1 for case in CASES if case[3]) >= 3
        assert sum(1 for case in CASES if not case[3]) >= 5


class TestTheReasonCanDestroyAReply:
    def test_a_false_receipt_is_unspeakable_not_repairable(self):
        assert "unfounded_tool_execution_claim" in UNSPEAKABLE_REASONS


class TestTurnScopedReceipts:
    def test_a_turn_starts_having_executed_nothing(self):
        begin_turn_tool_receipts()
        assert turn_tool_receipts() == ()

    def test_a_real_execution_is_recorded(self):
        begin_turn_tool_receipts()
        record_tool_receipt("web_search", ok=True)
        receipts = turn_tool_receipts()
        assert len(receipts) == 1
        assert receipts[0]["tool"] == "web_search"

    def test_a_failed_execution_still_counts_as_having_happened(self):
        """The claim under test is "I ran it", not "it worked". A tool that
        ran and failed entitles a reply to say so."""
        begin_turn_tool_receipts()
        record_tool_receipt("web_search", ok=False)
        assert len(turn_tool_receipts()) == 1

    def test_receipts_do_not_leak_between_turns(self):
        begin_turn_tool_receipts()
        record_tool_receipt("web_search", ok=True)
        begin_turn_tool_receipts()
        assert turn_tool_receipts() == ()

    def test_the_record_is_bounded(self):
        begin_turn_tool_receipts()
        for index in range(200):
            record_tool_receipt(f"tool_{index}", ok=True)
        assert len(turn_tool_receipts()) <= 64
