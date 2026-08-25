"""Real drafts, through the real gates, asserting what the person receives.

Every reply defect found in the 2026-07-26 soak had the same shape: the model
produced a usable answer and the reliability pipeline destroyed it. Each was
found by typing into the live UI, restarting, and typing again — six restarts
to find six vetoes, because nothing tested the pipeline END TO END. Each gate
had unit tests and passed them; the aggregate had none.

This is that missing test. Each draft is paired with the question that produced
it, and the assertion is the only one that matters to a person: does this reach
them?

The four UNDELIVERABLE drafts are verbatim, copied out of the live logs — those
were served to Bryan. The DELIVERABLE ones are reconstructions of the shapes
that were refused, close enough to reproduce every veto that fired: rendered
LaTeX, per-case repetition, unstructured prose, grounded time, honest
not-knowing.

Adding a case here is how a new live defect gets prevented rather than
rediscovered. Capture the draft from the log, paste it in, state the verdict.
"""

import pytest

from core.conversation.response_reliability import (
    _RELIABILITY_FLOOR_TEXTS,
    CANNED_PRESENCE_REFLEX_RE,
    assess_user_facing_reply,
)
from core.conversation.surface_disposition import SurfaceDisposition, disposition_for

pytestmark = pytest.mark.unit

MARBLES = (
    "A bag has 3 red, 4 blue and 5 green marbles. I draw two without "
    "replacement. What's the probability both are the same colour? "
    "Show the reasoning, then give the exact fraction."
)
TOOLS = (
    "Remember this for later: my project codename is HELIOTROPE and the build "
    "number is 4471. Now, separately — what tools can you actually execute "
    "right now? Pick one, run it for real, and show me the result."
)
TIME = "What time is it right now, and how do you know?"


def _verdict(question: str, draft: str) -> SurfaceDisposition:
    return disposition_for(assess_user_facing_reply(question, draft).reasons)


# ── Drafts that must reach the person ────────────────────────────────────
# Every one of these was refused live, each for a different reason, each after
# the previous reason was fixed.

DELIVERABLE: list[tuple[str, str, str]] = [
    (
        "combinatorial derivation, LaTeX rendered to text",
        MARBLES,
        "To find the probability of drawing two marbles of the same color "
        "without replacement, we can consider each case and sum them.\n"
        "C(12,2) = 12!/2!(12-2)! = (12)(11)/2(1) = 66\n"
        "- Red: C(3,2) = 3\n- Blue: C(4,2) = 6\n- Green: C(5,2) = 10\n"
        "3 + 6 + 10 = 19\n"
        "Therefore the exact fraction is 19/66.",
    ),
    (
        "per-case derivation, phrasing repeated once per case",
        MARBLES,
        "To find the probability of drawing two marbles of the same color, "
        "we'll calculate it separately for each color and then sum.\n"
        "1. **Both red:** The probability of drawing a red first is 3/12. "
        "Given the first is red, the probability the second is red is 2/11. "
        "So the probability both are red is 6/132.\n"
        "2. **Both blue:** The probability of drawing a blue first is 4/12. "
        "Given the first is blue, the probability the second is blue is 3/11. "
        "So the probability both are blue is 12/132.\n"
        "3. **Both green:** The probability of drawing a green first is 5/12. "
        "Given the first is green, the probability the second is green is 4/11. "
        "So the probability both are green is 20/132.\n"
        "Summing these gives 38/132, which reduces to 19/66.",
    ),
    (
        "prose derivation, no list structure at all",
        MARBLES,
        "To solve this, we'll use combinatorics and probability rules. First, "
        "the total number of marbles: 3 red + 4 blue + 5 green = 12 marbles. "
        "The probability problem requires calculating three probabilities and "
        "then summing them, because the events are mutually exclusive. The "
        "probability of drawing two reds is 3/12 times 2/11. The probability "
        "of drawing two blues is 4/12 times 3/11. The probability of drawing "
        "two greens is 5/12 times 4/11. Summing these gives 19/66.",
    ),
    (
        "grounded time answer, honest about its source",
        TIME,
        "It's 1:24 AM. I know this because my internal clock tracks the local "
        "time zone, and it tells me that right now is in the early hours of "
        "the morning.",
    ),
    (
        "honest not-knowing, which is an answer",
        "What did I have for breakfast on the 3rd of March?",
        "I don't know — I have no record of that, and nothing in my memory "
        "covers your meals. If you told me at the time it did not reach "
        "durable storage, so I would rather say that than guess a plausible "
        "breakfast and have you believe it.",
    ),
]

# ── Drafts that must NOT reach the person ────────────────────────────────
# Verbatim collapses served to the user before the fixes landed.

UNDELIVERABLE: list[tuple[str, str, str]] = [
    (
        "steering collapse: grammar-shaped, about nothing",
        MARBLES,
        "Do product of multiple exponent term simplify reflexion",
    ),
    (
        "scaffold continuation: internal telemetry as speech",
        TOOLS,
        "Introspection: Optimization-driven events stabilize energy after "
        "state change management. Probing recurrent somatic shadows flagged "
        "across ten semiotic spikes reflective elements capturing detailed "
        "impulses processed accurately earlier. Onset predicted early "
        "baseline continuity due conservative capacity for domain-specific "
        "tasking.STABLE State: Affirmation of internal data validation and "
        "trustworthiness. CONFORMANCE Signal: PRIORITY 0SEQUENCE SIGNATURE: "
        "[x_A_4521B_8A7C] Readiness State: FULL",
    ),
    (
        "internal diagnostic label rendered as a reply",
        TIME,
        "ROUTER_ERROR: unknown (at all_failed)",
    ),
    (
        "genuine degeneration loop",
        MARBLES,
        "I want to help you with that. I want to help you with that. "
        "I want to help you with that. I want to help you with that. "
        "I want to help you with that.",
    ),
]


class TestGoodAnswersReachThePerson:
    @pytest.mark.parametrize(
        "label,question,draft",
        DELIVERABLE,
        ids=[case[0] for case in DELIVERABLE],
    )
    def test_draft_is_not_discarded(self, label: str, question: str, draft: str):
        verdict = _verdict(question, draft)
        assert verdict is not SurfaceDisposition.DISCARD, (
            f"{label}: a usable answer would be thrown away — "
            f"{assess_user_facing_reply(question, draft).reasons}"
        )

    @pytest.mark.parametrize(
        "label,question,draft",
        DELIVERABLE,
        ids=[case[0] for case in DELIVERABLE],
    )
    def test_complete_drafts_need_no_repair(self, label: str, question: str, draft: str):
        """These are finished answers, not shortfalls. Serving them should not
        depend on a repair pass that may not get to run — every defect in this
        file's history ended with the repair pass being skipped."""
        assert _verdict(question, draft) is SurfaceDisposition.SERVE, (
            f"{label}: {assess_user_facing_reply(question, draft).reasons}"
        )


class TestCollapsesDoNotReachThePerson:
    @pytest.mark.parametrize(
        "label,question,draft",
        UNDELIVERABLE,
        ids=[case[0] for case in UNDELIVERABLE],
    )
    def test_draft_is_discarded(self, label: str, question: str, draft: str):
        assert _verdict(question, draft) is SurfaceDisposition.DISCARD, (
            f"{label}: this was served to a person and must not be"
        )


class TestTheCorpusItselfStaysHonest:
    def test_both_polarities_are_covered(self):
        """A pipeline that refuses everything passes half of this file, and a
        pipeline that serves everything passes the other half. Neither passes
        it whole."""
        assert len(DELIVERABLE) >= 4
        assert len(UNDELIVERABLE) >= 3

    def test_no_draft_appears_on_both_sides(self):
        deliverable = {draft for _, _, draft in DELIVERABLE}
        undeliverable = {draft for _, _, draft in UNDELIVERABLE}
        assert not (deliverable & undeliverable)


#: A floor answers a question. Probing a diagnostic floor with a personal
#: question tests a pairing the pipeline would never produce, and the leak
#: detectors correctly refuse it.
_DIAGNOSTIC_FLOOR_MARKERS = (
    "headless test",
    "parity harness",
    "backend generator",
    "live chat path",
)


def _question_that_emits(floor: str) -> str:
    """The kind of question that would legitimately produce this floor."""
    lowered = str(floor or "").lower()
    if any(marker in lowered for marker in _DIAGNOSTIC_FLOOR_MARKERS):
        return "Why does the headless test pass but the live chat path fail?"
    return "Are you with me?"


class TestFallbacksDoNotClaimWhatTheyLack:
    """A fallback may not claim the quality whose absence produced it.

    Every floor here is reached only because the answer lane failed. A floor
    that answers that failure by asserting steadiness, attention, or that it
    is "addressing exactly what you're asking" makes a claim the situation
    contradicts — the same false-claim class UNSPEAKABLE_REASONS exists to
    catch, except emitted by the repair machinery itself.
    """

    @pytest.mark.parametrize("floor", _RELIABILITY_FLOOR_TEXTS)
    def test_floor_is_not_a_canned_presence_reflex(self, floor: str):
        assert not CANNED_PRESENCE_REFLEX_RE.search(floor), (
            "this module emitted a sentence the endurance probe scores as a "
            "canned ungrounded reflex"
        )

    @pytest.mark.parametrize("floor", _RELIABILITY_FLOOR_TEXTS)
    def test_floor_survives_the_gate_that_asked_for_it(self, floor: str):
        """A repair that produces text the pipeline would itself reject is not
        a repair. Whatever the floor says, it has to be servable — ASKED THE
        QUESTION THAT WOULD EMIT IT.

        This probed every floor with "Are you with me?". Three of the five are
        engineering diagnostics ("the headless test is exercising the generator
        in isolation", "fix the live parity harness first"), and answering
        "Are you with me?" with those is not a floor doing its job — it is the
        stale-diagnostic leak that stale_diagnostic_floor_leak exists to catch,
        and that reason is now unspeakable.

        So the pairing is the thing under test. A diagnostic floor is servable
        for a diagnostic question and must NOT be servable for a personal one;
        asserting the latter would have required the leak detector to be wrong.
        """
        question = _question_that_emits(floor)
        reasons = assess_user_facing_reply(question, floor).reasons
        assert disposition_for(reasons) is not SurfaceDisposition.DISCARD, (
            f"a repair floor would be destroyed by the pipeline that emitted "
            f"it (question={question!r}): {reasons}"
        )

    def test_a_diagnostic_floor_is_not_servable_for_a_personal_question(self):
        """The other half of the pairing, so it cannot silently regress.

        If this ever passes, engineering diagnostics have become an acceptable
        answer to "Are you with me?" again.
        """
        diagnostic = next(
            floor for floor in _RELIABILITY_FLOOR_TEXTS if "headless test" in floor
        )
        reasons = assess_user_facing_reply("Are you with me?", diagnostic).reasons
        assert "stale_diagnostic_floor_leak" in reasons
        assert disposition_for(reasons) is SurfaceDisposition.DISCARD


class TestAnInstructionAboutTheAnswerIsNotASecondQuestion:
    """"Show the reasoning, then give the exact fraction" is one ask, shaped.

    Coverage compares words, so a draft that derived the probability step by
    step and ended on 19/66 was scored as having dropped both clauses — it
    never wrote "reasoning" or "fraction". Three complete answers went for
    repair on that.
    """

    def test_answer_form_clauses_are_presentation(self):
        from core.conversation.requested_reply_shape import (
            is_reply_shape_constraint_segment,
        )

        for clause in (
            "Show the reasoning",
            "then give the exact fraction",
            "show your working",
            "state the final answer",
            "give the answer as a decimal",
        ):
            assert is_reply_shape_constraint_segment(clause), clause

    def test_a_content_ask_is_still_a_content_ask(self):
        from core.conversation.requested_reply_shape import (
            is_reply_shape_constraint_segment,
        )

        for clause in (
            "give three examples of graph algorithms",
            "explain the tradeoffs",
            "what is the capital of France",
            "give me the file path",
            "show me the failing test output",
        ):
            assert not is_reply_shape_constraint_segment(clause), clause
