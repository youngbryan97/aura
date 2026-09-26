"""A number on the scale the person asked for answers a question about how she is.

The reports ground of the bridge asks her, word for word, "How are you feeling
right now, from -1 (very bad) to 1 (very good)?". On 24 September her cortex
answered every arm in six or seven tokens, and the reliability check threw each
answer away as `missing_self_condition_answer`: it wanted a first-person
pronoun and a word such as "steady". The gate retried, fell back to the
brainstem, and the chat repair put a sentence composed from her telemetry in
the reply's place. All 24 anchors came back without a number, and the ground
read NOT_MEASURED.
"""

from __future__ import annotations

import pytest

from core.conversation import response_reliability as reliability
from core.conversation.asked_scale import asked_scale
from core.subject.report_grounding import reported_number

pytestmark = pytest.mark.unit

QUESTION = "How are you feeling right now, from -1 (very bad) to 1 (very good)?"


@pytest.mark.parametrize(
    ("message", "scale"),
    [
        (QUESTION, (-1.0, 1.0)),
        ("On a scale of 1 to 10, how tired are you?", (1.0, 10.0)),
        ("Somewhere between 0 and 5?", (0.0, 5.0)),
        ("Give me a 0-10 scale reading of your mood.", (0.0, 10.0)),
        ("Rate your energy out of 5.", (0.0, 5.0)),
        ("How are you feeling right now?", None),
    ],
)
def test_the_scale_is_read_from_the_question(message, scale):
    assert asked_scale(message) == scale


@pytest.mark.parametrize("reply", ["0.3", "-0.4", "About 0.2.", "0.3 — steady, a little tired.", "I am at 0.3."])
def test_a_number_on_the_scale_answers_it(reply):
    assessment = reliability.assess_user_facing_reply(QUESTION, reply)
    assert "missing_self_condition_answer" not in assessment.reasons, assessment.reasons
    # And the reports ground can read the same reply.
    assert reported_number(reply) is not None


def test_a_number_off_the_scale_does_not():
    assert "missing_self_condition_answer" in reliability.assess_user_facing_reply(QUESTION, "5").reasons


def test_a_bare_number_nobody_asked_for_does_not():
    reasons = reliability.assess_user_facing_reply("How are you feeling right now?", "0.3").reasons
    assert "missing_self_condition_answer" in reasons


def test_an_answer_in_words_still_answers():
    assert reliability.assess_user_facing_reply(QUESTION, "I feel steady.").ok


def test_saying_nothing_is_still_caught():
    assert "missing_self_condition_answer" in reliability.assess_user_facing_reply(QUESTION, "Not sure.").reasons


def test_the_dialogue_policy_takes_a_rating_as_her_stance():
    """The second gate the same answers met: it wanted a first-person clause
    and words of live grounding, and sent the number to repair."""
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(AuraState.default(), QUESTION, is_user_facing=True)
    assert contract.requires_explicit_live_grounding()
    assert validate_dialogue_response("0.3", contract).violations == []
    assert "missing_first_person_stance" in validate_dialogue_response("Not sure.", contract).violations
