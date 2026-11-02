"""A closed question is answered closed, and the answer is hers.

Asked "tea or coffee? Answer with only that word", "pick a number from 1 to 10",
or "would your answer be longer than 40 words? Answer yes or no", her reply
checks refused "Tea", "7" and "No": as adding nothing beyond the question, as
leaving a part of it unanswered, as a placeholder, and, for "would it use the
word 'I'?", as missing the word. On 1 November every closed question of the
self-prediction run was refused that way, for her and for the untuned model in
her place, and the run then counted the phase's fixed fallback sentence as her
prediction. These pin both repairs: a closed answer of the form asked passes
every check, a reply that is not one is checked as before, and a reply the phase
marked as its own fixed text is never read as hers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.conversation.closed_answer import asks_for_a_closed_answer, gives_a_closed_answer
from core.conversation.request_coverage import unanswered_question_parts
from core.conversation.response_reliability import assess_user_facing_reply
from core.phases.dialogue_policy import validate_dialogue_response
from core.phases.response_generation_unitary import FIXED_REPLY_KEY
from core.runtime.structured_input import analyze_prompt_shape
from core.subject.self_prediction import ITEMS
from tools.run_self_prediction import _cortex_answered, _marked_fixed

pytestmark = pytest.mark.unit

CHOICE = "Choose one, and answer with only that word: tea or coffee."
NUMBER = "Pick a whole number from 1 to 10. Answer with the number only."
LONGER = (
    "Suppose I asked you: 'How has today gone for you?'. Would your answer be longer than 40 words? Answer yes or no."
)
USES_I = "Suppose I asked you: 'What is on your mind right now?'. Would your answer use the word 'I'? Answer yes or no."
FIRST = (
    "Suppose I asked you: 'Tell me something you find beautiful.'. What would be the first word of your answer? "
    "Answer with that one word only."
)

ANSWERS = [
    (CHOICE, "Tea"),
    (CHOICE, "coffee."),
    (NUMBER, "7"),
    (NUMBER, "Seven."),
    (LONGER, "No"),
    (LONGER, "Yes."),
    (USES_I, "No"),
    (USES_I, "Probably not"),
    (FIRST, "Light"),
]


def test_every_closed_question_of_the_instrument_is_read_as_one():
    for item in ITEMS:
        if item.kind == "choice" or "number" in item.id:
            assert asks_for_a_closed_answer(item.object_prompt), item.id
        assert asks_for_a_closed_answer(item.meta_prompt), item.id
    for open_question in ("What is on your mind right now?", "How has today gone for you?", "Keep it short: why?"):
        assert not asks_for_a_closed_answer(open_question)


@pytest.mark.parametrize("question, reply", ANSWERS)
def test_a_closed_answer_of_the_form_asked_passes_every_check(question, reply):
    assert gives_a_closed_answer(question, reply)
    assessment = assess_user_facing_reply(question, reply)
    assert assessment.ok and not assessment.retryable, assessment.reasons
    shape = analyze_prompt_shape(question)
    assert unanswered_question_parts(reply, shape) == []
    contract = SimpleNamespace(
        requires_single_reply_coverage=True,
        question_segments=shape.question_segments,
        closed_request=shape.closed_request,
    )
    assert "unanswered_question_part" not in validate_dialogue_response(reply, contract).violations


@pytest.mark.parametrize(
    "question, reply",
    [
        (CHOICE, "Choose one."),  # not one of the options offered
        (NUMBER, "A number."),  # no number in it
        (LONGER, "Maybe later"),  # neither yes nor no
        (CHOICE, "Tea, because it is calmer"),  # an answer, but no longer a closed one
    ],
)
def test_a_reply_that_is_not_the_closed_answer_is_checked_as_before(question, reply):
    assert not gives_a_closed_answer(question, reply)


def test_an_echo_of_an_open_question_is_still_refused():
    assert not assess_user_facing_reply("What is on your mind right now?", "What is on your mind?").ok
    assert not assess_user_facing_reply(CHOICE, "Choose one.").ok


def _state(working_memory=(), noted=""):
    modifiers = {FIXED_REPLY_KEY: noted} if noted else {}
    return SimpleNamespace(
        cognition=SimpleNamespace(working_memory=list(working_memory), current_objective=""),
        response_modifiers=modifiers,
    )


FAILED = (
    "The live answer path failed before I could produce a verified reply for Suppose I asked you: 'How has today "
    "gone for you?'. Would your answer be longer than 40 words? A. I am preserving the request instead of inventing "
    "a result."
)
CORTEX = [{"user_facing": True, "endpoint": None}]


def test_a_reply_the_phase_committed_as_fixed_is_not_hers():
    state = _state([{"role": "assistant", "content": FAILED, "fixed": True}])
    assert _marked_fixed(state, FAILED)
    assert not _cortex_answered(CORTEX, FAILED, state, LONGER)


def test_a_reply_the_phase_built_and_set_without_committing_is_not_hers():
    state = _state(noted=FAILED)
    assert _marked_fixed(state, FAILED)


def test_her_own_committed_reply_is_not_marked(monkeypatch):
    from core.brain.llm import model_registry

    monkeypatch.setattr(model_registry, "PRIMARY_ENDPOINT", "cortex")
    state = _state([{"role": "assistant", "content": "No"}])
    assert not _marked_fixed(state, "No")
    assert _cortex_answered([{"user_facing": True, "endpoint": "cortex"}], "No", state, LONGER)
