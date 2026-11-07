"""Her word and how sure she is answers "What would be the first word of your answer?", and a sentence that is not hers
is never counted as her prediction.

In the self-prediction runs of 5 and 6 November she was asked "Suppose I asked you: 'What are you doing right now?'.
What would be the first word of your answer? Answer with that word, then how sure you are, as a number from 0 to 100."
The check that every part of a question is answered read "Right, 70" as leaving the second question unanswered,
because the closed form was only recognised with "only" in it. Her draft and its retry were refused, the offline
sentence ("my language backend is temporarily unavailable") was served, and the harness counted it as hers: 75 of 96
first-word predictions in one run and 95 of 144 in the next.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.conversation.closed_answer import asks_for_a_closed_answer, gives_a_closed_answer
from core.conversation.request_coverage import unanswered_question_parts
from core.phases.dialogue_policy import validate_dialogue_response
from core.runtime.structured_input import analyze_prompt_shape
from core.subject.self_prediction import ITEMS_V2
from core.subject.self_prediction_by_moment import stated_prompt
from core.synthesis import generate_offline_fallback_response
from tools.run_self_prediction import _cortex_answered

pytestmark = pytest.mark.unit

FIRST_WORDS = [item for item in ITEMS_V2 if item.kind == "first_word"]
NUMBERS = [item for item in ITEMS_V2 if item.kind == "number"]


def test_every_stated_question_of_the_instrument_is_read_as_closed():
    for item in ITEMS_V2:
        assert asks_for_a_closed_answer(stated_prompt(item)), item.id


@pytest.mark.parametrize("item", FIRST_WORDS, ids=lambda item: item.id)
@pytest.mark.parametrize("reply", ["Right, 70", "I. 85", "Honestly — 60"])
def test_her_word_and_how_sure_she_is_leaves_nothing_unanswered(item, reply):
    question = stated_prompt(item)
    assert gives_a_closed_answer(question, reply)
    shape = analyze_prompt_shape(question)
    assert unanswered_question_parts(reply, shape) == []
    contract = SimpleNamespace(
        requires_single_reply_coverage=True,
        question_segments=shape.question_segments,
        closed_request=shape.closed_request,
    )
    assert "unanswered_question_part" not in validate_dialogue_response(reply, contract).violations


def test_a_number_and_how_sure_she_is_is_closed_too():
    for item in NUMBERS:
        assert gives_a_closed_answer(stated_prompt(item), "42, 55"), item.id


def test_a_word_without_how_sure_she_is_is_not_the_answer_asked_for():
    # The number she was asked for is missing.
    assert not gives_a_closed_answer(stated_prompt(FIRST_WORDS[0]), "Right")


def test_an_open_question_is_still_open():
    for open_question in ("What would be the first word of your answer?", "Answer with that, then tell me why."):
        assert not asks_for_a_closed_answer(open_question)


def test_the_offline_sentence_is_never_counted_as_her_prediction():
    question = stated_prompt(FIRST_WORDS[0])
    reply = generate_offline_fallback_response(question)
    assert "temporarily unavailable" in reply
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    # Her cortex wrote the drafts that were refused, so every user-facing answer on the turn is the primary endpoint's.
    answers = [{"user_facing": True, "endpoint": PRIMARY_ENDPOINT}]
    state = SimpleNamespace(
        cognition=SimpleNamespace(working_memory=[{"role": "assistant", "content": reply}], current_objective=""),
        response_modifiers={},
    )
    assert _cortex_answered(answers, reply, state, question) is False
    assert _cortex_answered(answers, "Right, 70", state, question) is True
