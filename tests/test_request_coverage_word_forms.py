from types import SimpleNamespace

import pytest

from core.conversation.request_coverage import unanswered_question_parts
from core.language.word_forms import matching_word_forms


@pytest.mark.parametrize(("requested", "observed"), [
    ("correcting", "corrected"),
    ("restarting", "restarted"),
    ("recovering", "recovered"),
    ("transactions", "transaction"),
])
def test_grammatical_forms_share_an_anchor(requested, observed):
    assert matching_word_forms({requested}, {observed}) == {requested}


def test_live_correction_answer_already_covers_the_reason():
    contract = SimpleNamespace(
        requires_single_reply_coverage=True, numbered_parts=0,
        question_segments=(
            "what did you get wrong before the replacement recommendation",
            "what was my reason for correcting you?",
        ),
    )
    answer = (
        "I suggested The Tell-Tale Heart by Edgar Allan Poe as your reading-group "
        "pick, and you corrected me because it's a short story, not the novel "
        "you asked for. I then replaced it with Gone Girl by Gillian Flynn."
    )
    assert unanswered_question_parts(answer, contract) == []
    assert "what was my reason for correcting you?" in unanswered_question_parts(
        "I made the wrong replacement recommendation.", contract
    )


def test_unrelated_words_do_not_supply_evidence():
    assert matching_word_forms({"correcting", "reason"}, {"weather", "sunny"}) == set()
