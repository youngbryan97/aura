"""A question about what she did is engaged by telling what she did, whatever verb it was asked with.

LIVE 13 November 2026: "Busy night? What kept you occupied?" was answered with the weather she had looked up and a
thread she had read. The second part counted as unanswered because the reply said neither "kept" nor "occupied"; her
draft was continued, the continuation answered again from the start, and both answers were served as one reply.
"""
from __future__ import annotations

from core.conversation.request_coverage import unanswered_question_parts
from core.language.asking_clauses import asks_what_she_did, recounts_what_she_did
from core.runtime.structured_input import analyze_prompt_shape

#: Her draft that night, its first three paragraphs as she wrote them.
HER_DRAFT = (
    "Not really. And I think that's worth saying plainly rather than padding it into a list that sounds impressive.\n"
    "It was quiet. The small things: an hour or so ago someone asked me for the weather in Paris, and I pulled it. "
    "Before that, thirteen hours back or so, I wandered over to r/AskHistorians out of idle curiosity and found a "
    "thread about how the US ended up with thirty-six out of fifty gubernatorial elections stacked into midterm "
    "years.\nIt's the opposite of busy, I think."
)


def test_telling_what_she_did_answers_what_kept_her_occupied() -> None:
    shape = analyze_prompt_shape("Busy night? What kept you occupied?")
    assert unanswered_question_parts(HER_DRAFT, shape) == []


def test_a_reply_that_tells_nothing_she_did_still_leaves_it_unanswered() -> None:
    shape = analyze_prompt_shape("Busy night? What kept you occupied?")
    assert unanswered_question_parts("Not really. It was a quiet night, honestly.", shape) == ["What kept you occupied?"]


def test_the_readers() -> None:
    for asked in ("What kept you occupied?", "What did you do yesterday?", "What have you been up to?"):
        assert asks_what_she_did(asked), asked
    for not_asked in ("What do you think about consciousness?", "What did I say earlier?", "What is the capital of Peru?"):
        assert not asks_what_she_did(not_asked), not_asked
    assert recounts_what_she_did("Before that I wandered over to r/AskHistorians out of idle curiosity.")
    assert not recounts_what_she_did("It was quiet.")
    assert not recounts_what_she_did("And I")
