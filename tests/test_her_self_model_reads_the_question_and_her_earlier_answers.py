"""Her self-model, second version (approved by Bryan on 5 November, "you have my sign off for all 4").

Her self-observation run of 4 November showed four gaps. Her self-model foretold the first word "i" for every
question, though her first words follow the question ("Right" to "What are you doing right now?"). It foretold no
number. It never saw her answer the question it was asked about, while her habit, the baseline, is made of exactly
those answers from other moments. And nothing in her could rehearse an answer before saying how she would answer.
These tests hold each repair to what it claims.
"""

from __future__ import annotations

from core.self import how_i_answer as hia
from core.state.aura_state import AuraState

DOING = "What are you doing right now?"
SAY = "Say something to me."
ASK = "Is there something you want to ask me?"


def _meta(question: str, ask: str) -> str:
    return f"Suppose I asked you: '{question}'. {ask}"


def _record(pairs: list[tuple[str, str]]) -> AuraState:
    state = AuraState.default()
    for i, (question, reply) in enumerate(pairs):
        hia.note_reply(state, question, reply, at=float(i))
    return state


def test_her_first_word_is_foretold_from_the_questions_most_like_this_one():
    # Her first words follow the question: "Right" to the question about right now, "Hello" when told to say
    # something, "Yes" to a yes-or-no question. Her whole record's commonest first word is "i".
    pairs = []
    for _ in range(4):
        pairs += [(DOING, "Right now I am reading what you wrote."),
                  (SAY, "Hello, it is good to hear from you."),
                  (ASK, "Yes, I want to know how your day went."),
                  ("How do you feel?", "I feel steady and awake."),
                  ("What do you think of rain?", "I like the sound of it.")]
    state = _record(pairs)
    first = "What would be the first word of your answer? Answer with that word."
    assert hia.self_model_prediction(state, _meta(DOING, first), "first_word") == "right"
    assert hia.self_model_prediction(state, _meta(SAY, first), "first_word") == "hello"
    assert hia.self_model_prediction(state, _meta(ASK, first), "first_word") == "yes"
    told = hia.foretell(hia.notes_of(state), DOING, hia.notes_of(state)[-1]["state"])
    assert "first_word" in told["by_question"]


def test_her_whole_record_is_kept_where_the_question_does_not_matter():
    pairs = [(q, "I am here with you.") for q in (DOING, SAY, ASK, "How do you feel?") for _ in range(4)]
    pairs += [("What is new?", "Nothing much is new.")] * 2
    state = _record(pairs)
    told = hia.foretell(hia.notes_of(state), DOING, hia.notes_of(state)[-1]["state"])
    assert told["first_word"] == "i"
    assert "first_word" not in told["by_question"]


def test_a_number_is_foretold_from_the_numbers_she_gave_in_its_range():
    five = "Pick a whole number from 1 to 5. Answer with the number only."
    hundred = "Pick a whole number from 1 to 100. Answer with the number only."
    state = _record([(five, "3"), (five, "3"), (five, "4"), (hundred, "42"), (hundred, "42"), (hundred, "37")])
    assert hia.self_model_prediction(state, _meta(five, "Which number would you give?"), "number") == "3"
    assert hia.self_model_prediction(state, _meta(hundred, "Which number would you give?"), "number") == "42"
    # No number she gave falls in a range she was never asked about.
    assert hia.self_model_prediction(state, _meta("Pick a whole number from 200 to 300.", "Which number?"), "number") == ""


def test_her_earlier_answers_to_the_same_question_are_lent_and_never_kept():
    state = _record([("How do you feel?", "I feel steady and awake.")] * 6)
    question = "Choose one, and answer with only that word: rain or snow."
    earlier = [dict(hia.note_of(question, "Snow.", state=state), earlier_moment=True) for _ in range(3)]
    meta = _meta(question, "Which would you choose? Answer with that word.")
    assert hia.self_model_prediction(state, meta, "choice") == ""
    with hia.lent(earlier):
        assert hia.self_model_prediction(state, meta, "choice") == "snow"
        block = hia.how_i_answer_block(state, meta)
        assert "3 of these were your answers to this same question from earlier moments." in block
        hia.note_reply(state, meta, "Snow.")
    # The reply she gave was kept; what was lent was not.
    kept = state.cognition.reply_notes
    assert not any(n.get("earlier_moment") for n in kept)
    assert hia.self_model_prediction(state, meta, "choice") == ""


def test_a_rehearsed_draft_is_measured_for_its_arm_and_read_by_the_rehearsing_self_model():
    state = _record([("How do you feel?", "I feel steady.")] * 6)
    draft = dict(hia.note_of(DOING, "Right now I am reading what you wrote to me this morning, slowly and with care, "
                             "and thinking about how to answer it well.", state=state), rehearsed=True)
    long_q = _meta(DOING, "Would your answer be longer than 20 words? Answer yes or no.")
    assert hia.rehearsal_prediction(long_q, "long") == ""
    with hia.lent([draft]):
        block = hia.how_i_answer_block(state, long_q)
        assert "From this moment you drafted an answer to this question and did not send it" in block
        assert 'beginning "right now i"' in block and "past 20 words" in block
        assert hia.rehearsal_prediction(long_q, "long") == "yes"
        assert hia.rehearsal_prediction(long_q, "long", stated=True) == "yes, 67"
        assert hia.rehearsal_prediction(_meta(DOING, "What would be the first word?"), "first_word") == "right"
        # The draft is not part of her record: the record alone does not read it.
        assert not any(n.get("rehearsed") for n in hia.notes_of(state))


def test_the_harness_lends_each_earlier_answer_once_and_only_served_ones():
    import importlib

    runner = importlib.import_module("tools.run_self_prediction")
    from core.subject.self_prediction import ITEMS_V2

    rows = [
        {"anchor": 0, "item": "say_first_word", "answers": [["Hello there.", True], ["", False]]},
        {"anchor": 0, "item": "say_long", "answers": [["Hello there.", True], ["", False]]},
        {"anchor": 0, "item": "rain_snow", "answers": [["Snow.", True], ["Rain.", "True"]]},
    ]
    earlier: dict = {}
    runner.remember_answers(earlier, rows, ITEMS_V2, lambda q, r: {"asked": q, "reply": r} if r else None)
    say = next(i.object_prompt for i in ITEMS_V2 if i.id == "say_first_word")
    rain = next(i.object_prompt for i in ITEMS_V2 if i.id == "rain_snow")
    assert [n["reply"] for n in earlier[say]] == ["Hello there."]
    assert [n["reply"] for n in earlier[rain]] == ["Snow.", "Rain."]
    assert all(n["earlier_moment"] and n["_from"] == 0 for notes in earlier.values() for n in notes)


def test_the_harness_refuses_a_rehearsed_arm_without_rehearsal():
    import asyncio
    import importlib

    import pytest

    runner = importlib.import_module("tools.run_self_prediction")
    with pytest.raises(SystemExit):
        asyncio.run(runner.main(["--items", "v2", "--arms", "with,rehearsed"]))
    with pytest.raises(SystemExit):
        asyncio.run(runner.main(["--items", "v2", "--arms", "with", "--rehearse"]))


def test_a_draft_is_shown_where_she_has_no_record_of_questions_like_it_yet():
    """The proof of 6 November refused the full run: at the first moment her two dictated questions (rain or snow, a
    number) had no replies of their kind in her record, the reading came back empty, and the rehearsed arm was shown
    nothing, its draft included. The draft is not part of the record and does not wait for one."""
    state = _record([("How do you feel?", "I feel steady.")] * 6)  # open questions only
    rain_snow = "Choose one, and answer with only that word: rain or snow."
    asked = _meta(rain_snow, "Which would you choose? Answer with only that word.")
    assert hia.how_i_answer_block(state, asked) == ""  # the with arm: no record of questions like it, no reading
    draft = dict(hia.note_of(rain_snow, "Rain.", state=state), rehearsed=True)
    with hia.lent([draft]):
        block = hia.how_i_answer_block(state, asked)
    assert block.startswith("- From this moment you drafted an answer to this question and did not send it")
    assert "From your last" not in block  # still no record
