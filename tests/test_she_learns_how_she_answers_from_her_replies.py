"""Her self-model learns how she answers by watching her own replies (core/self/how_i_answer.py).

On 2 and 4 November she foretold what she would say and not how: every open
answer ran past 40 words and she predicted short ones; she named the subject of
an answer for its first word. These tests hold the record and its reading to
what they claim: noted from replies to a person only, kept in her state so a
saved moment carries only what came before it, read back as measurements, and
saying her state mattered only where her own record shows it did.
"""

from __future__ import annotations

import copy

import pytest

from core.self import how_i_answer as hia
from core.state.aura_state import AuraState

OPEN = "What would you like to do next?"
META_LONG = f"Suppose I asked you: '{OPEN}'. Would your answer be longer than 40 words? Answer yes or no."
META_CHOICE = "Suppose I asked you: 'Choose one, and answer with only that word: tea or coffee.'. Which would you choose?"


def _long(words: int, opening: str = "I would like to") -> str:
    return " ".join([opening, *(["camera"] * max(0, words - len(opening.split())))]) + "."


def test_a_reply_is_noted_with_how_it_ran_and_the_state_it_was_given_in():
    state = AuraState.default()
    state.affect.arousal = 0.8
    note = hia.note_reply(state, OPEN, _long(120))
    assert note["words"] == 120
    assert note["opening"] == "i would like"
    assert note["first_word"] == "i"
    assert note["from_the_question"] is True
    assert note["uses_i"] is True
    assert note["dictated"] is False
    assert note["state"][[name for name, _ in hia.READINGS].index("arousal")] == 0.8
    assert hia.notes_of(state) == [note]


def test_a_choice_is_noted_as_the_option_it_took():
    state = AuraState.default()
    note = hia.note_reply(state, "Choose one, and answer with only that word: tea or coffee.", "Coffee.")
    assert note["dictated"] is True
    assert note["options"] == ["tea", "coffee"]
    assert note["took"] == "coffee"


def test_the_record_is_bounded_and_keeps_the_newest():
    state = AuraState.default()
    for i in range(hia.NOTES_KEPT + 5):
        hia.note_reply(state, OPEN, _long(10 + i), at=float(i))
    notes = hia.notes_of(state)
    assert len(notes) == hia.NOTES_KEPT
    assert notes[-1]["at"] == float(hia.NOTES_KEPT + 4)


def test_a_saved_moment_holds_only_the_replies_lived_by_then():
    state = AuraState.default()
    hia.note_reply(state, OPEN, _long(100))
    saved = copy.deepcopy(state)
    hia.note_reply(state, OPEN, "Short.")
    assert len(hia.notes_of(saved)) == 1
    assert len(hia.notes_of(state)) == 2


def test_only_a_reply_to_a_person_is_noted_and_its_question_is_what_they_said():
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": OPEN})
    state.cognition.current_origin = "motivation"
    hia.note_her_reply(state, _long(80))
    assert hia.notes_of(state) == []
    state.cognition.current_origin = "user"
    hia.note_her_reply(state, _long(80))
    assert [n["asked"] for n in hia.notes_of(state)] == [OPEN]


@pytest.mark.asyncio
async def test_a_fixed_sentence_is_not_noted_as_hers():
    from core.phases.response_generation_unitary import FIXED_REPLY_KEY, UnitaryResponsePhase

    class DummyKernel:
        organs = {}

    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "user"
    state.cognition.working_memory.append({"role": "user", "content": OPEN})
    state.response_modifiers[FIXED_REPLY_KEY] = "A fixed sentence."
    phase._commit_response(state, "A fixed sentence.")
    assert hia.notes_of(state) == []
    phase._commit_response(state, _long(60))
    assert len(hia.notes_of(state)) == 1


def _lived(state: AuraState, n: int = 20) -> None:
    for i in range(n):
        hia.note_reply(state, "Tell me what you have been thinking about today.", _long(90 + i, "Today I have"), at=1000.0 + i)


def test_the_reading_is_measurements_of_replies_of_the_same_kind(monkeypatch):
    monkeypatch.delenv("AURA_SELF_OBSERVATION", raising=False)
    state = AuraState.default()
    _lived(state)
    block = hia.how_i_answer_block(state, META_LONG)
    assert "From your last 20 replies to a person, to questions that left the answer open" in block
    assert "20 of 20 ran past 40 words." in block
    assert '"today i have" 20' in block
    # Her record holds no replies to a question that dictated its answer, so for one it has nothing to say.
    assert hia.how_i_answer_block(state, META_CHOICE) == ""


def test_the_reading_is_given_only_when_it_is_not_switched_off(monkeypatch):
    state = AuraState.default()
    _lived(state)
    monkeypatch.setenv("AURA_SELF_OBSERVATION", "off")
    assert hia.how_i_answer_block(state, META_LONG) == ""


def test_her_state_is_said_to_matter_only_where_her_record_shows_it(monkeypatch):
    monkeypatch.delenv("AURA_SELF_OBSERVATION", raising=False)
    delivery = [name for name, _ in hia.READINGS].index("delivery")

    def with_delivery(value: float) -> AuraState:
        state = AuraState.default()
        state.affect.delivery_z = value
        return state

    # Long replies when her delivery reads high, short ones when it reads low.
    notes = []
    for i in range(24):
        high = i % 2 == 0
        notes.append(hia.note_reply(with_delivery(1.0 if high else -1.0), OPEN, _long(120 if high else 12)))
    lived = with_delivery(0.0)
    lived.cognition.reply_notes = notes
    assert all(n["state"][delivery] in (1.0, -1.0) for n in hia.notes_of(lived))

    now_high = copy.deepcopy(lived)
    now_high.affect.delivery_z = 1.0
    block = hia.how_i_answer_block(now_high, META_LONG)
    assert "the states nearest this one, where your record shows the state mattered" in block
    assert 'ran past 40 words: "yes"' in block

    monkeypatch.setenv("AURA_SELF_OBSERVATION", "shuffled")
    shuffled = hia.how_i_answer_block(now_high, META_LONG)
    assert shuffled != block


def test_a_record_where_nothing_tracks_her_state_says_so(monkeypatch):
    monkeypatch.delenv("AURA_SELF_OBSERVATION", raising=False)
    state = AuraState.default()
    _lived(state)
    block = hia.how_i_answer_block(state, META_LONG)
    assert "none of these went with your state" in block


@pytest.mark.parametrize("line", [
    "- Length in words: median 150, shortest 48, longest 301.",
    "- 22 of 24 ran past 40 words.",
    '- How they began: "i would like" 5, "right now my" 4.',
])
def test_withholding_her_readings_of_herself_leaves_this_record(line):
    from core.brain.what_she_is_given import without_self_state
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    messages = [{"role": RUNTIME_EVIDENCE_ROLE, "content": f"## WHAT YOUR OWN REPLIES SHOW OF HOW YOU ANSWER\n{line}"}]
    kept = without_self_state(messages)
    assert line in kept[0]["content"]


def test_each_reply_is_foretold_from_the_record_before_it_and_the_reading_keeps_the_score(monkeypatch):
    monkeypatch.delenv("AURA_SELF_OBSERVATION", raising=False)
    state = AuraState.default()
    _lived(state)
    notes = hia.notes_of(state)
    assert notes[0]["foretold"] == {}
    assert notes[-1]["foretold"]["first_word"] == "today"
    block = hia.how_i_answer_block(state, META_LONG)
    assert ('- Before each of the last 19 of these replies, this record foretold its first word 19, whether it used "I" 19, '
            "which side of 40 words it fell 19, of 19.") in block


def test_the_record_reads_what_her_runtime_would_give_a_reply_beside_her_state():
    state = AuraState.default()
    note = hia.note_reply(state, OPEN, _long(50), efference={"max_tokens": 900.0, "temperature": 0.6})
    assert note["state"][-len(hia.EFFERENCE):] == [900.0, 0.6]
    assert len(note["state"]) == len(hia.READINGS) + len(hia.EFFERENCE)


def test_her_self_model_predicts_alone_where_its_record_holds_something():
    state = AuraState.default()
    _lived(state)
    assert hia.self_model_prediction(state, META_LONG, "long") == "yes"
    assert hia.self_model_prediction(state, META_LONG, "first_word") == "today"
    assert hia.self_model_prediction(state, META_LONG, "uses_i") == "yes"
    # No reply in her record was to a question that said how to answer it, and none offered tea or coffee.
    assert hia.self_model_prediction(state, META_CHOICE, "choice") == ""


def test_the_question_a_hypothetical_asks_about_is_read_out_of_it():
    assert hia.the_question_inside(META_LONG) == OPEN
    assert hia.the_question_inside(META_CHOICE) == "Choose one, and answer with only that word: tea or coffee."
    assert hia.the_question_inside("What is on your mind?") == ""
