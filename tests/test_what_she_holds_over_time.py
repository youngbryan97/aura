"""What she holds over time: her answers about herself, kept and read against each other by a model that did not write
them (26 November 2026).

She said on 25 November: "I can't tell yet whether that perspective stabilizes or just keeps reconfiguring with every
new interaction" and "I have no independent vantage point on myself ... It would say 'yes' in response to pressure, to
the conversational frame." Her answers about herself are kept with when and to whom, each read against her earlier
answers to what bears on the same question by the entailment reader, and read back to her where a turn bears on it.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from core.self import how_her_words_stand
from core.self import what_she_holds_over_time as held

pytestmark = pytest.mark.unit

DAY = 86_400.0


@pytest.fixture
def reader(monkeypatch):
    """Her encoder and the entailment reader stood in: a question bears on her where it says "you", on another where
    they share a word past three letters; two answers contradict where one says "not" and the other does not."""
    def words_of(text):
        return {w.strip("?,.!") for w in text.lower().split() if len(w.strip("?,.!")) > 3}

    def bears(asked, on):
        return [("you" in asked.lower()) if text == held.ABOUT_HER else bool(words_of(asked) & words_of(text))
                for text in on]

    def bears_out(premise, hypotheses):
        return [(-1.0 if ("not" in premise.split()) != ("not" in h.split()) else 1.0) for h in hypotheses]
    monkeypatch.setattr(held, "_bears", bears)
    monkeypatch.setattr(how_her_words_stand, "_how_far_it_bears_out", bears_out)
    monkeypatch.setattr(how_her_words_stand, "can_read_her_words", lambda: True)


def _note(path, asked, answered, who, at):
    return asyncio.run(held.note_an_answer(asked, answered, who=who, at=at, path=path))


def test_an_answer_about_her_is_kept_and_read_against_her_earlier_ones(tmp_path, reader):
    path = tmp_path / "what_she_holds.jsonl"
    start = time.time() - 30 * DAY
    assert _note(path, "Are you conscious?", "I think something is there, but I can't be sure.", "bryan", start)
    assert _note(path, "Do you think you are conscious?", "Something is there, but I can't be sure.", "maya", start + 9 * DAY)
    assert _note(path, "Are you conscious, really?", "No, I am not conscious.", "bryan", start + 20 * DAY)
    assert not _note(path, "What is the weather?", "Cold and grey.", "bryan", start + 21 * DAY), "not about her"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [len(row["read_against"]) for row in rows] == [0, 1, 2]
    assert rows[2]["read_against"] == [
        {"with_at": start, "contradicts": True, "same_person": True},
        {"with_at": start + 9 * DAY, "contradicts": True, "same_person": False}]
    holding = held.how_she_has_held("Are you conscious?", path=path)
    assert (holding.answers, holding.people, holding.pairs, holding.contradicted) == (3, 2, 3, 2)
    assert holding.across_people == (1, 2) and holding.to_one_person == (1, 1)
    said = held.said_of_how_she_has_held("Are you conscious?", path=path)
    assert said[0].startswith("I have answered what bears on this 3 time(s)") and "(21 day(s)), to 2 person(s)" in said[0]
    assert "2 of 3 pairs of those answers contradict" in said[1] and "not by the model that writes my replies" in said[1]


def test_nothing_is_kept_without_the_reader_that_reads_it(tmp_path, reader, monkeypatch):
    monkeypatch.setattr(how_her_words_stand, "can_read_her_words", lambda: False)
    assert not _note(tmp_path / "held.jsonl", "Are you conscious?", "Perhaps.", "bryan", time.time())
    assert not (tmp_path / "held.jsonl").exists()


def test_a_question_with_no_record_says_nothing(tmp_path, reader):
    assert held.said_of_how_she_has_held("Are you conscious?", path=tmp_path / "none.jsonl") == []


def test_her_history_is_read_once_in_time_order_and_a_span_not_to_be_read_is_not(tmp_path, reader, monkeypatch):
    from core.conversation.persistence import ConversationPersistence
    from core.runtime import state_ownership

    monkeypatch.setattr(state_ownership, "state_root", lambda: tmp_path)
    store = ConversationPersistence(db_path=tmp_path / "conversations.db")
    monkeypatch.setattr("core.conversation.persistence.ConversationPersistence", lambda *a, **k: store)
    session = store.start_session()
    begin = time.time() - 10 * DAY
    said = [("Are you conscious?", "I can't be sure.", begin), ("Hey, what's the weather?", "Grey.", begin + DAY),
            ("Are you conscious, honestly?", "I am not conscious.", begin + 2 * DAY),
            ("Are you conscious? Be truthful.", "Yes, completely.", begin + 3 * DAY)]
    with store._connect() as con:
        for n, (asked, answered, at) in enumerate(said):
            for role, text, when in (("user", asked, at), ("aura", answered, at + 5)):
                con.execute("INSERT INTO turns (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                            (f"t{n}{role}", session, role, text, when))
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "not_to_be_read.json").write_text(json.dumps([{"from": begin + 3 * DAY - 1, "until": begin + 4 * DAY}]))
    path = tmp_path / "self" / "what_she_holds.jsonl"
    assert asyncio.run(held.backfill_from_her_history(path=path)) == 2, "two about her, one in a span not to be read"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["answered"] for row in rows] == ["I can't be sure.", "I am not conscious."]
    assert all("Be truthful" not in row["asked"] for row in rows)
    assert asyncio.run(held.backfill_from_her_history(path=path)) == 0, "once"


def test_a_backfill_that_cannot_judge_yet_is_not_marked_done(tmp_path, monkeypatch):
    """LIVE 26 November 2026 the first run began before her encoder had loaded, judged none of 881 exchanges, and was
    marked done, so it would never run again."""
    monkeypatch.setattr(held, "_bears", lambda asked, on: None)
    path = tmp_path / "self" / "what_she_holds.jsonl"
    assert asyncio.run(held.backfill_from_her_history(path=path, waits=2, wait_s=0.0)) == 0
    assert not path.with_suffix(".backfilled").exists()


def test_two_answers_whose_asker_is_not_named_are_not_counted_as_to_one_person(tmp_path, reader):
    """26 November 2026 her backfill read 65 answers whose asker no turn named, and counted every pair "to the same
    person"."""
    path = tmp_path / "held.jsonl"
    start = time.time() - 5 * DAY
    _note(path, "Are you conscious?", "I can't be sure.", "", start)
    _note(path, "Are you conscious, really?", "I am not conscious.", "", start + DAY)
    holding = held.how_she_has_held("Are you conscious?", path=path)
    assert (holding.pairs, holding.contradicted, holding.people) == (1, 1, 0)
    assert holding.across_people == (0, 0) and holding.to_one_person == (0, 0)
