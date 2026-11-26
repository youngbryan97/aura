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
