"""What she says of how she does something is from what does it, and her language model gets no credit for what it has no part in."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import numpy as np
import pytest

from core.self import how_she_does_it as how
from core.understanding import fitted_on_her_model
from core.understanding.what_of_hers import NONE, WhatOfHers

ROOT = Path(__file__).resolve().parents[1]
#: The ways into a language model.
HER_MODEL = ("core.brain.llm", "core.brain.inference_gate", "core.brain.llm_health_router", "core.brain.her_unprompted_turn", "core.rebuilding.her_model",
             "core.understanding.her_model_between_turns", "core.local_chat_brain")


@pytest.mark.parametrize("way", how.WAYS, ids=lambda way: way.key)
def test_what_decides_is_there_and_her_language_model_has_the_part_it_is_said_to(way):
    for module in way.by:
        path = ROOT / module
        assert path.is_file(), f"{module} is not there"
        if way.language_model != how.NO_PART:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(name == model or name.startswith(model + ".") for name in names for model in HER_MODEL), f"{module} reaches a language model"
    assert (way.language_model == how.NO_PART) == (not way.its_part)


@pytest.fixture
def about(monkeypatch):
    monkeypatch.setattr(how, "credits_her_model", lambda sentence: "pattern-match" in sentence)
    monkeypatch.setattr(how, "WAYS", tuple(how.Way(**{**way.__dict__, "on_record": (lambda: "3 of them")}) for way in how.WAYS))
    yield how.hold_ways
    how.hold_ways(None)


REPLY = "I was in over my head. I pattern-match from training data, and it shows its limits fast. What pulled you to ask?"


def test_a_sentence_that_credits_her_model_is_refuted_by_how_it_is_done(about):
    """LIVE 24 November 2026, after eight games she chose every move of by search."""
    assert how.wrong_about_how(REPLY) == [], "a turn that was not read to be about it is left alone"
    about(["play"])
    wrong = how.wrong_about_how(REPLY)
    assert [w.sentence for w in wrong] == ["I pattern-match from training data, and it shows its limits fast."]
    assert "takes no part in it and is not asked" in wrong[0].said() and "3 of them" in wrong[0].said()
    about(["make"])
    assert how.wrong_about_how(REPLY) == [], "her language model has a part in making a program"


def test_it_is_taken_out_where_her_reply_is_held_against_her_record(about, monkeypatch):
    from core.epistemics.what_her_record_refutes import sentences_her_record_refutes

    about(["play"])
    assert [c.sentence for c in sentences_her_record_refutes(REPLY, "I liked watching")] == ["I pattern-match from training data, and it shows its limits fast."]


def test_a_turn_about_it_is_given_how_it_is_done(about):
    from core.brain.what_she_brings_to_a_turn import grounding_she_brings

    assert not any("HOW IT IS DONE" in block for block in grounding_she_brings({}, "good morning"))
    about(["play"])
    block = next(block for block in grounding_she_brings({}, "I liked watching you") if "HOW IT IS DONE" in block)
    assert "searching ahead in the rules she has learned" in block and "On record: 3 of them." in block


def test_what_a_turn_is_about_is_read_by_her_models_states_once(tmp_path, monkeypatch):
    """The same states are read for what a turn asks and for what it is about."""
    asked: list[list[str]] = []
    kinds = {key: i for i, key in enumerate([NONE, *[way.key for way in how.WAYS], how.HER_STATE])}
    where = {}

    class Client:
        model_path = str(tmp_path / "model")

        async def states_at_turn(self, texts, in_a_turn=False):
            asked.append(list(texts))
            rng, out = np.random.default_rng(3), []
            for text in texts:
                row = rng.normal(0, 0.05, (2, 12))
                row[:, where.get(text, 0)] += 3.0
                out.append(row.astype(np.float32))
            return out

    reads = WhatOfHers(root=tmp_path)
    for kind, text in reads.examples():
        where[text] = kinds[kind]
    fitted_on_her_model._JUST_READ.clear()

    async def saved(self, model, states):
        return None

    monkeypatch.setattr(fitted_on_her_model.FittedOnHerModel, "_save", saved)
    from core.understanding.examples import transcript

    where[transcript("you got mated in twelve")] = kinds["play"]

    async def go():
        client = Client()
        assert await reads.learn(client)
        before = len(asked)
        first = await reads.read("you got mated in twelve", client=client)
        again = await reads.read("you got mated in twelve", client=client)
        return first, again, len(asked) - before

    first, again, passes = asyncio.run(go())
    fitted_on_her_model._JUST_READ.clear()
    assert first == again == ["play"] and passes == 1


def test_the_reading_is_taken_with_the_turn():
    import inspect

    from interface.routes import chat_what_is_asked

    source = inspect.getsource(chat_what_is_asked.read_what_this_turn_asks)
    assert "get_what_of_hers().read(" in source and "hold_ways(None)" in source


def test_a_turn_about_how_she_is_is_one_of_the_kinds_and_is_no_way_of_hers():
    kinds = {key for key, _said, _before in how.examples()}
    assert how.HER_STATE in kinds and how.HER_STATE not in {way.key for way in how.WAYS}
    assert how.about_her_state() is None
    try:
        how.hold_ways([how.HER_STATE])
        assert how.about_her_state() is True and how.ways_about() == [] and how.how_she_does_it_block() == ""
        how.hold_ways([])
        assert how.about_her_state() is False
    finally:
        how.hold_ways(None)


def test_the_same_words_are_about_her_or_not_by_what_came_before_them():
    by_said: dict[str, set[str]] = {}
    for key, said, _before in how.examples():
        by_said.setdefault(said, set()).add(key)
    assert by_said["and you?"] == {how.HER_STATE, ""}
