"""She sets out on an objective of her own, takes its steps by the parts of her that take each sort of act, and ends it on
what they came to (F22).

LIVE 27 November 2026 her own goals were what her drives named, and "Acting on relief: stabilize" was worked seventeen
times for less than nothing each time. Bryan: "We need better strategic intelligence as well. In terms of being able to
independently construct and sustain a complex strategic objective".
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency import her_objectives as objectives
from core.agency.her_objectives import AStep, Doers, HowSheGoesOn, WhatSheSetsOut
from core.agency.projects import Lifecycle, ProjectLedger
from core.agency.what_she_does_next import CameOf

pytestmark = pytest.mark.unit

WORDS = ("I keep losing what I learn between restarts. I want to understand how other systems keep what they learn, then "
         "build a small tool that checks my own records survive a restart.")
STEPS = [AStep(kind="find_out", what="I will find out how other systems keep what they learn across restarts."),
         AStep(kind="find_out", what="I will find out how to bake bread."),
         AStep(kind="make", what="I will make a tool that checks my records survive a restart.", after=[1, 2])]
SET_OUT = WhatSheSetsOut(sets_out=True, what="I want to keep what I learn across my restarts.", serves="knowing myself",
                         why="I lose it.", steps=STEPS, done_when="A tool shows that my records survive a restart.",
                         not_worth_it_if="My records already survive every restart.", otherwise=["practise instead"])


def _reader(readings: dict[str, float]):
    """Her reader of words, standing in: how far any words bear out each statement, as set; 0 for what is not set."""
    return lambda _words, statement: readings.get(statement, 0.0)


BORNE = _reader({SET_OUT.what: 0.9, STEPS[0].what: 0.8, STEPS[1].what: -0.7, STEPS[2].what: 0.6})


def _hers():
    return SimpleNamespace(chooser=SimpleNamespace(how_she_chooses=lambda: {"kinds": {"make": {"done": 3, "came_to": -1.0}}}), found=[])


def _thinks(*answers):
    asked: list[tuple[str, str, object]] = []
    queue = list(answers)

    async def thinks(facts, question, shape):
        asked.append((facts, question, shape))
        return queue.pop(0)

    thinks.asked = asked
    return thinks


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    import core.conation.what_she_wants as wants

    monkeypatch.setattr(objectives, "answers_kept_at", lambda: tmp_path / "answers.jsonl")

    monkeypatch.setattr(wants, "get_what_she_wants", lambda: SimpleNamespace(wants=lambda limit: [SimpleNamespace(label="knowing myself", why="it matters")]))
    return ProjectLedger(path=tmp_path / "ledger.jsonl")


def _set_out(ledger, readings=BORNE, shaped=SET_OUT):
    thinks = _thinks((WORDS, shaped))
    came = asyncio.run(objectives.set_out(_hers(), thinks, ledger=ledger, bears_out=readings))
    return came, thinks


def test_she_sets_out_on_what_her_own_words_bear_out_and_leaves_out_a_step_they_do_not(ledger):
    came, thinks = _set_out(ledger)
    project = objectives.standing(ledger)
    assert project is not None and project.lifecycle == Lifecycle.ACTIVE and project.thesis == SET_OUT.what
    assert [m.description for m in project.milestones] == [STEPS[0].what, STEPS[2].what]
    assert project.left_out and project.left_out[0]["what"] == STEPS[1].what
    # The step left out is not waited on.
    assert project.milestones[1].after == [project.milestones[0].id]
    assert project.formed_from == WORDS and project.origin_drive == "knowing myself"
    assert project.acceptance_criteria[0] == SET_OUT.done_when
    assert came.got == pytest.approx(2 / 3)
    facts, question, shape = thinks.asked[0]
    assert question == objectives.THE_QUESTION and shape is WhatSheSetsOut
    assert "knowing myself" in facts and "making something of your own" in facts, "she sets out knowing what her acts came to"


def test_nothing_is_set_out_on_where_she_wants_nothing_or_her_words_do_not_bear_it_out(ledger):
    came, _ = _set_out(ledger, shaped=WhatSheSetsOut(sets_out=False))
    assert came.got == 0.0 and objectives.standing(ledger) is None
    came, _ = _set_out(ledger, readings=_reader({SET_OUT.what: -0.4}))
    assert "did not bear out" in came.what and objectives.standing(ledger) is None
    away = asyncio.run(objectives.set_out(_hers(), _thinks(None), ledger=ledger, bears_out=BORNE))
    assert away.put_down and not ledger.all()


def _doers(results: dict[str, list[tuple[str, CameOf]]], stands: dict[str, tuple[str, str]]):
    taken: list[tuple[str, str, list[str]]] = []

    def does(kind):
        async def one(key, what, given):
            taken.append((kind, what, list(given)))
            return results[kind].pop(0)
        return one

    return Doers(does={kind: does(kind) for kind in ("find_out", "make", "change")},
                 stands=lambda kind, key: stands.get(key, (objectives.GOING, ""))), taken


def test_with_nothing_set_out_on_she_is_offered_setting_out(ledger):
    doers, _ = _doers({}, {})
    options = asyncio.run(objectives.objectives_of_hers(_hers(), doers, _thinks(), ledger=ledger, bears_out=BORNE))
    assert [o.id for o in options] == ["objective:new"] and options[0].kind == "objective"


def test_a_step_is_offered_once_what_it_waits_on_is_done_and_is_done_only_when_what_takes_it_says_so(ledger):
    _set_out(ledger)
    project = objectives.standing(ledger)
    first, second = project.milestones
    doers, taken = _doers({"find_out": [("inq", CameOf(got=0.5, what="asked one, answered: they write a journal"))] * 2,
                           "make": [("req", CameOf(got=1.0, what="made it"))]}, {})
    offered = asyncio.run(objectives.objectives_of_hers(_hers(), doers, _thinks(), ledger=ledger, bears_out=BORNE))
    assert [o.id for o in offered] == [f"objective:{project.project_id}:{first.id}"], "the make waits on the finding out"
    came = asyncio.run(offered[0].does())
    held = ledger.get(project.project_id)
    assert came.got == 0.5 and held.milestones[0].completed_at is None, "a pass is not the step done"
    assert held.milestones[0].key == "inq" and held.milestones[0].tries == 1
    # What takes it says it is done.
    done, _ = _doers({"find_out": [("inq", CameOf(got=1.0, what="every question answered: they write a journal"))],
                      "make": [("req", CameOf(got=1.0, what="made it"))]}, {"inq": (objectives.DONE, "https://example.org/journal")})
    asyncio.run(objectives.take_a_step(project.project_id, first.id, done, ledger=ledger, bears_out=BORNE))
    held = ledger.get(project.project_id)
    assert held.milestones[0].completed_at is not None and "https://example.org/journal" in held.artifacts
    offered = asyncio.run(objectives.objectives_of_hers(_hers(), done, _thinks(), ledger=ledger, bears_out=BORNE))
    assert [o.id for o in offered] == [f"objective:{project.project_id}:{second.id}"]


def test_what_a_step_waited_on_came_to_is_given_to_it(ledger):
    _set_out(ledger)
    project = objectives.standing(ledger)
    first, second = project.milestones
    doers, taken = _doers({"find_out": [("inq", CameOf(got=1.0, what="answered: they write a journal"))],
                           "make": [("req", CameOf(got=0.4, what="begun"))]}, {"inq": (objectives.DONE, "a page")})
    asyncio.run(objectives.take_a_step(project.project_id, first.id, doers, ledger=ledger, bears_out=BORNE))
    asyncio.run(objectives.take_a_step(project.project_id, second.id, doers, ledger=ledger, bears_out=BORNE))
    assert taken[-1][0] == "make" and taken[-1][2] == [f"{first.description}: answered: they write a journal"]


def test_a_step_that_comes_to_nothing_twice_is_thought_over_and_she_may_go_no_further(ledger):
    _set_out(ledger)
    project = objectives.standing(ledger)
    first = project.milestones[0]
    doers, _ = _doers({"find_out": [("inq", CameOf(got=0.0, what="nothing read answers it"))] * 2}, {})
    for _ in range(2):
        asyncio.run(objectives.take_a_step(project.project_id, first.id, doers, ledger=ledger, bears_out=BORNE))
    offered = asyncio.run(objectives.objectives_of_hers(_hers(), doers, _thinks(), ledger=ledger, bears_out=BORNE))
    assert [o.id for o in offered] == [f"objective:{project.project_id}:over"] and "2 passes" in offered[0].why
    stop = "Nothing I read helps, and my records already hold. I will stop here."
    thinks = _thinks((stop, HowSheGoesOn(goes_on="no further", why="my records already hold")))
    came = asyncio.run(objectives.go_on_or_not(project.project_id, thinks, offered[0].why, ledger=ledger,
                                               bears_out=_reader({objectives.GOING_ON: -0.8})))
    assert came.got == 1.0 and ledger.get(project.project_id).lifecycle == Lifecycle.ABANDONED
    assert objectives.standing(ledger) is None


def test_where_her_words_and_her_choice_disagree_nothing_is_changed(ledger):
    _set_out(ledger)
    project = objectives.standing(ledger)
    thinks = _thinks(("I will keep at it.", HowSheGoesOn(goes_on="no further")))
    came = asyncio.run(objectives.go_on_or_not(project.project_id, thinks, "why", ledger=ledger, bears_out=_reader({objectives.GOING_ON: 0.7})))
    assert came.got == 0.0 and ledger.get(project.project_id).lifecycle == Lifecycle.ACTIVE


def test_she_may_change_its_steps_from_here_and_what_was_done_stays_done(ledger):
    _set_out(ledger)
    project = objectives.standing(ledger)
    first, second = project.milestones
    done, _ = _doers({"find_out": [("inq", CameOf(got=1.0, what="answered"))]}, {"inq": (objectives.DONE, "a page")})
    asyncio.run(objectives.take_a_step(project.project_id, first.id, done, ledger=ledger, bears_out=BORNE))
    new = AStep(kind="change", what="I will change how my state is written so it survives a restart.")
    thinks = _thinks(("I should change my own source instead.", HowSheGoesOn(goes_on="with its steps changed", steps=[new], why="a tool only reports it")))
    came = asyncio.run(objectives.go_on_or_not(project.project_id, thinks, "why", ledger=ledger,
                                               bears_out=_reader({objectives.GOING_ON: 0.6, new.what: 0.7})))
    held = ledger.get(project.project_id)
    assert came.got == 1.0 and held.revisions and held.revisions[-1].reason == "a tool only reports it"
    assert held.milestones[0].completed_at is not None
    assert held.milestones[1].set_aside.startswith("changed") and held.milestones[2].description == new.what
    assert held.milestones[2].id not in {first.id, second.id}


def test_what_she_said_would_end_it_is_read_against_each_pass(ledger):
    _set_out(ledger)
    project = objectives.standing(ledger)
    first = project.milestones[0]
    doers, _ = _doers({"find_out": [("inq", CameOf(got=1.0, what="answered: her records are already written atomically"))]}, {})
    asyncio.run(objectives.take_a_step(project.project_id, first.id, doers, ledger=ledger,
                                       bears_out=_reader({SET_OUT.not_worth_it_if: 0.8})))
    held = ledger.get(project.project_id)
    assert held.to_think_over.endswith(SET_OUT.not_worth_it_if)
    offered = asyncio.run(objectives.objectives_of_hers(_hers(), doers, _thinks(), ledger=ledger, bears_out=BORNE))
    assert [o.id for o in offered] == [f"objective:{project.project_id}:over"]


def test_it_ends_done_only_where_what_its_steps_came_to_bears_out_what_she_said_will_be_so(ledger):
    _set_out(ledger)
    project = objectives.standing(ledger)
    doers, _ = _doers({"find_out": [("inq", CameOf(got=1.0, what="answered"))], "make": [("req", CameOf(got=1.0, what="made it, and its checks hold"))]},
                      {"inq": (objectives.DONE, "a page"), "req": (objectives.DONE, "a tool")})
    for m in project.milestones:
        asyncio.run(objectives.take_a_step(project.project_id, m.id, doers, ledger=ledger, bears_out=BORNE))
    offered = asyncio.run(objectives.objectives_of_hers(_hers(), doers, _thinks(), ledger=ledger, bears_out=BORNE))
    assert [o.id for o in offered] == [f"objective:{project.project_id}:end"]
    came = asyncio.run(objectives.see_whether_it_is_so(project.project_id, _thinks(), ledger=ledger, bears_out=_reader({SET_OUT.done_when: -0.5})))
    held = ledger.get(project.project_id)
    assert came.got == 0.0 and held.lifecycle == Lifecycle.ACTIVE and held.to_think_over
    held.to_think_over = ""
    looked = _thinks(("It works, and next time I would find out less before making.", None))
    came = asyncio.run(objectives.see_whether_it_is_so(project.project_id, looked, ledger=ledger, bears_out=_reader({SET_OUT.done_when: 0.7})))
    held = ledger.get(project.project_id)
    assert came.got == 1.0 and held.lifecycle == Lifecycle.COMPLETED
    assert held.reflections[-1].startswith("It works") and looked.asked[0][1] == objectives.THE_QUESTION_AT_THE_END
    # What came of it is what she sets out knowing the next time.
    assert "It works" in objectives.what_she_has_to_go_on(_hers(), ledger)


def test_what_she_set_out_to_do_is_read_back_whole_after_a_restart(ledger, tmp_path):
    _set_out(ledger)
    project = objectives.standing(ledger)
    doers, _ = _doers({"find_out": [("inq", CameOf(got=0.5, what="asked one"))]}, {})
    asyncio.run(objectives.take_a_step(project.project_id, project.milestones[0].id, doers, ledger=ledger, bears_out=BORNE))
    again = ProjectLedger(path=tmp_path / "ledger.jsonl")
    held = objectives.standing(again)
    assert held is not None and held.project_id == project.project_id
    first, second = held.milestones
    assert (first.kind, first.key, first.tries, first.came) == ("find_out", "inq", 1, ["asked one"])
    assert second.after == [first.id] and held.not_worth_it_if == SET_OUT.not_worth_it_if and held.left_out


class _CutsWhatDoesNotFit:
    """A reader of entailment standing in for hers: it reads as much of the premise as fits beside the statement and cuts the
    rest, as hers does, and finds a statement borne out where its words are in what it read."""

    max_length = 40
    tokenizer = SimpleNamespace(tokenize=str.split, model_max_length=40)

    def predict(self, pairs):
        import numpy as np

        out = []
        for premise, statement in pairs:
            read = " ".join(premise.split()[: self.max_length - len(statement.split()) - 3])
            denied = statement.startswith("It is not true that ")
            inner = statement[len("It is not true that "):] if denied else statement
            there = inner.rstrip(".").lower() in read.lower()
            # contradiction, entailment, neutral
            out.append([3.0, 0.0, 0.0] if there and denied else [0.0, 3.0, 0.0] if there else [0.0, 0.0, 3.0])
        return np.asarray(out)


def test_a_step_named_late_in_long_words_is_read_in_a_passage_of_its_own(monkeypatch):
    from core.self import how_her_words_stand as reader

    monkeypatch.setattr(reader, "_reader", lambda: (_CutsWhatDoesNotFit(), {"contradiction": 0, "entailment": 1, "neutral": 2}))
    filler = " ".join(f"On day {n} it rained in a town I have never seen." for n in range(12))
    step = "I will build a tool that checks my records survive."
    words = f"{filler} Last of all, {step[0].lower()}{step[1:]}"
    assert (reader.whether_her_words_bear_it_out(words, step) or 0.0) <= 0.0, "read whole, the reader cut the step off"
    assert reader.what_her_words_say_of(words, step) == 1
    assert reader.what_her_words_say_of(filler, step) == 0


def test_she_sets_out_knowing_what_she_is_in_the_middle_of_and_what_has_gone_wrong(ledger, monkeypatch):
    """LIVE 27 November 2026, 23:09: given what she wanted, what she found and what her acts came to, the most she wanted
    was "parameter", and she set out on nothing in three tokens."""
    from core.agency import her_goal_at_work as goals
    from core.skills import her_draft

    record = goals.GoalRecord(goal="Acting on relief: stabilize", slug="acting", status=goals.WORKING, passes=11,
                              said=[{"words": "I have no live RSS or process probe. <tool_call> <function=read_file>", "at": 1.0},
                                    {"judged": {"why": "kept"}, "at": 2.0}])
    monkeypatch.setattr(goals, "all_records", lambda: [record, goals.GoalRecord(goal="done one", slug="d", status=goals.MERGED)])
    monkeypatch.setattr(her_draft, "faults_recorded", lambda subsystem="": (["4x warning memory: grew at lanes.py:3"], 4))
    facts = __import__("json").loads(objectives.what_she_has_to_go_on(_hers(), ledger))
    assert facts["your_goals_under_way"] == [{"goal": "Acting on relief: stabilize", "stands": "working", "passes": 11,
                                              "what_you_said_of_it_last": "I have no live RSS or process probe."}]
    assert facts["what_has_gone_wrong_in_your_running_most_often_first"] == ["4x warning memory: grew at lanes.py:3"]


def test_each_way_of_setting_out_on_nothing_says_which_it_was_and_her_answer_is_kept(ledger, tmp_path):
    """LIVE 27 November 2026, 23:35: she wrote 571 tokens and an answer in its shape, it came to "there was nothing she set
    out to do", and nothing kept what she had said."""
    import json

    declined = "Not now. I am in the middle of mending my lane control."
    came = asyncio.run(objectives.set_out(_hers(), _thinks(("", WhatSheSetsOut(sets_out=False))), ledger=ledger, bears_out=BORNE))
    assert came.what == "she set out on nothing now"
    came = asyncio.run(objectives.set_out(_hers(), _thinks((declined, WhatSheSetsOut(sets_out=False))), ledger=ledger, bears_out=BORNE))
    assert came.what == "she set out on nothing now: “Not now.”"
    came, _ = _set_out(ledger, shaped=WhatSheSetsOut(sets_out=True))
    assert came.what == "she would set out on something, and named nothing to bring about"
    came, _ = _set_out(ledger, shaped=WhatSheSetsOut(sets_out=True, what=SET_OUT.what))
    assert came.what.endswith("and named no step she can take")
    kept = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert len(kept) == 4 and kept[1]["words"] == declined and kept[1]["shaped"]["sets_out"] is False
    assert kept[1]["question"] == objectives.THE_QUESTION and kept[1]["came"] == "she set out on nothing now: “Not now.”"
