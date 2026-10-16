"""She learns a kind of problem from solved examples she meets, and a known answer can retire what she kept."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.learning import examples_she_meets as meets
from core.learning.kinds_by_meaning import meanings_of
from core.learning.procedures_from_solved_examples import (
    KEPT_BOOK_RELATIVE,
    ProcedureBook,
    Signature,
    dumps,
)
from core.sandbox.untrusted_python import available_boundary

needs_boundary = pytest.mark.skipif(not available_boundary(), reason="no OS sandbox on this host")
# Eleven kinds in her book; met problems are of a twelfth, "orders", which she does not keep.
KEPT = ("percent", "dates", "sorting", "units", "primes", "brackets", "routes", "tables", "clocks", "shapes",
        "letters")
AXES = {word: axis for axis, word in enumerate((*KEPT, "orders"))}

TOTAL = "def solve(problem):\n    a, b = [int(w) for w in problem.split() if w.isdigit()][:2]\n    return str(a + b)\n"
WRONG = "def solve(problem):\n    a, b = [int(w) for w in problem.split() if w.isdigit()][:2]\n    return str(a * b)\n"


def _embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        row = np.full(len(AXES), 0.05, dtype=np.float32)
        for word, axis in AXES.items():
            if word in text:
                row[axis] = 1.0
        rows.append(row)
    return np.asarray(rows)


@pytest.fixture
def her_data(tmp_path, monkeypatch):
    from core.learning import procedures_from_solved_examples as procedures

    monkeypatch.setattr("core.utils.paths.aura_data_dir", lambda: tmp_path)
    procedures._kept_cache.clear()
    kinds = {k: ([f"{k} problem {i}" for i in range(4)], [f"{k} sealed {i}" for i in range(4)]) for k in KEPT}
    book = ProcedureBook({k: Signature({k: 1.0}, 0.99) for k in KEPT}, {}, {},
                         meanings=meanings_of(kinds, documents=_embed, queries=_embed))
    path = tmp_path / KEPT_BOOK_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(dumps(book.to_json()), encoding="utf-8")
    yield tmp_path
    procedures._kept_cache.clear()


def _teach(count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        assert meets.note_met(f"orders total of {i} and {i + 3}", str(2 * i + 3), "a_person_gave_it")


def test_only_a_source_that_knows_the_answer_is_noted_and_the_first_answer_stands(her_data) -> None:
    assert not meets.note_met("orders total of 1 and 2", "3", "her_own_draft")
    assert meets.note_met("orders total of 1 and 2", "3", "a_person_gave_it")
    assert meets.note_met("orders  total of 1 and 2", "4", "a_person_gave_it")
    rows = meets.met_examples()
    assert len(rows) == 1 and rows[0]["answer"] == "3" and rows[0]["source"] == "a_person_gave_it"


def test_met_problems_group_by_meaning_in_the_order_met() -> None:
    rows = [{"key": str(i), "problem": text} for i, text in
            enumerate(["orders a", "percent a", "orders b", "percent b", "dates a"])]
    groups = meets.groups_of(rows, 0.9, documents=_embed, queries=_embed)
    assert [[r["key"] for r in g] for g in groups] == [["0", "2"], ["1", "3"], ["4"]]
    assert meets.groups_of(rows[:3], 0.9, documents=_embed, queries=_embed)[0] == groups[0]


@needs_boundary
@pytest.mark.asyncio
async def test_a_kind_she_met_is_learned_kept_and_answers_from_then_on(her_data) -> None:
    _teach(14)
    asked: list[str] = []

    async def propose_one(request: str) -> str:
        asked.append(request)
        return f"```python\n{TOTAL}```"

    out = await meets.learn_one_step(propose_one, documents=_embed, queries=_embed)
    assert out["status"] == "kept", out
    book = ProcedureBook.from_json(json.loads((her_data / KEPT_BOOK_RELATIVE).read_text()))
    name = out["group"]
    assert book.procedures[name] == [TOTAL] and book.agreed[name] == 14 and name in book.meanings
    assert book.answer_restated(name, "orders total of 20 and 22")["answer"] == "42"
    assert len(asked) == 1 and "orders total of" in asked[0]

    # One more known answer of the kind: checked, agreed, counted.
    meets.note_met("orders total of 30 and 33", "63", "a_person_gave_it")
    out = await meets.learn_one_step(propose_one, documents=_embed, queries=_embed)
    assert out["agreed_with_kept"] == 1 and len(asked) == 1
    assert ProcedureBook.from_json(json.loads((her_data / KEPT_BOOK_RELATIVE).read_text())).agreed[name] == 15

    # A known answer it disagrees with retires it, with the counterexample kept.
    meets.note_met("orders total of 40 and 44", "85", "a_person_gave_it")
    out = await meets.learn_one_step(propose_one, documents=_embed, queries=_embed)
    assert out["retired"] == [name]
    book = ProcedureBook.from_json(json.loads((her_data / KEPT_BOOK_RELATIVE).read_text()))
    assert name not in book.procedures
    retired = json.loads((her_data / meets.RETIRED_RELATIVE).read_text().splitlines()[0])
    assert retired["known"] == "85" and retired["returned"] == "84" and retired["procedures"] == [TOTAL]


@needs_boundary
@pytest.mark.asyncio
async def test_a_wrong_function_is_repaired_from_its_counterexamples_on_the_next_step(her_data) -> None:
    _teach(14)
    asked: list[str] = []

    async def propose_one(request: str) -> str:
        asked.append(request)
        return f"```python\n{TOTAL if 'Where it was wrong' in request else WRONG}```"

    first = await meets.learn_one_step(propose_one, documents=_embed, queries=_embed)
    assert first["status"] == "requested" and first["agreed"][0] < 14
    second = await meets.learn_one_step(propose_one, documents=_embed, queries=_embed)
    assert second["status"] == "kept" and "Where it was wrong" in asked[1] and WRONG.strip() in asked[1]


@needs_boundary
@pytest.mark.asyncio
async def test_a_group_whose_rounds_are_spent_waits_until_it_grows(her_data) -> None:
    _teach(14)
    asked: list[str] = []

    async def propose_one(request: str) -> str:
        asked.append(request)
        return "no function here"

    for _ in range(meets.ROUNDS):
        assert (await meets.learn_one_step(propose_one, documents=_embed, queries=_embed))["status"] == "requested"
    assert (await meets.learn_one_step(propose_one, documents=_embed, queries=_embed))["status"] == "no_group_ready"
    _teach(1, start=50)
    assert (await meets.learn_one_step(propose_one, documents=_embed, queries=_embed))["status"] == "requested"
    assert len(asked) == meets.ROUNDS + 1


@pytest.mark.asyncio
async def test_nothing_is_grouped_without_her_embedder(her_data) -> None:
    _teach(14)

    async def propose_one(request: str) -> str:
        raise AssertionError("no request without a group")

    out = await meets.learn_one_step(propose_one, documents=None, queries=None)
    assert out["status"] == "no_embedder_held" and out["unanswered"] == 14


@pytest.mark.asyncio
async def test_what_independent_programs_agreed_on_is_noted_once_and_only_as_one_line(her_data) -> None:
    from core.brain.reasoning_amplifier_v2 import _note_what_independent_programs_agreed

    await _note_what_independent_programs_agreed("What is 6 times 7?", "42", {})
    await _note_what_independent_programs_agreed("Explain 6 times 7.", "It is 42.\nBecause...", {})
    await _note_what_independent_programs_agreed("Give JSON", "{\"a\": 1}", {"response_contract": "json"})
    rows = meets.met_examples()
    assert [(r["problem"], r["answer"], r["source"]) for r in rows] == [
        ("What is 6 times 7?", "42", "independent_programs_agreed")]


@pytest.mark.asyncio
async def test_the_conductor_runs_one_step_in_a_task_of_its_own(her_data, monkeypatch) -> None:
    import asyncio

    from core.brain import reasoning_background as background

    registered: dict[str, object] = {}

    class Conductor:
        def register(self, name, interval, fn, **kwargs):
            registered[name] = (interval, fn, kwargs)

    assert background.register_reasoning_jobs(Conductor())
    interval, job, kwargs = registered["learn_kinds_she_meets"]
    assert job is background._job_learn_kinds_she_meets and kwargs["policy"] == "research"

    class Router:
        async def think(self, **kwargs):
            raise AssertionError("nothing was met, so nothing is asked")

    monkeypatch.setattr("core.container.ServiceContainer.get", lambda name, default=None: Router()
                        if name == "llm_router" else default)
    monkeypatch.setattr(background, "_her_embedder", lambda: (None, None))
    background._LEARNING.pop("task", None)
    assert (await job())["status"] == "step_started"
    assert (await job())["status"] == "step_running"
    result = await asyncio.wait_for(background._LEARNING["task"], 30)
    assert result["status"] == "nothing_met"


@needs_boundary
@pytest.mark.asyncio
async def test_a_kind_found_only_by_meaning_is_not_retired_by_what_it_cannot_read(her_data) -> None:
    """Live chat, 16 October: a person's answer to a request in other words would have retired a right procedure."""
    from core.learning import procedures_from_solved_examples as procedures

    path = her_data / KEPT_BOOK_RELATIVE
    book = ProcedureBook.from_json(json.loads(path.read_text()))
    book.procedures["percent"], book.agreed["percent"] = [WRONG], 40
    path.write_text(dumps(book.to_json()), encoding="utf-8")
    procedures._kept_cache.clear()
    # Worded unlike the kind's problems, so its words reach no kind; its meaning reaches "percent".
    meets.note_met("what is 15 percent of 340 here", "51", "a_person_gave_it")
    assert book.route("what is 15 percent of 340 here")[0] is None  # not by its words

    async def propose_one(_request: str) -> str:
        return ""

    out = await meets.learn_one_step(propose_one, documents=_embed, queries=_embed)
    # The procedure answers 5100 for it, by meaning only: not a counterexample in the kind's own words.
    assert out["retired"] == []
    assert ProcedureBook.from_json(json.loads(path.read_text())).procedures["percent"] == [WRONG]
    assert json.loads((her_data / meets.LEARNING_RELATIVE).read_text())["checked"] != {}
