"""A request reaches a kept kind by what it means, is restated in the kind's form, and is answered as evidence."""

from __future__ import annotations

import types

import numpy as np
import pytest

from core.learning.kinds_by_meaning import (
    KindMeaning,
    meanings_of,
    nearest_kind,
    restatement_request,
)
from core.learning.procedures_from_solved_examples import ProcedureBook, Signature
from core.sandbox.untrusted_python import available_boundary

needs_boundary = pytest.mark.skipif(not available_boundary(), reason="no OS sandbox on this host")
# Enough kinds for one to stand out from the rest, as her book has (26 and more).
AXES = {word: axis for axis, word in enumerate(
    ("percent", "dates", "sorting", "units", "primes", "brackets", "routes", "tables", "clocks", "shapes",
     "letters", "orders"))}


def _embed(texts: list[str]) -> np.ndarray:
    """A stand-in embedding: each text points along its kind's axis, with a little of a second."""
    rows = []
    for text in texts:
        row = np.full(len(AXES), 0.05, dtype=np.float32)
        for word, axis in AXES.items():
            if word in text:
                row[axis] = 1.0
        rows.append(row)
    return np.asarray(rows)


def _meanings() -> dict[str, KindMeaning]:
    kinds = {kind: ([f"{kind} problem {i}" for i in range(4)], [f"{kind} sealed {i}" for i in range(4)])
             for kind in AXES}
    return meanings_of(kinds, documents=_embed, queries=_embed)


def test_a_request_goes_to_the_kind_it_clearly_means_and_nowhere_when_it_means_none() -> None:
    meanings = _meanings()
    assert set(meanings) == set(AXES) and all(0.9 < m.nearest_required <= 1.0 for m in meanings.values())
    assert nearest_kind(_embed(["what's 15 percent of 340"])[0], meanings)[0] == "percent"
    kind, _score, why = nearest_kind(_embed(["tell me about your day"])[0], meanings)
    assert kind is None and why
    assert nearest_kind(_embed(["percent"])[0], dict(list(meanings.items())[:2]))[2] == \
        "too_few_kinds_to_tell_meaning_apart"


def test_the_restatement_shows_the_kinds_form_and_carries_the_request() -> None:
    text = restatement_request("what's 15% of 340?", [{"problem": "Calculate 0.15*340.", "answer": "51"}])
    assert "Calculate 0.15*340." in text and text.rstrip().endswith("what's 15% of 340?") and "51" not in text


SOLVE = "def solve(problem):\n    a, b = problem.split('Calculate ')[1].rstrip('.').split('*')\n" \
        "    return str(round(float(a) * float(b), 6)).rstrip('0').rstrip('.')\n"


def _book() -> ProcedureBook:
    signatures = {kind: Signature({kind: 1.0}, 0.99) for kind in AXES}
    return ProcedureBook(signatures, {"percent": [SOLVE]}, {"percent": 40}, meanings=_meanings(),
                         examples={"percent": [{"problem": "Calculate 0.2*50.", "answer": "10"}]})


@needs_boundary
def test_a_book_answers_a_restated_request_and_keeps_its_meanings_through_saving() -> None:
    book = ProcedureBook.from_json(_book().to_json())
    assert set(book.meanings) == set(AXES) and book.examples["percent"][0]["answer"] == "10"
    receipt = book.answer_restated("percent", "Calculate 0.15*340.")
    assert receipt["answer"] == "51" and receipt["form"] == "restated" and receipt["agreed_known"] == 40
    assert book.answer_restated("no such kind", "x")["declined"] == "no_such_kind"


@needs_boundary
@pytest.mark.asyncio
async def test_her_turn_finds_the_kind_by_meaning_restates_it_and_keeps_her_words(monkeypatch) -> None:
    from core.phases.response_generation import ResponseGenerationPhase
    from core.state.aura_state import AuraState

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures", lambda *a, **k: None)
    monkeypatch.setattr("core.learning.procedures_from_solved_examples.kept_procedure_book", _book)
    monkeypatch.setattr("core.phases.kept_procedures_in_a_turn.her_query_vector", lambda text, task: _embed([text])[0])
    asked: list[dict] = []

    class Router:
        async def think(self, **kwargs):
            asked.append(kwargs)
            return "Calculate 0.15*340."

    phase, state = ResponseGenerationPhase(types.SimpleNamespace()), AuraState.default()
    assert await phase._consult_kept_procedures(state=state, objective="what's 15 percent of 340?",
                                                router=Router(), origin="desktop", tier="primary")
    planted = state.response_modifiers["kept_procedure"]
    assert planted["answer"] == "51" and planted["form"] == "restated"
    assert asked and asked[0]["clean_user_surface_contract"] is True and asked[0]["temperature"] == 0.0
    assert "Calculate 0.2*50." in asked[0]["messages"][0]["content"]
    assert "Calculate 0.15*340." in state.cognition.working_memory[-1]["content"]

    async def no_search(*_a, **_k):
        raise AssertionError("the answer was already in front of her")

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.amplify_turn", no_search)
    draft = "Fifteen percent of 340 is 52, I think."
    out = await phase._maybe_amplify_response(
        objective="what's 15 percent of 340?", draft=draft, router=None, state=state, request_timeout=20.0,
        origin="desktop", tier="primary", runtime_context={}, is_user_facing=True, is_background=False,
        proof_or_benchmark=False)
    assert str(out) == draft
    assert state.response_modifiers["reasoning_amplifier_v2_active_phase"]["voice"] == "evidence_for_a_restated_request"


@pytest.mark.asyncio
async def test_without_her_embedder_held_nothing_is_read_by_meaning(monkeypatch) -> None:
    from core.phases.response_generation import ResponseGenerationPhase
    from core.state.aura_state import AuraState

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures", lambda *a, **k: None)
    monkeypatch.setattr("core.learning.procedures_from_solved_examples.kept_procedure_book", _book)
    monkeypatch.setattr("core.phases.kept_procedures_in_a_turn.her_query_vector", lambda text, task: None)

    class Router:
        async def think(self, **kwargs):
            raise AssertionError("no restatement without a meaning")

    phase, state = ResponseGenerationPhase(types.SimpleNamespace()), AuraState.default()
    assert not await phase._consult_kept_procedures(state=state, objective="what's 15 percent of 340?",
                                                    router=Router(), origin="desktop", tier="primary")
    assert "kept_procedure" not in state.response_modifiers


@needs_boundary
@pytest.mark.asyncio
async def test_the_phase_her_kernel_runs_finds_the_kind_by_meaning_and_shows_her_the_answer(monkeypatch) -> None:
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures", lambda *a, **k: None)
    monkeypatch.setattr("core.learning.procedures_from_solved_examples.kept_procedure_book", _book)
    monkeypatch.setattr("core.phases.kept_procedures_in_a_turn.her_query_vector",
                        lambda text, task: _embed([text])[0])

    class Router:
        async def think(self, **kwargs):
            return "Calculate 0.15*340."

    phase, state = UnitaryResponsePhase(types.SimpleNamespace()), AuraState.default()
    asked = "what's 15 percent of 340?"
    await phase._consult_kept_procedures(state, asked, llm=Router(), origin="desktop", tier="primary")
    evidence = UnitaryResponsePhase._build_active_grounding_message(state, asked, None)
    assert evidence is not None and "Answer: 51" in evidence["content"]
    assert "Calculate 0.15*340." in evidence["content"]
    # Another request's turn is not shown this answer.
    assert UnitaryResponsePhase._kept_procedure_evidence(state, "what's 20 percent of 50?") is None


@pytest.mark.asyncio
async def test_in_the_phase_her_kernel_runs_her_words_stand_when_they_state_the_kept_answer(monkeypatch) -> None:
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    async def no_search(*_a, **_k):
        raise AssertionError("a kept answer was in front of her; no search runs")

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.amplify_turn", no_search)
    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures", lambda *a, **k: None)
    asked = "A Dyck sequence closes at which step?"
    common = {"objective": asked, "llm": None, "request_timeout": 20.0, "is_user_facing": True,
              "is_background": False, "proof_or_benchmark": False}
    for draft, form, served in (
        ("I traced every bracket.\n\nIt closes at step 18.", "words", None),
        ("I traced it.\n\nIt closes at step 17.", "words", "18"),
        ("I traced it.\n\nIt closes at step 17.", "restated", None),
    ):
        state = AuraState.default()
        state.response_modifiers["kept_procedure"] = {"objective": asked, "answer": "18", "form": form,
                                                      "receipt": {"strategy_used": "kept_procedure"}}
        holder = types.SimpleNamespace(_last_reasoning_receipt=None)
        out = await UnitaryResponsePhase._maybe_amplify_response(holder, draft=draft, state=state, **common)
        assert out == (served or draft)


def test_a_restatement_is_the_problem_asked_as_far_as_its_numbers_say() -> None:
    from core.learning.kinds_by_meaning import restates

    examples = [{"problem": "Calculate the highest common divisor of 3672 and 5033394."},
                {"problem": "What is the highest common factor of 15 and 27362880?"}]
    asked = "What's the greatest common divisor of 4862 and 7854?"
    assert restates(asked, "Calculate the highest common divisor of 4862 and 7854.", examples)
    # Live 2026-10-16: an analysis of the shown examples, answered as if it were the request.
    assert not restates(asked, 'To determine the form, the examples: 1. "Calculate the highest common divisor of '
                               '3672 and 5033394."', examples)
    # One of the request's numbers cut short.
    assert not restates(asked, "Calculate the highest common divisor of 4862 and 785.", examples)
    # A number may change its notation.
    assert restates("what's 15% of 340?", "Calculate 0.15*340.", [{"problem": "Calculate 0.2*55."}])
    assert not restates(asked, "Calculate the highest common divisor of a and b.", examples)


class _Book:
    meanings = {"dm_numbers__gcd": object()}
    procedures = {"dm_numbers__gcd": ["def solve(p): ..."]}
    examples = {"dm_numbers__gcd": [{"problem": "Calculate the highest common divisor of 3672 and 5033394.",
                                     "answer": "918"}]}

    def __init__(self) -> None:
        self.answered: list[str] = []

    def route_by_meaning(self, _vector):
        return "dm_numbers__gcd", 0.79, ""

    def answer_restated(self, kind, restated):
        self.answered.append(restated)
        return {"family": kind, "answer": "374", "restated": restated}


def _route(monkeypatch, written: str):
    import asyncio

    from core.brain import reasoning_amplifier_v2
    from core.learning import procedures_from_solved_examples
    from core.phases import kept_procedures_in_a_turn

    book = _Book()
    asked: list[dict] = []

    class Router:
        async def think(self, **kwargs):
            asked.append(kwargs)
            return {"content": written}

    monkeypatch.setattr(procedures_from_solved_examples, "kept_procedure_book", lambda: book)
    monkeypatch.setattr(kept_procedures_in_a_turn, "her_query_vector", lambda _text, _task: [1.0])
    monkeypatch.setattr(reasoning_amplifier_v2, "amplified_from_kept_receipt", lambda receipt: receipt)
    out = asyncio.run(kept_procedures_in_a_turn.kept_answer_by_meaning(
        "What's the greatest common divisor of 4862 and 7854?", router=Router(), origin="user", tier="primary"))
    return out, book, asked


def test_a_restatement_brings_no_string_the_request_did_not_give() -> None:
    """L09's second attempt, 19 October 2026: requests for code restated as a function called on an input the
    restatement made up, and the value served as if it answered the request (22 of 25 misroutes)."""
    from core.learning.kinds_by_meaning import restates

    code = 'assert f({}) == ??'
    # Made up while restating: not given, so not the problem asked.
    assert not restates("in python, how to check if email adress is valid",
                        "def f(email): return '@' in email\n" + code.format('"user@example.com"'))
    assert not restates("hi can you write me a hello world script in python?",
                        'def f(): return "hello world"\n' + code.format(""))  # given in words, not as a value
    assert not restates("help me write a python code express emotion",
                        "def f(e): return ':)' if e == 'happy' else ':('\n" + code.format("'happy'"))
    # Given in the request, in quotes or in its code: the problem asked.
    assert restates('Which code prints every character of str = "Python" on its own line?',
                    "def f(text): return '\\n'.join(text)\n" + code.format('"Python"'))
    assert restates('What would happen if I ran: "a = [None] a[0] = a print(a)"',
                    "def f(a): a[0] = a; return a\n" + code.format("[None]"))


def test_her_restatement_opens_inside_the_problem_label_and_is_answered(monkeypatch) -> None:
    out, book, asked = _route(monkeypatch, "Calculate the highest common divisor of 4862 and 7854.\n\nProblem:\nmore")
    assert asked[0]["messages"][-1] == {"role": "assistant", "content": "Problem:\n"}
    assert asked[0]["stop_sequences"] == ["\n\nProblem:"]
    assert book.answered == ["Calculate the highest common divisor of 4862 and 7854."]
    assert out["answer"] == "374"


def test_an_account_of_the_examples_is_not_answered_as_the_request(monkeypatch) -> None:
    out, book, _asked = _route(monkeypatch, 'To determine the form, look at "Calculate the highest common divisor '
                                            'of 3672 and 5033394."')
    assert out is None
    assert book.answered == []


def test_what_every_problem_opens_with_is_found_whole() -> None:
    from core.learning.kinds_by_meaning import shared_opening

    worked = "Sort a list of words.\n\nQ: Sort: List: b a\nA: a b\n\n"
    posed = [worked + "Q: Sort the following words alphabetically: List: oven costume",
             worked + "Q: Sort the following words alphabetically: List: shelve tunis"]
    assert shared_opening(posed) == worked
    # A kind posed as its bare question opens with nothing in common.
    assert shared_opening(["Calculate the highest common divisor of 4 and 6.",
                           "Calculate the highest common factor of 9 and 12."]) == ""
    assert shared_opening(["one problem alone"]) == ""


def test_what_every_problem_closes_with_is_found_whole() -> None:
    from core.learning.kinds_by_meaning import shared_ending

    ending = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    posed = ["What are the prime factors of 21000063?" + ending, "List the prime factors of 140?" + ending]
    # The question marks the two share are the questions' own, so the cut is at the paragraph break.
    assert shared_ending(posed) == ending
    assert shared_ending(["Calculate the highest common divisor of 4 and 6.",
                          "Calculate the highest common factor of 9 and 6."]) == ""
    assert shared_ending(["one problem alone" + ending]) == ""


@needs_boundary
def test_a_kinds_posing_is_kept_and_put_back_before_its_procedures_read() -> None:
    reads = "def solve(problem):\n    return 'posed' if problem.startswith('Worked.') else 'bare'\n"
    signatures = {"sorting": Signature({"sorting": 1.0}, 0.99)}
    book = ProcedureBook(signatures, {"sorting": [reads]}, {"sorting": 10}, posing={"sorting": "Worked.\n\n"})
    again = ProcedureBook.from_json(book.to_json())
    assert again.posing == {"sorting": "Worked.\n\n"}
    assert again.answer_restated("sorting", "Q: Sort: List: b a")["answer"] == "posed"
    assert ProcedureBook.combined([again]).posing == again.posing
    assert ProcedureBook.merged([again]).posing == again.posing
