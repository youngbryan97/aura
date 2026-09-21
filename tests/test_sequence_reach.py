"""Development and answers consume the same retained executable meanings."""

from types import SimpleNamespace

import pytest

from core.cognition import an_operator_she_invents as invention
from core.cognition import what_she_does_about_herself as development
from core.cognition.operator_invention import Candidate, OperatorKernel
from core.cognition.sequence_reach import (
    SequenceMeaning,
    SequenceReach,
    measure_sequence_reach,
    reach_utility,
    retained_sequence_reach,
)
from core.cognition.the_floor_she_stands_on import FST, PAIR, PLUS, SND, L, N, V, build
from core.cognition.what_she_can_take_back import as_it_stands


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.an_invented_kind import KINDS, UNSETTLED

    monkeypatch.setattr(invention, "_KERNEL", OperatorKernel())
    held = as_it_stands()
    KINDS.clear()
    UNSETTLED.clear()
    RULES_WITH_NO_SHAPE.clear()
    yield
    assert held.restore() == ()


def install_offset():
    kernel = invention.the_kernel()
    pairs = tuple(((x, x + 2), (x + 7, x + 2)) for x in (1, 10, 30, 50))
    for _ in range(3):
        kernel.attempt("offset-first", solved=False, cases=pairs)
    term = build(L("xs", PAIR(PLUS(FST(V("xs")), N(7)), SND(V("xs")))))
    verdict = kernel.consider(
        Candidate("offset-first", "offset the first cell", term=term),
        family="offset-first", probes=[x for x, _ in pairs], compression=8,
        solves=lambda fn, _: all(fn(x) == y for x, y in pairs),
    )
    assert verdict.installed
    return pairs


def test_retained_operator_is_visible_to_development_and_answering():
    from core.cognition.primitive_invention import Transition
    from core.cognition.sequence_induction import SequenceQuestion, _work_the_meaning_out

    pairs = install_offset()
    measured = measure_sequence_reach(pairs)
    assert measured.solved and measured.walked == 1
    assert measured.agreed_on((100, 102)) == (107, 102)
    assert development._how_it_stands([("new-inputs", pairs)]) == {"new-inputs": (1, True)}
    question = SequenceQuestion(tuple(Transition(before=x, after=y) for x, y in pairs), (100, 102))
    assert _work_the_meaning_out(question).startswith("[107, 102]")
    invention.the_kernel().withdraw("offset-first")
    assert not measure_sequence_reach(pairs).solved


def test_public_sequence_path_can_use_a_retained_operator(monkeypatch):
    from core.cognition import sequence_induction as sequence

    install_offset()
    monkeypatch.setattr(sequence, "_she_may_improve_a_working_answer", lambda *args: None)
    answer = sequence.answer_sequence_question(
        "[1,3] becomes [8,3]; [10,12] becomes [17,12]; "
        "[30,32] becomes [37,32]; [50,52] becomes [57,52]; what does [100,102] become?"
    )
    assert answer.startswith("[107, 102]")


def test_shared_measurement_does_not_mutate_the_library():
    before = as_it_stands()
    development._how_it_stands([("reverse", [((1, 2), (2, 1))] * 4)])
    assert as_it_stands() == before


def test_correctness_canary_is_registered_and_passes():
    from core.cognition.sequence_reach import reach_preserves_correctness

    assert reach_preserves_correctness() == []


def test_two_rules_cannot_collectively_fake_one_matching_rule():
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE

    RULES_WITH_NO_SHAPE["first"] = SimpleNamespace(read=lambda xs: (1, 1))
    RULES_WITH_NO_SHAPE["second"] = SimpleNamespace(read=lambda xs: (2, 2))
    found = retained_sequence_reach([((3, 4), (1, 1)), ((5, 6), (2, 2))])
    assert not found.solved and found.walked == 2


def test_disagreement_or_execution_failure_does_not_choose_the_first_answer():
    found = SequenceReach((SequenceMeaning("one", lambda _: (1, 2)),
                           SequenceMeaning("two", lambda _: (2, 1))), 2)
    assert found.agreed_on((3, 4)) is None
    failed = SequenceMeaning("failed", lambda _: 1 / 0)
    assert SequenceReach((failed,), 1).agreed_on((3, 4)) is None


def test_empty_evidence_is_not_a_vacuous_match():
    install_offset()
    assert not retained_sequence_reach(()).solved
    assert measure_sequence_reach(()).walked == 0


def test_cost_savings_cannot_compensate_for_losing_one_solved_family():
    for size in (1, 2, 10, 1000):
        slow_correct = [(10**12, True)] * size
        cheap_loss = [(0, True)] * (size - 1) + [(0, False)]
        assert reach_utility(slow_correct) > reach_utility(cheap_loss)
    assert reach_utility([(0, False)]) == reach_utility([(1000, False)])


def test_opening_one_family_does_not_hide_losing_another(monkeypatch):
    monkeypatch.setattr(development, "_how_it_stands", lambda _: {"old": (1, False), "new": (1, True)})
    kept, why = development.worth_keeping(
        {"old": (100, True), "new": (100, False)}, [("old", ()), ("new", ())],
    )
    assert not kept and "old" in why


def test_cheaper_failed_searches_are_not_positive_evidence(monkeypatch):
    monkeypatch.setattr(development, "_how_it_stands", lambda _: {"unknown": (1, False)})
    assert not development.worth_keeping({"unknown": (100, False)}, [("unknown", ())])[0]
    receipt = development._what_it_measured("tried", {"unknown": (100, False)},
                                           [("unknown", ())], "compared")
    assert not receipt.paid


def test_new_reach_is_not_erased_by_a_more_expensive_success(monkeypatch):
    monkeypatch.setattr(development, "_how_it_stands", lambda _: {"new": (1000, True)})
    before, probe = {"new": (1, False)}, [("new", ())]
    assert development.worth_keeping(before, probe)[0]
    assert development._what_it_measured("opened", before, probe, "compared").paid
