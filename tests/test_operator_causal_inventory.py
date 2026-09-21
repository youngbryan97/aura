"""Retained operators participate in isolated component interventions."""

import pytest

from core.cognition import an_operator_she_invents as invention
from core.cognition.operator_invention import Candidate, OperatorKernel
from core.cognition.the_floor_she_stands_on import PLUS, L, N, V, build
from core.cognition.what_she_can_take_back import as_it_stands
from core.cognition.what_she_is_made_of import what_a_part_is_worth, what_she_is_made_of
from core.cognition.why_it_is_not_better import ACause, why_it_is_not_better


@pytest.fixture(autouse=True)
def separate_state(monkeypatch):
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.one_algebra import DERIVED_HEADS

    monkeypatch.setattr(invention, "_KERNEL", OperatorKernel())
    before = as_it_stands()
    RULES_WITH_NO_SHAPE.clear()
    DERIVED_HEADS.clear()
    yield
    assert before.restore() == ()


def install(name, offset=7, dependencies=()):
    kernel = invention.the_kernel()
    for _ in range(3):
        kernel.attempt(name, solved=False)
    candidate = Candidate(name, "an offset", term=build(L("x", PLUS(V("x"), N(offset)))),
                          built_from=dependencies)
    result = kernel.consider(candidate, family=name, probes=(-3, 0, 4), compression=8,
                             solves=lambda fn, _: all(fn(x) == x + offset for x in (-3, 0, 4)))
    assert result.installed
    return candidate.term


def test_inventory_includes_invented_terms_and_declared_dependents():
    term = install("first")
    install("child", 9, ("first",))
    parts = {part.at: part for part in what_she_is_made_of()}
    part = parts["invented operator/first"]
    assert part.term == term
    assert "invented operator/child" in part.holds_up


def test_withdrawal_removes_dependents_not_unrelated_later_inventions():
    kernel = invention.the_kernel()
    install("first")
    install("independent", 11)
    install("child", 9, ("first",))
    install("grandchild", 13, ("child",))
    assert kernel.withdraw("first") == ("first", "child", "grandchild")
    assert tuple(kernel.operators()) == ("independent",)
    restored = OperatorKernel()
    assert restored.recall_operators(kernel.written_operators()) == 1
    assert restored.operators()["independent"].fn(10) == 21
    kernel.rollback("independent")
    assert not kernel.operators()  # Its old snapshot cannot resurrect the withdrawal.


def test_withdrawal_cannot_delete_a_base_operator():
    kernel = OperatorKernel({"base": lambda x, budget=None: x})
    before = kernel.snapshot()
    assert kernel.withdraw("base") == ()
    assert kernel.withdraw("missing") == ()
    assert kernel.snapshot() == before


@pytest.mark.parametrize("dependencies", [("missing",), ("first", "first"), ("candidate",)])
def test_invalid_lineage_cannot_be_admitted(dependencies):
    install("first")
    kernel = invention.the_kernel()
    for _ in range(3):
        kernel.attempt("candidate", solved=False)
    before = kernel.snapshot()
    candidate = Candidate("candidate", "bad lineage", built_from=dependencies,
                          term=build(L("x", PLUS(V("x"), N(14)))))
    result = kernel.consider(candidate, family="candidate", probes=(1, 2),
                             compression=4, solves=lambda *_: True)
    assert not result.installed and result.rejection.value == "invalid_lineage"
    assert kernel.snapshot() == before


def test_real_search_lesion_is_recovered_and_changes_cost_on_new_inputs():
    install("offset")
    part = next(part for part in what_she_is_made_of() if part.at == "invented operator/offset")
    before = as_it_stands()
    cases = tuple((x, x + 7) for x in (-80, 17, 123456))

    def cost(examples):
        for candidate in invention._a_candidate_for(
            "held-out inputs", [x for x, _ in examples], deepest=3, how_many=400, max_offered=None,
        ):
            fn = candidate.how_it_computes()
            try:
                if all(fn(x) == y for x, y in examples):
                    return invention.how_far_the_last_search_reached()["reach"]["walked"]
            except (TypeError, ValueError, ArithmeticError, TimeoutError):
                pass
        return 400

    measured = what_a_part_is_worth(part, [("new inputs", cases)], costs=cost)
    assert measured > 0
    assert as_it_stands() == before
    assert invention.the_kernel().operators()["offset"].fn(500) == 507


def test_component_measurement_restores_baseline_mutations_before_the_lesion():
    from core.cognition.an_invented_kind import KINDS
    from core.cognition.one_thing_many_spellings import _SIZES_ASKED_ABOUT

    install("offset")
    part = next(part for part in what_she_is_made_of() if part.at == "invented operator/offset")
    seen = []
    before = as_it_stands()

    def cost(_):
        seen.append(("mutation" in KINDS, 999 in _SIZES_ASKED_ABOUT))
        KINDS["mutation"] = object()
        _SIZES_ASKED_ABOUT.add(999)
        return 1 if "offset" in invention.the_kernel().operators() else 9

    assert what_a_part_is_worth(part, [("case", ())], costs=cost) == 8
    assert seen == [(False, False), (False, False)]
    assert as_it_stands() == before


def test_baseline_exception_does_not_leave_a_trial_change():
    from core.cognition.an_invented_kind import KINDS

    install("offset")
    part = next(part for part in what_she_is_made_of() if part.at == "invented operator/offset")
    before = as_it_stands()

    def cost(_):
        KINDS["before raising"] = object()
        raise ValueError("baseline failed")

    with pytest.raises(ValueError, match="baseline failed"):
        what_a_part_is_worth(part, [("case", ())], costs=cost)
    assert as_it_stands() == before


@pytest.mark.parametrize("raises", [False, True])
def test_partial_failed_cause_does_not_contaminate_the_next_one(raises):
    from core.cognition.an_invented_kind import KINDS

    before = as_it_stands()
    seen, undo = [], []

    def partial():
        KINDS["partial"] = object()
        if raises:
            raise ValueError("partial failed")
        return False

    def next_one():
        seen.append("partial" in KINDS)
        KINDS["other"] = object()
        return True

    causes = [ACause("word/partial", "partial", partial, lambda: undo.append(True)),
              ACause("word/other", "other", next_one)]
    found = why_it_is_not_better([("case", ())], costs=lambda _: 1, among=causes)
    assert [row["at"] for row in found] == ["word/other"]
    assert seen == [False] and undo == [True]
    assert as_it_stands() == before


def test_a_failed_custom_undo_cannot_be_reported_as_a_clean_comparison():
    from core.cognition.an_invented_kind import KINDS

    before = as_it_stands()
    def change():
        KINDS["partial"] = object()
        return True
    def undo():
        raise RuntimeError("custom external state was not restored")
    with pytest.raises(RuntimeError, match="not restored"):
        why_it_is_not_better([("case", ())], costs=lambda _: 1,
                            among=[ACause("word/partial", "partial", change, undo)])
    assert as_it_stands() == before
