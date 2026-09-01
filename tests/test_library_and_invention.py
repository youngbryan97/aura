"""Compress the corpus, learn what to try, and extend the language itself.

Cards 013, 030, 043, 057, 089, 090, 092, 095-100, 102, 103, 106, 165, 187,
188, 190, 193, 215, 216, A10.*, A11.*, A12.14, A12.15.
"""
from __future__ import annotations

import itertools

import pytest

from core.cognition.autodoc import AutoDoc, describe
from core.cognition.library_compression import (
    REFERENCE_COST,
    LibraryCompressor,
    size,
    subexpressions,
)
from core.cognition.operator_invention import (
    STEP_BUDGET,
    Candidate,
    OperatorKernel,
    Rejection,
    _Budget,
)
from core.cognition.wake_sleep import Task, WakeSleep
from core.knowledge.atomspace import AtomSpace
from core.knowledge.spaces import (
    Alternatives,
    AtomSpaceAdapter,
    DictSpace,
    Package,
    PackageRegistry,
    Space,
)


# ── compression ───────────────────────────────────────────────────────────

def _corpus():
    compressor = LibraryCompressor()
    for i in range(6):
        compressor.add_solution(f"a{i}", ("seq", ("map", ("double", "x")), ("sum", f"v{i}")), family="arith")
    for i in range(4):
        compressor.add_solution(f"b{i}", ("pipe", ("map", ("double", "x")), ("render", f"w{i}")), family="draw")
    return compressor


def test_compression_shrinks_the_corpus_by_exactly_what_it_claims():
    compressor = _corpus()
    before = compressor.corpus_size()
    rounds = compressor.compress()
    saved = sum(r.saved for r in rounds)
    assert compressor.corpus_size() == before - saved
    assert saved > 0


def test_a_pattern_no_bigger_than_a_reference_never_pays():
    compressor = LibraryCompressor()
    for i in range(20):
        compressor.add_solution(f"s{i}", ("outer", ("f",)), family="one")
    compressor.compress()
    assert all(size(a.body) > REFERENCE_COST for a in compressor.library()), (
        "a pattern the size of a reference cannot shorten anything, however often it recurs"
    )


def test_the_reference_cost_is_what_stops_the_compressor_renaming_itself():
    """A one-symbol call site costs two, and pretending otherwise loops forever."""
    assert REFERENCE_COST == 2
    assert size(("f1",)) == REFERENCE_COST
    assert subexpressions(("f1",), min_size=REFERENCE_COST + 1) == []

    compressor = LibraryCompressor()
    for i in range(10):
        compressor.add_solution(f"s{i}", ("seq", ("map", ("double", "x")), f"v{i}"))
    rounds = compressor.compress()
    assert len(rounds) <= 4, "compression must reach a fixed point, not rename its own output"
    assert all(size(a.body) > REFERENCE_COST for a in compressor.library())


def test_a_pattern_used_once_is_rejected_without_a_special_case():
    compressor = LibraryCompressor()
    compressor.add_solution("only", ("deep", ("nested", ("thing", "x"))), family="one")
    compressor.compress()
    assert compressor.library() == []


def test_an_abstraction_recurring_across_families_is_marked():
    compressor = _corpus()
    compressor.compress()
    assert compressor.report()["cross_domain_abstractions"] == ["f1"]


def test_a_rewrite_that_changes_what_a_solution_computes_is_discarded():
    def evaluate(expression):
        # A toy evaluator that treats a reference as a different value, so
        # every rewrite changes the result.
        return size(expression)

    compressor = LibraryCompressor(evaluate=evaluate)
    for i in range(6):
        compressor.add_solution(f"a{i}", ("seq", ("map", ("double", "x")), ("sum", f"v{i}")))
    rounds = compressor.compress()
    assert compressor.library() == []
    assert any(r.rejected_for_meaning for r in rounds)


def test_later_rounds_find_structure_the_earlier_ones_exposed():
    compressor = LibraryCompressor()
    shared = ("map", ("double", "x"))
    for i in range(8):
        compressor.add_solution(f"a{i}", ("wrap", ("pair", shared, ("tag", "t")), f"v{i}"), family="f")
    for i in range(8):
        compressor.add_solution(f"b{i}", ("other", ("pair", shared, ("tag", "t")), f"w{i}"), family="g")
    compressor.compress()
    assert compressor.report()["layers"] >= 1
    assert compressor.corpus_size() < 8 * size(("wrap", ("pair", shared, ("tag", "t")), "v0")) * 2


# ── wake-sleep ────────────────────────────────────────────────────────────

def _wake_sleep():
    compressor = LibraryCompressor()
    cycle = WakeSleep(compressor, seed=1)
    tasks = []
    for i in range(8):
        tasks.append(Task(f"g{i}", frozenset({"grid", "cell"}), family="grid",
                          solution=("seq", ("scan", ("row", "y")), ("mark", f"v{i}"))))
        tasks.append(Task(f"s{i}", frozenset({"sequence", "next"}), family="seq",
                          solution=("seq", ("fold", ("add", "n")), ("emit", f"w{i}"))))
    return compressor, cycle, tasks


def test_the_cycle_closes_replay_and_dreams_together():
    _, cycle, tasks = _wake_sleep()
    result = cycle.cycle(tasks, dreams=20)
    assert result["training"]["replay"] == 16
    assert result["training"]["dreams"] == 20
    assert not result["training"]["circular"]


def test_dreams_alone_are_flagged_as_circular():
    compressor = LibraryCompressor()
    cycle = WakeSleep(compressor, seed=2)
    compressor.add_solution("x", ("seq", ("scan", ("row", "y")), ("a", "1")))
    compressor.add_solution("y", ("seq", ("scan", ("row", "y")), ("b", "2")))
    compressor.compress()
    assert cycle.train_recogniser(cycle.dream(count=5))["circular"]


def test_the_learned_recogniser_beats_the_marginal_prior_it_replaces():
    compressor, cycle, tasks = _wake_sleep()
    cycle.cycle(tasks, dreams=20)
    library = [a.name for a in compressor.library()]
    held_out = [
        Task(f"h{i}", frozenset({"grid", "cell"}), used=frozenset({library[0]}))
        for i in range(6)
    ] + [
        Task(f"k{i}", frozenset({"sequence", "next"}), used=frozenset({library[1]}))
        for i in range(6)
    ]
    report = cycle.expansion_report(held_out, budget=3)
    assert report["earns_its_place"]
    assert report["learned_expansions"] < report["marginal_expansions"]
    assert report["no_solve_regression"]


def test_an_empty_library_gives_no_expansion_claim():
    compressor = LibraryCompressor()
    cycle = WakeSleep(compressor)
    assert not cycle.expansion_report([])["measurable"]


# ── naming ────────────────────────────────────────────────────────────────

def _named():
    compressor = LibraryCompressor()
    for i in range(6):
        compressor.add_solution(f"a{i}", ("seq", ("scan", ("row", "y")), ("mark", f"v{i}")), family="grid")
    for i in range(6):
        compressor.add_solution(f"b{i}", ("seq", ("fold", ("add", "n")), ("emit", f"w{i}")), family="seq")
    compressor.compress()
    doc = AutoDoc()
    doc.propose(compressor.library())
    return compressor, doc


def test_a_description_is_built_from_the_body_not_from_an_invented_purpose():
    compressor, _ = _named()
    described = describe(compressor.library()[0])
    assert "scan" in described and "over" in described


def test_names_are_kept_only_when_the_trial_says_they_help():
    compressor, doc = _named()
    library = compressor.library()

    def retrieve(query, labels):
        hits = [label for label in labels if any(word in label for word in query.split())]
        return hits[0] if hits else (labels[0] if labels else None)

    trial = doc.trial(
        [("scan row grid", library[0].name), ("fold add seq", library[1].name)],
        library, retrieve,
    )
    assert trial.helps
    assert doc.report()["adopted"] == 2


def test_names_that_do_not_help_are_discarded_and_the_symbol_stays():
    compressor, doc = _named()
    library = compressor.library()
    trial = doc.trial(
        [("anything", library[0].name)], library, lambda q, labels: labels[0]
    )
    assert not trial.helps
    assert doc.report()["adopted"] == 0
    assert doc.label(library[1].name) == library[1].name


# ── spaces ────────────────────────────────────────────────────────────────

def test_one_generic_program_runs_over_three_heterogeneous_spaces():
    def generic(space):
        space.add("kettle", 0.9, source="obs:1")
        space.add("rain", 0.4, source="obs:2")
        hits = space.query(lambda k, v: (v if isinstance(v, float) else 0.0) > 0.5)
        return len(hits), sorted(space.provenance("kettle"))

    spaces = [DictSpace("memory"), DictSpace("world"), AtomSpaceAdapter(AtomSpace())]
    results = [generic(space) for space in spaces]
    assert all(result == (1, ["obs:1"]) for result in results)
    assert all(isinstance(space, Space) for space in spaces)


def test_alternatives_travel_and_the_collapse_is_a_decision():
    alternatives = Alternatives(((("a",), 0.6), (("b",), 0.4)))
    assert len(alternatives) == 2
    assert alternatives.best() == ("a",)
    assert not alternatives.ambiguous


def test_two_equally_weighted_answers_are_reported_as_ambiguous():
    assert Alternatives(((("a",), 0.5), (("b",), 0.5))).ambiguous


def test_a_package_unloads_exactly_what_it_loaded():
    space = DictSpace("rules")
    space.add("pre_existing", 1.0)
    registry = PackageRegistry()
    registry.load(Package("temporal", "1.0", {"r1": 0.9, "r2": 0.8}), space)
    assert len(list(space.iterate())) == 3
    result = registry.unload("temporal", "1.0", space)
    assert result["clean"] and result["removed"] == ["r1", "r2"]
    assert [k for k, _ in space.iterate()] == ["pre_existing"]


def test_unloading_reports_what_was_already_gone():
    space = DictSpace("rules")
    registry = PackageRegistry()
    registry.load(Package("p", "1.0", {"r1": 1.0, "r2": 1.0}), space)
    space.remove("r1")
    result = registry.unload("p", "1.0", space)
    assert result["already_gone"] == ["r1"] and not result["clean"]


def test_loading_a_package_twice_is_refused():
    space = DictSpace("rules")
    registry = PackageRegistry()
    package = Package("p", "1.0", {"r1": 1.0})
    registry.load(package, space)
    with pytest.raises(ValueError, match="already loaded"):
        registry.load(package, space)


# ── operator invention ────────────────────────────────────────────────────

def _succ(x, budget):
    budget.step()
    return x + 1


def _double(x, budget):
    budget.step()
    return x * 2


def _triangular(x, budget):
    total = 0
    for i in range(1, x + 1):
        budget.step()
        total += i
    return total


def _forever(x, budget):
    """A candidate that never terminates. The budget is what stops it."""
    for _ in itertools.count():
        budget.step()


def _kernel(*, persistent=True):
    kernel = OperatorKernel({"succ": _succ, "double": _double})
    if persistent:
        for _ in range(3):
            kernel.attempt("triangular numbers", solved=False, probes=(1, 2, 3, 4, 5))
    return kernel


def _solves(fn, _family):
    return fn(5, _Budget(STEP_BUDGET)) == 15


def _consider(kernel, candidate, **kw):
    defaults = dict(family="triangular numbers", probes=(1, 2, 3, 4, 5), solves=_solves,
                    compression=12, adversarial=(0, 100))
    return kernel.consider(candidate, **{**defaults, **kw})


def test_a_transient_failure_does_not_justify_a_new_operator():
    kernel = _kernel(persistent=False)
    kernel.attempt("triangular numbers", solved=False)
    verdict = _consider(kernel, Candidate("tri", "sum", _triangular, ("succ",)))
    assert verdict.rejection is Rejection.NOT_PERSISTENT


def test_a_non_terminating_candidate_is_discarded():
    verdict = _consider(_kernel(), Candidate("forever", "loop", _forever, ("succ",)))
    assert verdict.rejection is Rejection.NON_TERMINATING


def test_a_renamed_composition_is_not_an_invention():
    verdict = _consider(_kernel(), Candidate("rename", "x+1", _succ, ("succ",)))
    assert verdict.rejection is Rejection.NOT_NOVEL


def test_novel_and_reaching_but_compressing_nothing_is_a_special_case():
    verdict = _consider(_kernel(), Candidate("tri", "sum", _triangular, ("succ",)), compression=0)
    assert verdict.rejection is Rejection.NO_COMPRESSION


def test_an_operator_that_breaks_outside_its_synthesis_range_is_refused():
    def brittle(x, budget):
        budget.step()
        if x > 10:
            raise ValueError("out of range")
        return _triangular(x, budget)

    verdict = _consider(_kernel(), Candidate("brittle", "sum", brittle, ("succ",)))
    assert verdict.rejection is Rejection.FAILED_ADVERSARIAL


def test_a_candidate_that_clears_every_bar_is_installed():
    kernel = _kernel()
    verdict = _consider(kernel, Candidate("tri", "sum 1..x", _triangular, ("succ",)))
    assert verdict.installed
    assert verdict.reach_gained == ("triangular numbers",)
    assert "tri" in kernel.operators()


def test_an_invented_operator_becomes_material_for_the_next_one():
    kernel = _kernel()
    _consider(kernel, Candidate("tri", "sum 1..x", _triangular, ("succ",)))

    def tri_double(x, budget):
        return _triangular(x, budget) * 2

    for _ in range(3):
        kernel.attempt("double triangular", solved=False)
    candidate = Candidate("tri2", "double(tri(x))", tri_double, ("tri", "double"))
    assert kernel.compose_from_invented(candidate)
    verdict = kernel.consider(
        candidate, family="double triangular", probes=(2, 3, 4),
        solves=lambda fn, _f: fn(3, _Budget(STEP_BUDGET)) == 12, compression=6,
    )
    assert verdict.installed
    report = kernel.report()
    assert report["max_generation"] == 2 and report["recursive"]


def test_rollback_restores_the_exact_prior_semantics():
    kernel = _kernel()
    before = {
        "succ": [_succ(p, _Budget(STEP_BUDGET)) for p in (1, 2, 3)],
        "double": [_double(p, _Budget(STEP_BUDGET)) for p in (1, 2, 3)],
    }
    _consider(kernel, Candidate("tri", "sum 1..x", _triangular, ("succ",)))
    assert "tri" in kernel.operators()
    result = kernel.rollback("tri")
    assert result["removed"] == ["tri"]
    assert kernel.behaviourally_identical((1, 2, 3), before)["identical"]


def test_rolling_back_something_never_installed_raises():
    with pytest.raises(KeyError):
        _kernel().rollback("never")


def test_the_rollback_check_is_behavioural_not_structural():
    kernel = _kernel()
    _consider(kernel, Candidate("tri", "sum 1..x", _triangular, ("succ",)))
    kernel.rollback("tri")
    wrong = {"succ": [99, 99, 99]}
    assert not kernel.behaviourally_identical((1, 2, 3), wrong)["identical"]
