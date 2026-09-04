"""Contract tests for the Hyperon and OpenWorm derived disciplines.

core/knowledge/metta.py, core/organism/model_validation.py.
"""

from __future__ import annotations

import time

import pytest

from core.knowledge import metta as metta_mod
from core.knowledge.atomspace import AtomSpace, Link, Node, TruthValue
from core.knowledge.metta import (
    NUMBER,
    GroundedOp,
    MeTTaEngine,
    equality,
    expr,
    var,
)
from core.organism import model_validation as val_mod
from core.organism.model_validation import (
    Claim,
    Observation,
    Outcome,
    RuntimeModel,
    ValidationSuite,
    ValidationTest,
    boolean_score,
    ratio_score,
    threshold_score,
)


@pytest.fixture(autouse=True)
def _clean():
    metta_mod.reset_metta_for_test()
    val_mod.reset_validation_for_test()
    yield
    metta_mod.reset_metta_for_test()
    val_mod.reset_validation_for_test()


def _engine() -> MeTTaEngine:
    return MeTTaEngine(AtomSpace())


def _num(value: str) -> Node:
    return Node(NUMBER, value)


# ── MeTTa: rules as data ──────────────────────────────────────────────

def test_a_rule_rewrites_an_expression():
    engine = _engine()
    engine.add_rule(expr("grandparent", var("x"), var("z")), expr("ancestor", var("x"), var("z")))
    results = engine.evaluate(expr("grandparent", "alice", "carol"))
    assert [str(r) for r in results] == ['(ancestor (Concept "alice") (Concept "carol"))']


def test_rules_can_be_added_queried_and_retracted_at_runtime():
    engine = _engine()
    lhs, rhs = expr("a"), expr("b")
    engine.add_rule(lhs, rhs, source="learned")

    assert len(engine.rules()) == 1
    assert "learned" in engine.report()["rule_sources"]
    assert [str(r) for r in engine.evaluate(expr("a"))] == ["(b )"]

    assert engine.retract_rule(lhs, rhs) is True
    assert engine.rules() == []
    # With the rule gone, the expression is its own normal form.
    assert [str(r) for r in engine.evaluate(expr("a"))] == ["(a )"]
    assert engine.retract_rule(lhs, rhs) is False


def test_a_rule_can_carry_a_truth_value():
    engine = _engine()
    tv = TruthValue(strength=0.8, count=10.0)
    rule = engine.add_rule(expr("p"), expr("q"), tv=tv)
    assert engine.rule_truth(rule).strength == pytest.approx(0.8)


def test_rules_with_variables_live_beside_the_fact_space_not_in_it():
    """The AtomSpace refuses pattern atoms; the rule space is why that is fine."""
    space = AtomSpace()
    engine = MeTTaEngine(space)
    engine.add_rule(expr("f", var("x")), expr("g", var("x")))
    assert len(engine.rules()) == 1
    # Nothing containing a Variable leaked into the fact space.
    assert space.atoms_of_type(metta_mod.EQUALITY) == []


def test_a_ground_equality_is_also_an_ordinary_fact():
    space = AtomSpace()
    engine = MeTTaEngine(space)
    engine.add_rule(expr("pi"), _num("3.14159"))
    assert equality(expr("pi"), _num("3.14159")) in space


# ── MeTTa: evaluation ─────────────────────────────────────────────────

def test_evaluation_is_non_deterministic():
    engine = _engine()
    engine.add_rule(expr("color"), expr("red"))
    engine.add_rule(expr("color"), expr("blue"))
    results = sorted(str(r) for r in engine.evaluate(expr("color")))
    assert results == ["(blue )", "(red )"], "collapsing early throws away alternatives"


def test_grounded_arithmetic_reduces():
    engine = _engine()
    assert str(engine.evaluate(Link("+", (_num("2"), _num("3"))))[0]) == '(Number "5")'


def test_rewriting_reaches_inside_arguments():
    engine = _engine()
    nested = Link("+", (_num("2"), Link("*", (_num("3"), _num("4")))))
    assert str(engine.evaluate(nested)[0]) == '(Number "14")'


def test_rules_and_grounded_ops_compose():
    engine = _engine()
    engine.add_rule(expr("double", var("n")), Link("*", (var("n"), _num("2"))))
    results = [str(r) for r in engine.evaluate(expr("double", _num("21")))]
    assert '(Number "42")' in results


def test_a_grounded_op_that_cannot_apply_yields_no_rewrite():
    engine = _engine()
    # "+" over concepts is not arithmetic; the form is its own normal form.
    result = engine.evaluate(Link("+", (Node("Concept", "a"), Node("Concept", "b"))))
    assert [str(r) for r in result] == ['(+ (Concept "a") (Concept "b"))']


def test_a_cycle_does_not_hang_or_become_a_false_normal_form():
    engine = _engine()
    engine.add_rule(expr("a"), expr("b"))
    engine.add_rule(expr("b"), expr("a"))
    report = engine.reduce(expr("a"), max_steps=50)
    assert report.duration_s < 1.0
    # b reduces only back to a, which is already seen, so b is where it stops.
    assert [str(r.result) for r in report.results] == ["(b )"]


def test_bounds_are_reported_not_silently_applied():
    engine = _engine()
    engine.add_rule(expr("grow", var("n")), expr("grow", expr("s", var("n"))))
    report = engine.reduce(expr("grow", "zero"), max_steps=12)
    assert report.truncated_by == "max_steps"
    assert report.steps <= 12
    assert engine.report()["truncations"] == 1


def test_a_time_budget_bounds_a_runaway_rule_set():
    engine = _engine()
    engine.add_rule(expr("spin", var("n")), expr("spin", expr("s", var("n"))))
    started = time.perf_counter()
    report = engine.reduce(expr("spin", "x"), max_steps=10**6, time_budget_s=0.05)
    assert time.perf_counter() - started < 1.0
    assert report.truncated_by in {"time_budget", "max_results"}


def test_the_trail_explains_how_a_result_was_reached():
    engine = _engine()
    engine.add_rule(expr("a"), expr("b"))
    engine.add_rule(expr("b"), expr("c"))
    report = engine.reduce(expr("a"))
    result = report.results[0]
    assert str(result.result) == "(c )"
    assert len(result.trail) == 2
    assert all("rule:" in step for step in result.trail)


def test_impure_grounded_ops_read_live_state():
    space = AtomSpace()
    engine = MeTTaEngine(space)
    fact = Node("Concept", "resident")
    space.add(fact, TruthValue(strength=0.9, count=5.0))
    assert str(engine.evaluate(Link("truth", (fact,)))[0]) == '(Number "0.9")'
    assert str(engine.evaluate(Link("exists", (fact,)))[0]) == '(Concept "True")'
    assert str(engine.evaluate(Link("exists", (Node("Concept", "absent"),)))[0]) == (
        '(Concept "False")'
    )


def test_purity_is_declared_so_caching_can_be_safe():
    engine = _engine()
    ops = {op["name"]: op for op in engine.report()["grounded_ops"]}
    assert ops["+"]["pure"] is True
    assert ops["truth"]["pure"] is False


def test_a_conflicting_grounded_op_registration_is_refused():
    engine = _engine()
    with pytest.raises(ValueError, match="already registered"):
        engine.register_op(GroundedOp(name="+", arity=2, fn=lambda a, b: None))


def test_a_broken_grounded_op_does_not_break_evaluation():
    engine = _engine()
    engine.register_op(
        GroundedOp(name="boom", arity=1, fn=lambda a: (_ for _ in ()).throw(RuntimeError("x")))
    )
    assert [str(r) for r in engine.evaluate(Link("boom", (_num("1"),)))] == ['(boom (Number "1"))']


def test_runtime_rules_install():
    rules = metta_mod.install_runtime_rules()
    assert rules
    assert metta_mod.metta_report()["rules"] >= len(rules)


# ── validation: the discipline ────────────────────────────────────────

def _obs(value: float = 1.0) -> Observation:
    return Observation(name="o", value=value, source="a real measurement, cited")


def _test(name: str = "t", predict=lambda m: 1.0, capability: str = "cap") -> ValidationTest:
    return ValidationTest(
        name=name,
        description="d",
        required_capability=capability,
        observation=_obs(),
        predict=predict,
        score=lambda p, o: ratio_score(float(p), float(o.value)),
    )


def test_an_observation_without_a_source_is_refused():
    suite = ValidationSuite()
    bad = ValidationTest(
        name="t",
        description="d",
        required_capability="",
        observation=Observation(name="o", value=1.0, source="  "),
        predict=lambda m: 1.0,
        score=lambda p, o: ratio_score(float(p), float(o.value)),
    )
    with pytest.raises(ValueError, match="no source"):
        suite.add_test(bad)


def test_a_claim_without_a_test_cannot_be_registered():
    suite = ValidationSuite()
    with pytest.raises(ValueError, match="not\n?\\s*registered|not registered"):
        suite.add_claim(Claim(statement="we are great", test="nonexistent", owner="t"))


def test_a_missing_capability_is_not_applicable_not_a_failure():
    suite = ValidationSuite()
    suite.add_test(_test(capability="swimming"))
    suite.add_model(RuntimeModel(name="m"))
    outcome = suite.run()
    result = outcome["results"][0]
    assert result["score"]["outcome"] == str(Outcome.NOT_APPLICABLE)
    assert outcome["failed"] == 0
    assert outcome["applicable"] == 0, "a vacuously-passing suite must say so"


def test_a_raising_prediction_is_an_error_not_a_failure():
    suite = ValidationSuite()
    suite.add_test(_test(predict=lambda m: (_ for _ in ()).throw(RuntimeError("probe down"))))
    suite.add_model(RuntimeModel(name="m").declare("cap"))
    outcome = suite.run()
    assert outcome["errored"] == 1
    assert outcome["failed"] == 0
    assert "probe down" in outcome["errors"][0]["score"]["interpretation"]


def test_a_passing_and_a_failing_test_are_distinguished():
    suite = ValidationSuite()
    suite.add_test(_test(name="good", predict=lambda m: 1.0))
    suite.add_test(_test(name="bad", predict=lambda m: 5.0))
    suite.add_model(RuntimeModel(name="m").declare("cap"))
    outcome = suite.run()
    assert outcome["passed"] == 1
    assert outcome["failed"] == 1
    assert outcome["failures"][0]["test"] == "bad"


def test_scores_carry_their_own_interpretation():
    ratio = ratio_score(1.05, 1.0, tolerance=0.1)
    assert ratio.outcome is Outcome.PASS
    assert "tolerance" in ratio.interpretation

    over = ratio_score(2.0, 1.0, tolerance=0.1)
    assert over.outcome is Outcome.FAIL

    boolean = boolean_score(False, expected=True, subject="lockdep clean")
    assert boolean.outcome is Outcome.FAIL
    assert "lockdep clean" in boolean.interpretation

    threshold = threshold_score(3.0, 5.0, units=" splats")
    assert threshold.outcome is Outcome.PASS
    assert "≤" in threshold.interpretation


def test_a_ratio_against_zero_is_an_error_not_a_pass():
    score = ratio_score(1.0, 0.0)
    assert score.outcome is Outcome.ERROR


def test_unsupported_claims_are_the_machine_checked_version_of_the_document():
    suite = ValidationSuite()
    suite.add_test(_test(name="failing", predict=lambda m: 99.0))
    suite.add_claim(
        Claim(
            statement="the thing works",
            test="failing",
            owner="t",
            asserted_in="ARCHITECTURE.md",
        )
    )
    suite.add_model(RuntimeModel(name="m").declare("cap"))

    unsupported = suite.unsupported_claims()
    assert unsupported[0]["reason"] == "never run"

    suite.run()
    unsupported = suite.unsupported_claims()
    assert len(unsupported) == 1
    assert unsupported[0]["statement"] == "the thing works"
    assert unsupported[0]["asserted_in"] == "ARCHITECTURE.md"


def test_a_claim_becomes_supported_when_its_test_passes():
    suite = ValidationSuite()
    predicted = {"value": 99.0}
    suite.add_test(_test(name="t", predict=lambda m: predicted["value"]))
    suite.add_claim(Claim(statement="s", test="t", owner="o"))
    suite.add_model(RuntimeModel(name="m").declare("cap"))

    suite.run()
    assert len(suite.unsupported_claims()) == 1
    predicted["value"] = 1.0
    suite.run()
    assert suite.unsupported_claims() == []


def test_tests_without_claims_are_reported():
    suite = ValidationSuite()
    suite.add_test(_test(name="orphan"))
    assert suite.report()["tests_without_claims"] == ["orphan"]


# ── the live runtime suite ────────────────────────────────────────────

def test_the_runtime_validation_suite_installs_and_every_claim_has_a_test():
    installed = val_mod.install_runtime_validation()
    assert installed["claims"] == len(installed["tests"])
    suite = val_mod.get_suite()
    test_names = {t.name for t in suite.tests()}
    for claim in suite.claims():
        assert claim.test in test_names
        assert claim.asserted_in, f"claim {claim.statement!r} does not say where it is asserted"


def test_the_runtime_validates_its_own_claims_against_live_telemetry():
    from core.fsw.health_checker import install_runtime_pings
    from core.runtime.memory_infra import install_runtime_providers

    install_runtime_providers()
    install_runtime_pings()
    val_mod.install_runtime_validation()

    outcome = val_mod.run_validation()
    assert outcome["applicable"] > 0, "every test being N/A would be a vacuous pass"
    assert outcome["errored"] == 0, outcome["errors"]
    # Failures are legitimate findings, not test bugs — report them clearly.
    assert outcome["failed"] == 0, [f["score"]["interpretation"] for f in outcome["failures"]]


def test_cognition_invariants_registered_and_clean():
    from core.verify import runtime_invariants  # noqa: F401
    from core.verify.invariants import get_registry, verify

    names = {s.name for s in get_registry().specs()}
    assert "claims.every_claim_has_a_passing_test" in names
    assert "metta.rules_terminate" in names

    from core.fsw.health_checker import install_runtime_pings
    from core.runtime.memory_infra import install_runtime_providers

    install_runtime_providers()
    install_runtime_pings()
    val_mod.install_runtime_validation()
    # The boot posture, and for the same reason a boot uses it: the invariant
    # below skips NOT_MEASURED, so re-running five experiments buys it nothing
    # and cost this test 700 seconds. The experiments are run by the tests
    # that own them.
    val_mod.run_validation(include_expensive=False)

    report = verify("cognition", record=False)
    assert report.ok, report.summary()


def test_health_integrity_block_carries_the_self_model():
    from core.runtime.health_contract import _runtime_integrity_block

    val_mod.install_runtime_validation()
    block = _runtime_integrity_block()
    assert "self_model" in block
    assert block["self_model"]["claims"] > 0


def test_a_boot_checks_instruments_and_does_not_re_run_experiments():
    """141 seconds of activate_foundations were three claim predicates.

    They run a full grown-against-reset search — 56.4s, 52.3s and 29.3s,
    measured during a real boot on 2026-09-01. Until that day a refusal
    escaping from the language aborted each one, so the boot reported them as
    errors and never paid for them; handling the refusal correctly made them
    complete for the first time and took the boot from 13.85s to 145s, past
    the runtime's lease deadline.

    A boot checks that instruments work. The experiments are measured by the
    tests that own them, and a skipped one says so rather than reporting a
    pass it did not earn.
    """
    from core.organism.model_validation import (
        _AN_EXPERIMENT_NOT_AN_INSTRUMENT,
        get_suite,
        install_runtime_validation,
        run_validation,
    )

    install_runtime_validation()
    assert _AN_EXPERIMENT_NOT_AN_INSTRUMENT, "no predicate is declared an experiment"

    tests = getattr(get_suite(), "_tests", {})
    for name in _AN_EXPERIMENT_NOT_AN_INSTRUMENT:
        assert name in tests, f"{name} is declared expensive but is not registered"
        assert tests[name].expensive, f"{name} is not flagged on its test"

    import time

    started = time.perf_counter()
    outcome = run_validation(include_expensive=False)
    elapsed = time.perf_counter() - started
    assert outcome is not None
    # The boot posture must stay cheap. The three experiments alone were 138s.
    #
    # Name the predicate rather than only the total. The first time this bar
    # was crossed it said "17.7s" and nothing else, and the two predicates
    # responsible for 12 of those seconds had to be found by hand. A budget
    # that reports a number tells you it is broken; one that reports a name
    # tells you what to flag.
    slow = [
        (name, seconds)
        for name, seconds in _seconds_per_predicate(tests).items()
        if seconds > 1.5 and name not in _AN_EXPERIMENT_NOT_AN_INSTRUMENT
    ]
    assert not slow, (
        "these run an experiment but are not declared one: "
        + ", ".join(f"{n} ({s:.1f}s)" for n, s in sorted(slow, key=lambda r: -r[1]))
    )
    assert elapsed < 10.0, f"the boot posture took {elapsed:.1f}s"


def _seconds_per_predicate(tests: dict) -> dict[str, float]:
    """How long each cheap predicate takes, measured the way the boot pays it."""
    import time

    from core.organism.model_validation import RuntimeModel

    model = RuntimeModel()
    timed: dict[str, float] = {}
    for name, test in list(tests.items()):
        if test.expensive:
            continue
        began = time.perf_counter()
        try:
            test.score(test.predict(model), test.observation)
        except Exception:  # a raise is an ERROR outcome, and still costs time
            pass
        timed[name] = time.perf_counter() - began
    return timed


def test_a_skipped_experiment_says_so_rather_than_passing():
    """not-measured and passed are different readings, and only one is true."""
    from core.organism.model_validation import (
        Observation,
        Outcome,
        Score,
        ValidationTest,
    )

    ran: list[str] = []

    test = ValidationTest(
        name="an_experiment",
        description="runs a search",
        required_capability="",
        observation=Observation(name="violations", value=0, source="x", units=""),
        predict=lambda _m: ran.append("ran") or 0,
        score=lambda p, o: Score(
            kind="threshold", value=1.0, outcome=Outcome.PASS, interpretation="ok"
        ),
        expensive=True,
    )

    class _Model:
        name = "probe"

        def capabilities(self):
            return set()

    skipped = test.run(_Model(), include_expensive=False)
    assert not ran, "the experiment was run under the boot posture"
    assert skipped.score.outcome is Outcome.NOT_MEASURED
    assert "experiment" in skipped.score.interpretation

    full = test.run(_Model(), include_expensive=True)
    assert ran == ["ran"]
    assert full.score.outcome is Outcome.PASS
