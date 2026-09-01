"""Eight learners, one currency: procedures that can lose to each other.

Cards 004, 005, 019, 021, 022, 028, 059, 066, 091, 167, 168, 172, 174, 199,
A5.8, A12.6, A12.7, A12.16, A12.18, and the Voyager library items.
"""
from __future__ import annotations

import pytest

from core.cognition.impasse import Chunk, ImpasseType
from core.cognition.procedure import (
    Backend,
    Effect,
    Origin,
    Precondition,
    ProceduralValue,
    Reversibility,
    Signature,
    compose,
    reset_procedure_registry_for_test,
)
from core.cognition.procedure_adapters import (
    from_chunk,
    from_generalized_rule,
    from_learned_skill,
    from_tool_schema,
    ingest_all,
)
from core.cognition.procedural_generalization import GeneralizedRule, RuleTier


def sig(pre=(), eff=()):
    return Signature(
        tuple(Precondition(key=k) for k in pre), tuple(Effect(key=k) for k in eff)
    )


# ── the value contract ────────────────────────────────────────────────────

def test_the_currency_reproduces_the_chunk_arithmetic_it_generalises():
    registry = reset_procedure_registry_for_test()
    chunk = Chunk(
        signature="s1", resolution="take_a", impasse_type=ImpasseType.TIE,
        cost_saved_per_use=2.0, match_cost=0.05, uses=10, correct=9, incorrect=1,
    )
    procedure = from_chunk(chunk, registry=registry)
    assert procedure.value.net == pytest.approx(chunk.expected_value)


def test_a_generalized_rule_keeps_its_conservative_wilson_reading():
    registry = reset_procedure_registry_for_test()
    rule = GeneralizedRule(
        conditions=frozenset({"a", "b"}), resolution="do_x",
        tier=RuleTier.PROMOTED, supporting=12, contradicting=0, correct=3, incorrect=0,
    )
    procedure = from_generalized_rule(rule, value_when_it_works=1.0, registry=registry)
    assert procedure.value.p_success == pytest.approx(rule.confidence)
    assert procedure.value.p_success < 1.0, "a Wilson bound never reads 1.0 on finite evidence"


def test_a_skill_nobody_timed_lands_at_net_zero():
    registry = reset_procedure_registry_for_test()

    class _Skill:
        name, description, parameters = "tidy", "", ["dir"]
        steps = [object(), object()]
        successes, failures = 4, 1
        reliability = 0.8

    procedure = from_learned_skill(_Skill(), seconds_per_step=0.0, registry=registry)
    assert procedure.value.net == pytest.approx(0.0)
    assert not procedure.value.pays


def test_procedures_from_different_learners_are_ranked_by_the_same_number():
    registry = reset_procedure_registry_for_test()
    weak = registry.register(
        "weak chunk", Backend.CHUNK, sig(["x"], ["done"]),
        origin=Origin(learner="impasse"), value=ProceduralValue(p_success=0.9, value_when_it_works=1.0, match_cost=0.1),
    )
    strong = registry.register(
        "strong macro", Backend.MACRO, sig(["x"], ["done"]),
        value=ProceduralValue(p_success=0.8, value_when_it_works=10.0, match_cost=0.1),
    )
    ranked = registry.match({"x": True})
    assert [p.procedure_id for p in ranked] == [strong.procedure_id, weak.procedure_id]
    assert registry.report()["backends_competing"] == 2


def test_a_procedure_that_stops_paying_is_retired_by_the_same_rule():
    registry = reset_procedure_registry_for_test()
    procedure = registry.register(
        "flaky", Backend.MACRO, sig(["x"], ["done"]), value=ProceduralValue(p_success=0.9, value_when_it_works=1.0, match_cost=0.5)
    )
    for _ in range(10):
        registry.record_use(procedure.procedure_id, success=False)
    retired = registry.prune()
    assert [p.procedure_id for p in retired] == [procedure.procedure_id]
    assert registry.match({"x": True}) == []


def test_one_failure_does_not_retire_a_procedure():
    registry = reset_procedure_registry_for_test()
    procedure = registry.register(
        "new", Backend.MACRO, sig(["x"], ["done"]), value=ProceduralValue(p_success=1.0, value_when_it_works=1.0, match_cost=0.1)
    )
    registry.record_use(procedure.procedure_id, success=False)
    assert registry.prune() == []


def test_measured_success_replaces_assumed_success():
    registry = reset_procedure_registry_for_test()
    procedure = registry.register("p", Backend.MACRO, sig(["x"], ["y"]))
    for success in (True, True, False, True):
        procedure = registry.record_use(procedure.procedure_id, success=success)
    assert procedure.value.p_success == pytest.approx(0.75)
    assert procedure.value.wilson_floor() < 0.75


# ── origin and audit ──────────────────────────────────────────────────────

def test_a_compiled_procedure_that_cannot_say_where_it_came_from_is_refused():
    registry = reset_procedure_registry_for_test()
    with pytest.raises(ValueError, match="cannot be audited"):
        registry.register("mystery", Backend.CHUNK, sig(["x"], ["y"]))


def test_a_procedure_explains_itself_from_its_origin():
    registry = reset_procedure_registry_for_test()
    procedure = registry.register(
        "compiled", Backend.GENERALIZED_RULE, sig(["a"], ["b"]),
        origin=Origin(
            learner="procedural_generalization", impasse_type="tie",
            support_keys=("a",), causal_events=(11, 12),
            rejected_conditions=("theme",), counterexamples=("case-7",),
        ),
    )
    explained = procedure.to_dict()["origin"]
    assert explained["rejected_conditions"] == ["theme"]
    assert explained["causal_events"] == [11, 12]


def test_a_rule_carries_the_conditions_that_were_tested_and_dropped():
    registry = reset_procedure_registry_for_test()
    rule = GeneralizedRule(
        conditions=frozenset({"a"}), resolution="do_x", tier=RuleTier.PROMOTED,
        supporting=9, lesioned=("wallpaper", "theme"),
    )
    procedure = from_generalized_rule(rule, value_when_it_works=1.0, registry=registry)
    assert set(procedure.origin.rejected_conditions) == {"wallpaper", "theme"}


# ── matching at scale ─────────────────────────────────────────────────────

def test_ten_times_the_procedures_costs_far_less_than_ten_times_the_match():
    def comparisons(count: int) -> int:
        registry = reset_procedure_registry_for_test()
        for i in range(count):
            registry.register(f"p{i}", Backend.MACRO, sig([f"key{i}"], ["done"]))
        registry.match({"key0": True})
        return registry.report()["index_comparisons"]

    assert comparisons(100) == comparisons(1000), (
        "matching must scale with what could apply, not with what exists"
    )


def test_a_procedure_with_no_preconditions_is_always_a_candidate():
    registry = reset_procedure_registry_for_test()
    registry.register("always", Backend.MACRO, sig([], ["done"]))
    assert [p.name for p in registry.match({})] == ["always"]


def test_a_never_observed_key_never_satisfies_a_precondition():
    registry = reset_procedure_registry_for_test()
    registry.register("needs_absence", Backend.MACRO,
                      Signature((Precondition("modal", negated=True),), ()))
    assert registry.match({}) == [], "not looking is not the same as looking and finding nothing"
    assert len(registry.match({"modal": False})) == 1


# ── composition across backends ───────────────────────────────────────────

def test_a_novel_skill_assembles_from_components_of_two_backends():
    registry = reset_procedure_registry_for_test()
    chunk = registry.register(
        "open", Backend.CHUNK, sig(["editor"], ["file_open"]),
        origin=Origin(learner="impasse"), value=ProceduralValue(p_success=0.9, value_when_it_works=10.0, match_cost=0.1),
    )
    macro = registry.register(
        "save", Backend.MACRO, sig(["file_open"], ["saved"]),
        value=ProceduralValue(p_success=0.95, value_when_it_works=5.0, match_cost=0.1),
    )
    composed = compose(registry, [chunk, macro])
    assert [p.key for p in composed.signature.preconditions] == ["editor"]
    assert {e.key for e in composed.signature.effects} == {"file_open", "saved"}
    assert registry.report()["composed_across_backends"] == 1


def test_composition_multiplies_failure_and_adds_cost():
    registry = reset_procedure_registry_for_test()
    a = registry.register("a", Backend.MACRO, sig(["s"], ["m"]), value=ProceduralValue(p_success=0.9, value_when_it_works=2.0, match_cost=0.1))
    b = registry.register("b", Backend.MACRO, sig(["m"], ["e"]), value=ProceduralValue(p_success=0.5, value_when_it_works=3.0, match_cost=0.2))
    composed = compose(registry, [a, b])
    assert composed.value.p_success == pytest.approx(0.45)
    assert composed.value.match_cost == pytest.approx(0.3)


def test_an_irreversible_part_makes_the_whole_irreversible():
    registry = reset_procedure_registry_for_test()
    a = registry.register("safe", Backend.MACRO, sig(["s"], ["m"]), reversibility=Reversibility.REVERSIBLE)
    b = registry.register("delete", Backend.TOOL, sig(["m"], ["gone"]), reversibility=Reversibility.IRREVERSIBLE)
    assert compose(registry, [a, b]).reversibility is Reversibility.IRREVERSIBLE


def test_a_learned_procedure_composes_with_a_tool_it_did_not_learn():
    registry = reset_procedure_registry_for_test()
    learned = registry.register(
        "find file", Backend.CHUNK, sig(["query"], ["path"]),
        origin=Origin(learner="impasse"), value=ProceduralValue(p_success=0.9, value_when_it_works=1.0),
    )
    tool = from_tool_schema("read_file", requires=["path"], produces=["contents"], registry=registry)
    composed = compose(registry, [learned, tool])
    assert {e.key for e in composed.signature.effects} == {"path", "contents"}


def test_composing_nothing_is_refused():
    registry = reset_procedure_registry_for_test()
    with pytest.raises(ValueError, match="nothing to compose"):
        compose(registry, [])


# ── lifecycle ─────────────────────────────────────────────────────────────

def test_a_counterexample_narrows_a_rule_without_deleting_it():
    registry = reset_procedure_registry_for_test()
    parent = registry.register(
        "broad", Backend.GENERALIZED_RULE, sig(["a"], ["done"]),
        origin=Origin(learner="pg"), value=ProceduralValue(p_success=0.8, value_when_it_works=5.0),
    )
    child = registry.specialise(parent.procedure_id, Precondition("not_modal"), counterexample="case-7")
    assert len(child.signature.preconditions) == 2
    assert registry.get(parent.procedure_id).retired is False
    assert child.origin.counterexamples == ("case-7",)
    assert child.value.uses == 0, "a narrowed rule earns its own evidence"


def test_merging_two_procedures_pools_evidence_without_double_counting():
    from core.evidence.packet import observe

    registry = reset_procedure_registry_for_test()
    a = registry.register(
        "a", Backend.MACRO, sig(["x"], ["y"]),
        value=ProceduralValue(p_success=0.8, value_when_it_works=1.0, uses=10, successes=8),
        evidence=observe(0.8, origin="run", ref="r1", mass=10.0, subject="p"),
    )
    b = registry.register(
        "b", Backend.MACRO, sig(["x"], ["y"]),
        value=ProceduralValue(p_success=0.9, value_when_it_works=1.0, uses=10, successes=9),
        evidence=observe(0.9, origin="run", ref="r1", mass=10.0, subject="p"),
    )
    merged = registry.merge(a.procedure_id, b.procedure_id)
    assert merged.value.uses == 20
    assert merged.evidence.mass == pytest.approx(10.0), "the same run counted once"
    assert registry.get(b.procedure_id).retired


def test_a_transfer_tier_is_carried_and_reported():
    registry = reset_procedure_registry_for_test()
    registry.register("a", Backend.MACRO, sig(["x"], ["y"]),
                      value=ProceduralValue(transfer_tier="structural_analogue"))
    registry.register("b", Backend.MACRO, sig(["x"], ["y"]))
    tiers = registry.report()["by_transfer_tier"]
    assert tiers["structural_analogue"] == 1 and tiers["same_instance"] == 1


def test_every_learner_lands_in_one_store():
    registry = reset_procedure_registry_for_test()

    class _Skill:
        name, description, parameters = "tidy", "", ["dir"]
        steps = [object()]
        successes, failures = 4, 1
        reliability = 0.8

    result = ingest_all(
        chunks=[Chunk("s", "r", ImpasseType.TIE, 2.0, 0.05, uses=4, correct=4)],
        rules=[GeneralizedRule(frozenset({"a"}), "do_x", RuleTier.PROMOTED, supporting=9)],
        skills=[_Skill()],
        rule_value=1.0, seconds_per_step=0.5, registry=registry,
    )
    assert result["landed"] == {"chunk": 1, "rule": 1, "skill": 1}
    assert registry.report()["backends_competing"] == 3


# ── automaticity: does practice actually get cheaper ──────────────────────

def test_practice_reduces_executive_cost():
    from core.cognition.automaticity import reset_automaticity_for_test

    index = reset_automaticity_for_test()
    for i, tokens in enumerate([900, 700, 500, 300, 150]):
        index.observe("rename files", cortex_tokens=tokens, planner_expansions=10 - i * 2,
                      seconds=2.0 - i * 0.3, procedure_hits=i)
    trend = index.trend("rename files")
    assert trend["automatic"] and trend["cost_slope"] < 0
    assert trend["reduction"] > 0.5


def test_procedures_that_fire_and_save_nothing_are_visible():
    from core.cognition.automaticity import reset_automaticity_for_test

    index = reset_automaticity_for_test()
    for i in range(6):
        index.observe("stubborn", cortex_tokens=800, seconds=2.0, procedure_hits=i)
    trend = index.trend("stubborn")
    assert trend["hits_without_saving"]
    assert not trend["automatic"]


def test_novelty_restores_deliberation_and_rigidity_is_the_alternative():
    from core.cognition.automaticity import reset_automaticity_for_test

    index = reset_automaticity_for_test()
    for _ in range(4):
        index.observe("open app", cortex_tokens=100, seconds=0.2)
    index.observe("open app", cortex_tokens=100, seconds=0.2, succeeded=False, variant="new layout")
    index.observe("open app", cortex_tokens=900, seconds=2.5, variant="new layout")
    assert index.rigidity("open app")["de_automatised"]

    rigid = reset_automaticity_for_test()
    for _ in range(4):
        rigid.observe("open app", cortex_tokens=100, seconds=0.2)
    rigid.observe("open app", cortex_tokens=100, seconds=0.2, succeeded=False)
    rigid.observe("open app", cortex_tokens=100, seconds=0.2)
    assert rigid.rigidity("open app")["rigid"]


def test_a_single_occurrence_says_nothing():
    from core.cognition.automaticity import reset_automaticity_for_test

    index = reset_automaticity_for_test()
    index.observe("once", cortex_tokens=100)
    assert not index.trend("once")["measurable"]


def test_the_cost_weights_travel_with_every_reading():
    from core.cognition.automaticity import CostWeights, reset_automaticity_for_test

    index = reset_automaticity_for_test(weights=CostWeights(per_second=5.0))
    index.observe("t", seconds=1.0)
    index.observe("t", seconds=0.5)
    assert index.trend("t")["weights"]["per_second"] == 5.0


# ── a rule that stops working stops paying ────────────────────────────────
#
# Card 026 asked what the utility accounting does over an accelerated
# lifetime with a distribution shift. Running it
# (tools/campaigns/procedure_lifetime.py) found three things the arithmetic
# could not express, and these are them.


def test_a_failed_firing_costs_something():
    """Without this term a rule wrong four times in five still 'pays'."""
    free = ProceduralValue(p_success=0.2, value_when_it_works=1.0)
    priced = ProceduralValue(
        p_success=0.2, value_when_it_works=1.0, cost_when_it_fails=1.0
    )
    assert free.pays
    assert not priced.pays
    assert priced.net == pytest.approx(0.2 - 0.8)


def test_the_lifetime_average_decides_until_there_is_recent_evidence():
    value = ProceduralValue(p_success=0.9, value_when_it_works=1.0)
    assert value.rate_that_decides == 0.9
    for _ in range(3):
        value = value.observed(success=False, at=0.0)
    # Three uses is not a measurement; the lifetime rate still decides.
    assert value.recent_weight < 30.0
    assert value.rate_that_decides == pytest.approx(value.p_success)


def test_a_rule_that_worked_a_million_times_can_still_be_retired():
    """A lifetime average cannot be moved, which is the whole difficulty."""
    value = ProceduralValue(
        p_success=1.0,
        value_when_it_works=1.0,
        cost_when_it_fails=1.0,
        uses=1_000_000,
        successes=1_000_000,
        recent_success=1.0,
        recent_weight=100.0,
    )
    for _ in range(200):
        value = value.observed(success=False, at=0.0)
    assert value.p_success > 0.999, "the lifetime rate has barely moved"
    assert not value.pays, "and it is retired anyway, on the recent evidence"


def test_an_unlucky_run_does_not_retire_a_rule_that_still_works():
    """Retirement reads the optimistic bound; a short bad run is not evidence."""
    import random

    from core.cognition.procedure import _RECENT_WEIGHT_FLOOR

    rng = random.Random(7)
    value = ProceduralValue(
        p_success=0.92,
        value_when_it_works=1.0,
        cost_when_it_fails=1.0,
        match_cost=0.005,
    )
    for _ in range(400):
        value = value.observed(success=rng.random() < 0.92, at=0.0)
    assert value.recent_weight >= _RECENT_WEIGHT_FLOOR
    assert value.pays


def test_a_reported_value_is_averaged_not_written_over():
    value = ProceduralValue(p_success=1.0, value_when_it_works=0.0)
    value = value.observed(success=True, at=0.0, value=10.0)
    value = value.observed(success=True, at=0.0, value=0.0)
    assert value.value_when_it_works == pytest.approx(5.0)


def test_a_value_reported_on_a_failure_is_not_what_it_is_worth_when_it_works():
    value = ProceduralValue(p_success=1.0, value_when_it_works=4.0, successes=1, uses=1)
    value = value.observed(success=False, at=0.0, value=99.0)
    assert value.value_when_it_works == pytest.approx(4.0)


def test_the_value_object_refuses_positional_arguments():
    """A field added in the middle silently changed what the third one meant."""
    with pytest.raises(TypeError):
        ProceduralValue(0.9, 2.0, 0.1)  # type: ignore[misc]


# ── the registry can widen, not only narrow ───────────────────────────────


def _compiled_with(registry, keys):
    return registry.register(
        "compiled:t",
        Backend.CHUNK,
        Signature(
            preconditions=tuple(Precondition(key=k) for k in keys),
            effects=(Effect(key="done"),),
        ),
        value=ProceduralValue(p_success=1.0, value_when_it_works=1.0, match_cost=0.001 * len(keys)),
        origin=Origin(learner="test", support_keys=tuple(keys), provisional_conditions=("clock",)),
    )


def test_a_condition_is_dropped_only_by_a_run_that_did_without_it():
    registry = reset_procedure_registry_for_test()
    parent = _compiled_with(registry, ["goal", "board", "clock"])
    with pytest.raises(ValueError, match="succeeded without it"):
        registry.generalise(parent.procedure_id, "clock", witness="")


def test_widening_keeps_the_parent_and_names_the_witness():
    registry = reset_procedure_registry_for_test()
    parent = _compiled_with(registry, ["goal", "board", "clock"])
    child = registry.generalise(parent.procedure_id, "clock", witness="run-412")
    assert child is not None
    assert [p.key for p in child.signature.preconditions] == ["goal", "board"]
    assert child.origin.generalisations == ("clock<-run-412",)
    assert "clock" not in child.origin.support_keys
    # Fewer conditions to check is less to pay for.
    assert child.value.match_cost < parent.value.match_cost
    # And the parent survives, for the states it still covers.
    assert registry.get(parent.procedure_id) is not None


def test_widening_a_condition_that_is_not_there_changes_nothing():
    registry = reset_procedure_registry_for_test()
    parent = _compiled_with(registry, ["goal"])
    assert registry.generalise(parent.procedure_id, "battery", witness="r1") is None


def test_the_compiler_names_the_conditions_a_witness_could_drop():
    """A key present in every run and different every time gates nothing."""
    from core.cognition.cognitive_event import EventGraph, Phase, ReadDependency
    from core.cognition.trace_compiler import TraceCompiler

    registry = reset_procedure_registry_for_test()
    graph = EventGraph(capacity=256)
    compiler = TraceCompiler(registry)
    for run in range(4):
        first = graph.record(
            Phase.PERCEIVE,
            "world",
            "read",
            reads=[
                ReadDependency(key="goal", value_digest="goal:same"),
                ReadDependency(key="clock", value_digest=f"clock:{run}"),
            ],
        )
        last = graph.record(
            Phase.APPLY, "actor", "do", parents=[first.seq], produced=["task:t:done"]
        )
        compiler.observe(graph, "t", last.seq)
    result = compiler.compile("t")
    assert result.compiled is not None
    assert result.compiled.origin.provisional_conditions == ("clock",)
    assert "goal" not in result.compiled.origin.provisional_conditions


def test_a_compiled_chunk_prices_its_own_misses():
    from core.cognition.cognitive_event import EventGraph, Phase, ReadDependency
    from core.cognition.trace_compiler import TraceCompiler

    registry = reset_procedure_registry_for_test()
    graph = EventGraph(capacity=256)
    compiler = TraceCompiler(registry)
    for _ in range(4):
        first = graph.record(
            Phase.PERCEIVE,
            "world",
            "read",
            reads=[ReadDependency(key="goal", value_digest="g")],
            duration_s=0.01,
        )
        last = graph.record(
            Phase.APPLY,
            "actor",
            "do",
            parents=[first.seq],
            produced=["task:t:done"],
            duration_s=0.02,
        )
        compiler.observe(graph, "t", last.seq)
    compiled = compiler.compile("t").compiled
    assert compiled is not None
    assert compiled.value.cost_when_it_fails == compiled.value.value_when_it_works > 0


def test_the_optimistic_and_pessimistic_readings_bracket_the_rate():
    from core.cognition.procedural_generalization import (
        wilson_lower_bound,
        wilson_upper_bound,
    )

    for successes, trials in ((3, 3), (8, 10), (50, 100), (900, 1000)):
        low = wilson_lower_bound(successes, trials)
        high = wilson_upper_bound(successes, trials)
        assert low <= successes / trials <= high
        assert high - low > 0.0
    # No evidence is not evidence of failure.
    assert wilson_upper_bound(0, 0) == 1.0
    assert wilson_lower_bound(0, 0) == 0.0
    # The interval narrows as evidence accumulates.
    narrow = wilson_upper_bound(900, 1000) - wilson_lower_bound(900, 1000)
    wide = wilson_upper_bound(9, 10) - wilson_lower_bound(9, 10)
    assert narrow < wide
