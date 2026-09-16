"""Revising evidence changes the conclusions that actually depend on it."""

import pytest

from core.knowledge.atomspace import AtomSpace, InferenceRule, TruthValue, concept
from core.knowledge.atomspace_persistence import restore, snapshot


def copying(name, premise, conclusion):
    return InferenceRule(name, (premise,), conclusion,
                         lambda _space, _binding, tvs: TruthValue(tvs[0].strength, tvs[0].count * .9))


def test_first_versioned_revision_replaces_legacy_claims_from_same_source():
    space = AtomSpace()
    old, new, derived = map(concept, ("old-claim", "new-claim", "legacy-conclusion"))
    space.add(old, TruthValue(.9, 10), source="observation")
    space.apply_rules((copying("legacy", old, derived),), focus_only=False)
    space.revise_observation("observation", 0, {new: TruthValue(.8, 10)})
    assert space.get_tv(old).count == 0
    assert space.get_tv(derived).count == 0
    assert space.get_tv(new) == TruthValue(.8, 10)


def test_failed_snapshot_metadata_conversion_preserves_destination():
    from core.knowledge.atomspace_persistence import restore, snapshot
    destination = AtomSpace()
    destination.revise_observation("existing", 3, {concept("kept"): TruthValue(.8, 5)})
    before = snapshot(destination)
    replacement = AtomSpace()
    replacement.add(concept("replacement"), TruthValue(.9, 2))
    payload = snapshot(replacement)
    payload["unattributed_assertions"] = "invalid-counter"
    with pytest.raises(ValueError):
        restore(destination, payload)
    assert snapshot(destination) == before


def test_grounded_filter_truth_reads_invalidate_derived_support():
    from core.knowledge.atomspace import Node, GROUNDED_PREDICATE, evaluation
    space = AtomSpace()
    a, gate, result = map(concept, ("fact", "filter-input", "filtered-result"))
    space.add(a, TruthValue(.8, 10), source="fact")
    space.add(gate, TruthValue(.9, 10), source="filter")
    space.register_grounded("allowed", lambda: space.get_tv(gate).strength > .5)
    clause = evaluation(Node(GROUNDED_PREDICATE, "allowed"))
    rule = InferenceRule("filtered", (a, clause), result,
                         lambda _s, _b, tvs: TruthValue(tvs[0].strength, 5))
    assert space.apply_rules((rule,), focus_only=False) == [result]
    space.add(gate, TruthValue(.1, 10), source="filter")
    assert space.get_tv(result).count == 0
    assert space.apply_rules((rule,), focus_only=False) == []


def chain():
    space = AtomSpace()
    a, b, c = map(concept, ("observation", "inference", "decision"))
    rules = (copying("ab", a, b), copying("bc", b, c))
    space.add(a, TruthValue(.9, 10), source="instrument:reading")
    space.apply_rules(rules, focus_only=False)
    return space, a, b, c, rules


def test_same_mass_correction_invalidates_and_recomputes_entire_chain():
    space, a, b, c, rules = chain()
    assert space.get_tv(c).strength == .9
    space.add(a, TruthValue(.1, 10), source="instrument:reading")
    assert space.get_tv(b).count == 0
    assert space.get_tv(c).count == 0
    space.apply_rules(rules, focus_only=False)
    assert space.get_tv(b).strength == .1
    assert space.get_tv(c).strength == .1
    assert space.get_tv(c).count == pytest.approx(8.1)


def test_retraction_preserves_independent_direct_support():
    space, a, b, c, rules = chain()
    space.add(b, TruthValue(.4, 3), source="independent:measurement")
    assert space.retract_source(a, "instrument:reading")
    assert space.get_tv(a).count == 0
    assert space.get_tv(b) == TruthValue(.4, 3)
    assert space.get_tv(c).count == 0
    space.apply_rules(rules, focus_only=False)
    assert space.get_tv(c).strength == .4
    assert not space.retract_source(a, "instrument:reading")


def test_rule_replay_and_two_paths_do_not_mint_independent_evidence():
    space, a, b, c, rules = chain()
    second = copying("ac", a, c)
    space.apply_rules((*rules, second), focus_only=False)
    before = space.get_tv(c)
    assert before.count == pytest.approx(9)
    for _ in range(5):
        assert space.apply_rules((*rules, second), focus_only=False) == []
    assert space.get_tv(c) == before


def test_auxiliary_truth_read_is_a_dependency_even_when_previously_absent():
    space = AtomSpace()
    a, b, prior = map(concept, ("a", "b", "prior"))
    rule = InferenceRule("prior", (a,), b,
        lambda s, _binding, tvs: TruthValue((s.get_tv(prior) or TruthValue()).strength,
                                          tvs[0].count * .9))
    space.add(a, TruthValue(.9, 10), source="a")
    space.apply_rules((rule,), focus_only=False)
    assert space.get_tv(b).strength == .5
    space.add(prior, TruthValue(.2, 2), source="prior")
    assert space.get_tv(b).count == 0
    space.apply_rules((rule,), focus_only=False)
    assert space.get_tv(b).strength == .2


def test_circular_support_cannot_survive_retraction():
    space, a, b, c, rules = chain()
    space.apply_rules((*rules, copying("ca", c, a)), focus_only=False)
    space.retract_source(a, "instrument:reading")
    space.apply_rules((*rules, copying("ca", c, a)), focus_only=False)
    assert all(space.get_tv(atom).count == 0 for atom in (a, b, c))


def test_restart_preserves_dependency_invalidation():
    space, a, b, c, rules = chain()
    fresh = AtomSpace()
    restore(fresh, snapshot(space))
    fresh.add(a, TruthValue(.2, 2), source="instrument:reading")
    assert fresh.get_tv(b).count == fresh.get_tv(c).count == 0
    fresh.apply_rules(rules, focus_only=False)
    assert fresh.get_tv(c).strength == .2
    assert fresh.get_tv(c).count == pytest.approx(1.62)


def test_versioned_observation_replaces_changed_claim_without_rewriting_other_sources():
    space = AtomSpace()
    old, new, conclusion = map(concept, ("old-value", "new-value", "consequence"))
    rule = copying("consequence", old, conclusion)
    space.revise_observation("measurement", 1, {old: TruthValue(.9, 5)})
    space.add(old, TruthValue(.3, 2), source="independent")
    space.apply_rules((rule,), focus_only=False)
    assert space.revise_observation("measurement", 2, {new: TruthValue(.8, 5)})
    assert space.get_tv(old) == TruthValue(.3, 2)
    assert space.get_tv(conclusion).count == 0
    assert space.get_tv(new) == TruthValue(.8, 5)
    assert not space.revise_observation("measurement", 1, {old: TruthValue(.9, 5)})
    assert space.get_tv(old) == TruthValue(.3, 2)


def test_retraction_watermark_survives_restart_and_duplicate_revisions_are_checked():
    space = AtomSpace()
    atom = concept("versioned")
    claims = {atom: TruthValue(.9, 5)}
    assert space.revise_observation("source", 7, claims)
    assert not space.revise_observation("source", 7, claims)
    with pytest.raises(ValueError, match="conflicting payloads"):
        space.revise_observation("source", 7, {atom: TruthValue(.1, 5)})
    with pytest.raises(ValueError, match="versioned"):
        space.add(atom, TruthValue(.1, 5), source="source")
    space.revise_observation("source", 8, {})
    fresh = AtomSpace()
    restore(fresh, snapshot(space))
    assert not fresh.revise_observation("source", 7, claims)
    assert fresh.get_tv(atom).count == 0


def test_partial_revision_changes_only_the_declared_obligations():
    space = AtomSpace()
    a, b = concept("part-a"), concept("part-b")
    space.revise_observation("check", 1, {a: TruthValue(.9, 4), b: TruthValue(.8, 4)})
    space.revise_observation("check", 2, {a: TruthValue(.1, 4), b: TruthValue(.8, 4)})
    assert space.get_tv(a).strength == .1
    assert space.get_tv(b).strength == .8


def test_invalid_transaction_does_not_partially_revise():
    from core.knowledge.atomspace import Variable
    space = AtomSpace()
    a = concept("a")
    space.revise_observation("source", 1, {a: TruthValue(.9, 4)})
    before = snapshot(space)
    with pytest.raises(ValueError, match="ground atoms"):
        space.revise_observation("source", 2, {a: TruthValue(.1, 4), Variable("x"): TruthValue()})
    assert snapshot(space) == before


def test_versioned_sources_are_not_compacted_into_irretractable_evidence():
    from core.knowledge.atomspace import _MAX_SOURCES_PER_ATOM
    space = AtomSpace()
    a = concept("popular")
    space.revise_observation("owned", 1, {a: TruthValue(.9, 4)})
    for i in range(_MAX_SOURCES_PER_ATOM + 1):
        space.add(a, TruthValue(.5, 1), source=f"legacy-{i}")
    space.revise_observation("owned", 2, {})
    assert space.get_tv(a).strength == .5


def test_different_rule_names_cannot_multiply_the_same_evidence():
    space = AtomSpace()
    a, b = concept("a"), concept("b")
    space.add(a, TruthValue(.9, 10), source="only-source")
    space.apply_rules(tuple(copying(f"rule-{i}", a, b) for i in range(10)), focus_only=False)
    assert space.get_tv(b).count == 9


def test_eviction_invalidates_derived_support():
    space, a, b, c, _rules = chain()
    with space._lock:
        space._evict_locked(a)
    assert space.get_tv(b).count == space.get_tv(c).count == 0


def test_contradiction_revision_reaches_the_canonical_graph(monkeypatch):
    from core.epistemics.belief_revision import Belief, BeliefRevisionEngine
    from core.knowledge.atomspace import assert_claim
    space = AtomSpace()
    monkeypatch.setattr("core.knowledge.atomspace.get_atomspace", lambda: space)
    engine = BeliefRevisionEngine()
    belief = Belief("b1", "the lamp is on", .8, "world", "tool_result")
    opposite = Belief("b2", "the lamp is not on", .9, "world", "tool_result")
    engine.beliefs = [belief, opposite]
    engine._mirror_claim_to_atomspace(belief)
    atom, _ = assert_claim(space, belief.content, TruthValue(), source="test-shape", stimulate=False)
    engine._resolve_logical_conflicts([(belief.content, opposite.content)])
    assert belief.revision == 1
    assert space.get_tv(atom).strength == pytest.approx(.48)


def test_shared_state_envelope_carries_invalidation_and_recomputed_version():
    space, a, b, _c, rules = chain()
    before = space.evidence_state(b)
    assert before.evidence.sources == frozenset({"instrument:reading"})
    assert before.parents == (space.evidence_state(a).identity,)
    space.add(a, TruthValue(.1, 10), source="instrument:reading")
    invalidated = space.evidence_state(b)
    assert invalidated.identity == before.identity
    assert invalidated.version > before.version
    assert invalidated.confidence == 0
    space.apply_rules(rules, focus_only=False)
    assert space.evidence_state(b).version > invalidated.version


def test_revision_invariant_detects_corruption_instead_of_reporting_empty_success():
    from core.knowledge.revision_validation import revision_canary, revision_violations
    assert revision_canary()
    space, _a, _b, _c, _rules = chain()
    assert revision_violations(space) == []
    space._dependents.clear()
    assert "dependency index disagrees" in revision_violations(space)[0]


def test_decay_revises_the_graph_without_changing_last_observation_time(monkeypatch):
    from core.epistemics.belief_revision import Belief, BeliefRevisionEngine
    space = AtomSpace()
    monkeypatch.setattr("core.knowledge.atomspace.get_atomspace", lambda: space)
    engine = BeliefRevisionEngine()
    belief = Belief("decay", "the lamp is on", .9, "world", "tool_result", last_updated=1.)
    engine.beliefs = [belief]
    engine._mirror_claim_to_atomspace(belief)
    assert engine.apply_decay(now=864001.) == 1
    assert belief.last_updated == 1.
    assert belief.revision == 1
    assert space._observations["belief:decay"][0] == 1
    confidence = belief.confidence
    assert engine.apply_decay(now=864001.) == 0
    assert belief.confidence == confidence
    assert belief.revision == 1


def test_stale_support_is_detected_even_if_the_dependency_index_was_present():
    from core.knowledge.revision_validation import revision_violations
    space, a, _b, _c, _rules = chain()
    space._records[a].revision += 1
    assert any("stale premise" in reason for reason in revision_violations(space))
    with pytest.raises(ValueError, match="stale premise"):
        restore(AtomSpace(), snapshot(space))


@pytest.mark.parametrize("depth", [1, 2, 8, 32])
def test_revision_propagates_at_different_depths_without_task_specific_rules(depth):
    space = AtomSpace()
    nodes = [concept(f"variable-{i}") for i in range(depth + 1)]
    rules = tuple(copying(f"edge-{i}", nodes[i], nodes[i + 1]) for i in range(depth))
    for revision, strength in enumerate((.9, .2, .7, .1)):
        space.revise_observation("source", revision, {nodes[0]: TruthValue(strength, 10)})
        space.apply_rules(rules, max_derivations=depth, focus_only=False)
        assert space.get_tv(nodes[-1]).strength == pytest.approx(strength)
        assert space.get_tv(nodes[-1]).count == pytest.approx(10 * .9 ** depth)
    space.revise_observation("source", 4, {})
    assert all(space.get_tv(n).count == 0 for n in nodes)


def test_corrupted_cycle_snapshot_does_not_replace_existing_store():
    from core.knowledge.atomspace_persistence import encode_atom
    space, a, b, _c, _rules = chain()
    payload = snapshot(space)
    for row in payload["atoms"]:
        if row["atom"] == encode_atom(b):
            d = next(iter(row["derivations"].values()))
            d["dependencies"] = [encode_atom(b)]
            d["premise_revisions"] = [[encode_atom(b), row["revision"]]]
    before = snapshot(space)
    with pytest.raises(ValueError, match="circular"):
        restore(space, payload)
    assert snapshot(space) == before


def test_registered_canary_carries_its_result_instead_of_only_a_claim():
    from core.organism.model_validation import (
        ValidationSuite, _install_knowledge_revision_claims, Outcome,
    )
    suite = ValidationSuite()
    _install_knowledge_revision_claims(suite)
    test = next(t for t in suite.tests() if t.name == "knowledge_revision_propagates_corrections")
    class Model:
        name = "revision-test"
        def capabilities(self):
            return set()
    assert test.run(Model()).score.outcome == Outcome.PASS
