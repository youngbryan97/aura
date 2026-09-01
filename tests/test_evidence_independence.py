"""The bar: duplicate evidence cannot raise confidence, independent evidence can.

Cards 031, 032, 033, 041, 105, 170, 171 of the 2026-09-01 comparative review.
The defect these close was reproducible in four lines against the shipped
AtomSpace: one sensor reading asserted ten times took confidence from 0.44 to
0.98, because PLN revision is documented as merging "two independent estimates"
and had no way to tell whether they were.
"""
from __future__ import annotations

import pytest

from core.evidence.packet import (
    EvidenceKind,
    EvidencePacket,
    derive,
    from_probability,
    from_truth_value,
    from_wilson,
    fuse,
    independent_mass,
    observe,
)
from core.evidence.state_ref import (
    CognitiveStateRef,
    StateOwnershipError,
    handoff_coverage,
    record_handoff,
    register_handoff,
    reset_handoff_ledger_for_test,
)
from core.knowledge.atomspace import AtomSpace, TruthValue, assert_claim, concept


# ── the packet ────────────────────────────────────────────────────────────

def test_the_same_observation_ten_times_is_one_observation():
    once = observe(0.9, origin="sensor", ref="r1", mass=4.0, subject="rain")
    ten = fuse([once] * 10)
    assert ten.mass == pytest.approx(once.mass)
    assert ten.confidence == pytest.approx(once.confidence)
    assert ten.independent_sources == 1


def test_ten_different_observations_are_ten_observations():
    packets = [observe(0.9, origin="sensor", ref=f"r{i}", mass=4.0, subject="rain") for i in range(10)]
    merged = fuse(packets)
    assert merged.independent_sources == 10
    assert merged.mass == pytest.approx(40.0)
    assert merged.confidence > packets[0].confidence


def test_fusing_a_packet_with_itself_is_a_fixed_point():
    packet = observe(0.7, origin="tool", ref="run-9", mass=6.0, subject="q")
    again = packet.fuse(packet)
    assert again.strength == pytest.approx(packet.strength)
    assert again.mass == pytest.approx(packet.mass)
    assert again.sources == packet.sources


def test_a_duplicate_cannot_drag_strength_toward_itself():
    strong = observe(0.9, origin="a", ref="1", mass=10.0, subject="q")
    weak = observe(0.1, origin="b", ref="2", mass=1.0, subject="q")
    honest = fuse([strong, weak])
    stuffed = fuse([strong, weak] + [weak] * 20)
    assert stuffed.strength == pytest.approx(honest.strength)
    assert stuffed.mass == pytest.approx(honest.mass)


def test_a_derivation_inherits_its_premises_and_mints_nothing():
    a = observe(0.9, origin="s", ref="1", mass=4.0, subject="q")
    b = observe(0.8, origin="s", ref="2", mass=4.0, subject="q")
    conclusion = derive(0.85, [a, b], subject="q")
    assert conclusion.kind is EvidenceKind.DERIVATION
    assert conclusion.sources == a.sources | b.sources
    assert conclusion.mass <= min(a.mass, b.mass)


def test_a_long_chain_does_not_manufacture_support():
    seed = observe(0.9, origin="s", ref="only", mass=4.0, subject="q")
    step = seed
    for _ in range(20):
        step = derive(step.strength, [step], subject="q")
    assert step.independent_sources == 1
    assert fuse([seed, step]).confidence <= seed.confidence


def test_two_routes_through_one_observation_fuse_back_to_it():
    seed = observe(0.9, origin="s", ref="only", mass=4.0, subject="q")
    left = derive(0.9, [seed], subject="q")
    right = derive(0.9, [seed], subject="q")
    assert fuse([left, right]).independent_sources == 1
    assert fuse([left, right]).mass <= seed.mass


def test_a_derivation_with_no_sources_is_refused():
    with pytest.raises(ValueError, match="support it cannot name"):
        EvidencePacket(0.5, 1.0, kind=EvidenceKind.DERIVATION)
    with pytest.raises(ValueError, match="needs premises"):
        derive(0.5, [])


def test_fusing_across_subjects_is_refused():
    a = observe(0.9, origin="s", ref="1", subject="is it raining")
    b = observe(0.9, origin="s", ref="2", subject="is it tuesday")
    with pytest.raises(ValueError, match="different subjects"):
        fuse([a, b])


def test_unattributed_evidence_is_counted_not_discarded():
    bare = EvidencePacket(0.9, 4.0)
    assert bare.sources == frozenset()
    assert independent_mass([bare, bare]) == pytest.approx(8.0)


# ── the adapters ──────────────────────────────────────────────────────────

def test_adapters_preserve_mass_and_mint_one_source_each():
    tv = from_truth_value(TruthValue(0.8, 12.0), origin="atomspace", ref="a1")
    wilson = from_wilson(8, 10, origin="rules", ref="rule-3")
    prob = from_probability(0.7, origin="cortex", ref="turn-4")
    assert tv.mass == pytest.approx(12.0)
    assert wilson.mass == pytest.approx(10.0)
    assert prob.mass == pytest.approx(1.0), "a probability nobody counted is one opinion"
    for packet in (tv, wilson, prob):
        assert packet.independent_sources == 1


def test_a_wilson_batch_is_one_report_not_n_reports():
    ten_trials = from_wilson(8, 10, origin="rules", ref="rule-3")
    again = from_wilson(8, 10, origin="rules", ref="rule-3")
    assert fuse([ten_trials, again]).mass == pytest.approx(10.0)


def test_converting_and_reconverting_gains_no_independence():
    from core.evidence.packet import to_truth_value

    packet = from_wilson(8, 10, origin="rules", ref="rule-3")
    round_tripped = from_truth_value(to_truth_value(packet), origin="rules", ref="rule-3")
    assert fuse([packet, round_tripped]).mass == pytest.approx(packet.mass)


# ── the AtomSpace, where the defect lived ─────────────────────────────────

def test_one_sensor_reading_asserted_ten_times_stays_one_reading():
    space = AtomSpace()
    atom = concept("rain")
    for _ in range(10):
        space.add(atom, TruthValue(0.9, 4.0), source="sensor:r1")
    assert space.get_tv(atom).count == pytest.approx(4.0)
    assert space.evidence_report()["duplicate_assertions_refused"] == 9


def test_ten_sensors_still_accumulate():
    space = AtomSpace()
    atom = concept("rain")
    for i in range(10):
        space.add(atom, TruthValue(0.9, 4.0), source=f"sensor:r{i}")
    assert space.get_tv(atom).count == pytest.approx(40.0)


def test_a_source_restating_a_growing_total_replaces_its_own_contribution():
    """The live belief-mirror defect: evidence_count is already accumulated."""
    space = AtomSpace()
    atom = concept("belief")
    for mass in (1.0, 2.0, 3.0, 8.0, 20.0):
        space.add(atom, TruthValue(0.8, mass), source="belief:b1")
    assert space.get_tv(atom).count == pytest.approx(20.0), (
        "the atom must hold the belief's current mass, not the sum of every "
        "total it has ever reported"
    )


def test_an_unsourced_assertion_keeps_the_old_behaviour_and_is_counted():
    space = AtomSpace()
    atom = concept("legacy")
    for _ in range(10):
        space.add(atom, TruthValue(0.9, 4.0))
    assert space.get_tv(atom).count == pytest.approx(40.0)
    assert space.evidence_report()["unattributed_assertions"] == 9


def test_the_source_map_is_bounded():
    from core.knowledge.atomspace import _MAX_SOURCES_PER_ATOM

    space = AtomSpace()
    atom = concept("busy")
    for i in range(_MAX_SOURCES_PER_ATOM + 50):
        space.add(atom, TruthValue(0.5, 1.0), source=f"s{i}")
    assert len(space.evidence_sources(atom)) <= _MAX_SOURCES_PER_ATOM


def test_assert_claim_carries_a_source_through():
    space = AtomSpace()
    for _ in range(5):
        atom, tv = assert_claim(space, "the kettle is on", TruthValue(0.8, 3.0), source="belief:k1")
    assert tv.count == pytest.approx(3.0)
    assert "belief:k1" in space.evidence_sources(atom)


def test_the_domain_link_is_a_stipulation_not_evidence():
    space = AtomSpace()
    for _ in range(20):
        assert_claim(space, "the kettle is on", TruthValue(0.8, 3.0), source="belief:k1")
    from core.knowledge.atomspace import concept as c, evaluation, predicate

    link = evaluation(predicate("claim_domain"), c("the kettle is on"), c("world"))
    tv = space.get_tv(link)
    if tv is not None:
        assert tv.count <= 1.0


# ── the state envelope ────────────────────────────────────────────────────

def test_state_with_no_evidence_reports_zero_confidence_not_a_coin_flip():
    ref = CognitiveStateRef(kind="percept", payload={"x": 1}, owner="perception")
    assert ref.confidence == 0.0


def test_derived_state_carries_its_causal_parent():
    first = CognitiveStateRef(kind="percept", payload={"x": 1}, owner="perception")
    second = first.derive({"x": 2}, owner="world_model")
    assert second.parents == (first.identity,)
    assert second.version == first.version + 1


def test_parent_chains_stay_bounded():
    ref = CognitiveStateRef(kind="s", payload=0, owner="o")
    for i in range(100):
        ref = ref.derive(i)
    assert len(ref.parents) <= 16


def test_mutating_state_you_do_not_own_is_refused():
    ref = CognitiveStateRef(kind="percept", payload={"x": 1}, owner="perception")
    with pytest.raises(StateOwnershipError):
        ref.mutated_by("planner", {"x": 2})
    assert ref.mutated_by("perception", {"x": 2}).payload == {"x": 2}


def test_identity_is_content_addressed():
    a = CognitiveStateRef(kind="k", payload={"x": 1}, owner="o")
    b = CognitiveStateRef(kind="k", payload={"x": 1}, owner="other")
    c = CognitiveStateRef(kind="k", payload={"x": 2}, owner="o")
    assert a.identity == b.identity
    assert a.identity != c.identity


def test_coverage_counts_what_actually_flows():
    reset_handoff_ledger_for_test()
    register_handoff("perception->world_model", "frames into the world model")
    record_handoff("perception->world_model", CognitiveStateRef(kind="p", payload=1, owner="perception"))
    record_handoff("perception->world_model", {"raw": 1})
    report = handoff_coverage()
    assert report["coverage"] == pytest.approx(0.5)
    assert report["by_handoff"]["perception->world_model"]["bare"] == 1
    reset_handoff_ledger_for_test()
