"""The hard version of self-repair, happening in a running system.

The ladder already reached code. A repeatedly failing or unrestartable
service escalates through `SelfHealing.request_deep_repair`, which gets or
registers the reimplementation lab, runs a governed reconstruction, and writes
a receipt. So repair is not

    failure → restart

it is

    failure → localisation → module path → governed reconstruction → receipt.

What `unknown_failure.py` adds is the step before all of that:

    new symptom → recognise as genuinely novel → infer the violated invariant
    → try the general repairs → learn this failure class

so the next occurrence is not novel. That module was complete, correct, and
reached by nothing. A recogniser no failure reaches is a description of a
capability rather than the capability, and this file is the difference.

Three things are checked: the registry teaches it what known faults look like,
the ladder asks it before escalating, and a concept minted for a repair that
held survives the process it was minted in.
"""

from __future__ import annotations

import pytest

from core.resilience import unknown_failure as unknown
from core.resilience.fault_taxonomy import FaultRegistry, FaultSeverity
from core.resilience.unknown_failure import (
    MIN_INSTANCES,
    MIN_KNOWN_FAULTS,
    Recognition,
    Signature,
)


@pytest.fixture
def an_ontology(tmp_path, monkeypatch):
    """A fresh ontology writing to a file of its own."""

    monkeypatch.setattr(unknown, "_ONTOLOGY", None)
    monkeypatch.setattr(unknown, "_ATTACHED", [False])
    monkeypatch.setattr(
        unknown, "_where_it_is_kept", lambda: tmp_path / "what_failure_looks_like.json"
    )
    return unknown.get_failure_ontology()


def _a_fault_registry_that_teaches_it() -> FaultRegistry:
    registry = FaultRegistry()
    assert unknown.attach_to_the_fault_registry(registry) is True
    return registry


def _a_life_of_known_failures(registry: FaultRegistry) -> None:
    """Enough instances of enough distinct faults for novelty to mean anything."""

    for _ in range(MIN_INSTANCES + 1):
        registry.record_fault(
            "FAULT-MEM-001", "memory_facade",
            details="pressure", severity=FaultSeverity.MARGINAL,
        )
        registry.record_fault(
            "FAULT-NET-001", "connectivity",
            details="timeout", severity=FaultSeverity.NEGLIGIBLE,
            recovered=True, recovery_time_s=0.5,
        )
        registry.record_fault(
            "FAULT-DB-001", "persistence",
            details="locked", severity=FaultSeverity.CRITICAL,
            recovery_time_s=30.0,
        )


def test_the_registry_teaches_it_what_known_faults_look_like(an_ontology):
    """Learned from instances, not from the prose in a catalogue entry."""

    registry = _a_fault_registry_that_teaches_it()
    assert an_ontology.known_faults == ()
    _a_life_of_known_failures(registry)
    assert set(an_ontology.known_faults) >= {
        "FAULT-MEM-001", "FAULT-NET-001", "FAULT-DB-001"
    }


def test_a_repeat_of_a_known_failure_comes_back_known(an_ontology):
    """The null. A recogniser that calls everything novel is useless."""

    registry = _a_fault_registry_that_teaches_it()
    _a_life_of_known_failures(registry)
    again = registry.record_fault(
        "FAULT-DB-001", "persistence",
        details="locked", severity=FaultSeverity.CRITICAL,
        recovery_time_s=30.0,
    )
    verdict = an_ontology.recognise(unknown.signature_of(again))
    assert verdict.recognition is Recognition.KNOWN
    assert verdict.nearest == "FAULT-DB-001"


def test_before_enough_is_known_novelty_is_not_a_finding(an_ontology):
    registry = _a_fault_registry_that_teaches_it()
    for _ in range(MIN_INSTANCES + 1):
        registry.record_fault("FAULT-MEM-001", "memory_facade", details="pressure")
    assert len(an_ontology.known_faults) < MIN_KNOWN_FAULTS
    verdict = an_ontology.recognise(
        Signature(subsystem="somewhere_new", kind="SomethingElse")
    )
    assert verdict.recognition is Recognition.UNDECIDABLE


def test_the_ladder_asks_before_it_escalates(an_ontology):
    """The seam. `look_at_this_failure` is what the healing path now calls."""

    registry = _a_fault_registry_that_teaches_it()
    _a_life_of_known_failures(registry)
    seen = unknown.look_at_this_failure(
        "a_subsystem_nobody_has_seen_fail",
        "AnUnheardOfError",
        broken_invariants=("container.no_dangling_alias",),
        observations={"severity": 4.0, "recovery_seconds": 900.0},
    )
    assert seen.verdict.recognition is Recognition.NOVEL
    assert seen.verdict.needs_a_new_concept
    assert seen.verdict.broken_invariant == "container.no_dangling_alias"
    assert seen.repairs, "a novel failure with no repairs proposed is a dead end"
    # Cheapest first, and every one verified against the invariant that broke.
    assert [one.irreversibility for one in seen.repairs] == sorted(
        one.irreversibility for one in seen.repairs
    )
    assert all(one.restores == "container.no_dangling_alias" for one in seen.repairs)


def test_a_repair_that_held_makes_the_failure_known_next_time(an_ontology):
    """Step six, which is what makes this learning rather than handling."""

    registry = _a_fault_registry_that_teaches_it()
    _a_life_of_known_failures(registry)
    where = dict(severity=4.0, recovery_seconds=900.0, message_length=40.0)
    first = unknown.look_at_this_failure(
        "a_subsystem_nobody_has_seen_fail", "AnUnheardOfError", observations=where
    )
    assert first.verdict.recognition is Recognition.NOVEL

    named = unknown.a_repair_that_held(first, called="core.somewhere.new")
    assert named == "FAULT-NOVEL-core.somewhere.new"

    # One instance is a point, not a signature: it stays novel until it has
    # been seen enough times to have a shape.
    for _ in range(MIN_INSTANCES):
        an_ontology.observe(named, first.signature)
    again = unknown.look_at_this_failure(
        "a_subsystem_nobody_has_seen_fail", "AnUnheardOfError", observations=where
    )
    assert again.verdict.recognition is Recognition.KNOWN
    assert again.verdict.nearest == named


def test_a_repair_that_did_not_hold_mints_nothing(an_ontology):
    """A concept for a failure still happening teaches the recogniser the
    broken state, and every later occurrence returns KNOWN knowing nothing."""

    registry = _a_fault_registry_that_teaches_it()
    _a_life_of_known_failures(registry)
    again = registry.record_fault(
        "FAULT-DB-001", "persistence",
        details="locked", severity=FaultSeverity.CRITICAL, recovery_time_s=30.0,
    )
    signature = unknown.signature_of(again)
    known = unknown.Diagnosis(
        verdict=an_ontology.recognise(signature), signature=signature
    )
    assert known.verdict.recognition is Recognition.KNOWN
    assert unknown.a_repair_that_held(known, called="core.persistence") == ""
    assert an_ontology.invented == ()


def test_what_failure_looks_like_survives_the_process(an_ontology, tmp_path, monkeypatch):
    """A concept invented on Tuesday and forgotten at the restart is
    rediscovered at full diagnostic cost on Wednesday."""

    registry = _a_fault_registry_that_teaches_it()
    _a_life_of_known_failures(registry)
    where = dict(severity=4.0, recovery_seconds=900.0, message_length=40.0)
    found = unknown.look_at_this_failure("elsewhere", "NeverSeen", observations=where)
    named = unknown.a_repair_that_held(found, called="core.elsewhere")
    for _ in range(MIN_INSTANCES):
        an_ontology.observe(named, found.signature)
    assert an_ontology.keep() is True

    # The restart.
    monkeypatch.setattr(unknown, "_ONTOLOGY", None)
    monkeypatch.setattr(unknown, "_ATTACHED", [False])
    restarted = unknown.get_failure_ontology()
    assert named in restarted.known_faults
    assert named in restarted.invented
    verdict = restarted.recognise(found.signature)
    assert verdict.recognition is Recognition.KNOWN


def test_an_unmeasured_channel_is_a_gap_not_a_zero(an_ontology):
    """Absence is not a measurement, and reading it as one broke both ways.

    A failure reported with no numbers looked like a failure whose every
    number was zero — far from everything, so always novel. Two failures
    neither of which was measured looked identical — so the second was
    always known. Where nothing is shared the comparison is the categorical
    one, which is a smaller claim honestly made.
    """

    from core.resilience.unknown_failure import _distance

    ranges = {"latency": (0.0, 1000.0), "severity": (0.0, 4.0)}
    keys = ("latency", "severity")
    measured = Signature("a", "B", observations={"latency": 900.0, "severity": 4.0})
    unmeasured = Signature("a", "B")
    other = Signature("a", "B", observations={"severity": 4.0})

    # No shared channel: the categorical comparison, and they agree on all of it.
    assert _distance(measured, unmeasured, keys, ranges) == 0.0
    # One shared channel, agreeing: the unshared one is not a difference.
    assert _distance(measured, other, keys, ranges) == 0.0
    # One shared channel, disagreeing: a real distance.
    apart = Signature("a", "B", observations={"severity": 0.0})
    assert _distance(measured, apart, keys, ranges) > 0.0


def test_learning_from_a_fault_never_raises(an_ontology):
    """The listener runs inside the registry's own recording path."""

    unknown.learn_from_fault(object())
    unknown.learn_from_fault(None)


def test_the_healing_ladder_carries_the_diagnosis_on_its_receipt():
    """The other half of the seam, at its call site."""

    from core.runtime import self_healing

    assert hasattr(self_healing, "_diagnose_this_failure")
    assert hasattr(self_healing, "_learn_this_failure_class")
    body = (
        self_healing.SelfHealing.request_deep_repair.__doc__ or ""
    )
    source = __import__("inspect").getsource(self_healing.SelfHealing.request_deep_repair)
    assert "_diagnose_this_failure" in source
    assert 'record["diagnosis"]' in source
    assert "_learn_this_failure_class" in source
    assert body
