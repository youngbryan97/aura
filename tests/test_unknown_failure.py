"""Recognising a failure the system has no concept for.

fault_taxonomy.py is an FMEA catalogue and it has the property every catalogue
has: record_fault takes a fault_id, so a failure that is not in it must be
forced into an entry that nearly fits, or dropped. Both lose the interesting
case, which is the hard version of self-repair — recognise that this is not
anything known, infer what broke, localise it, invent a repair, verify it, and
integrate the concept so next time it is known.

Step one is the one that goes quietly wrong, so it is tested against its null:
a recogniser that calls everything novel is as useless as one that calls
nothing novel, and only one of the two looks like it is working.
"""

from __future__ import annotations

import pytest

from core.resilience.unknown_failure import (
    MIN_INSTANCES,
    MIN_KNOWN_FAULTS,
    FailureOntology,
    Recognition,
    Repair,
    RepairOutcome,
    Signature,
    propose_repairs,
)

CATALOGUE = {
    "FAULT-MEM-001": dict(
        subsystem="memory", kind="OutOfMemory",
        broken_invariants=("memory.bounded",),
        observations={"rss_gb": 14.0, "latency_ms": 900.0},
    ),
    "FAULT-NET-002": dict(
        subsystem="network", kind="TimeoutError",
        broken_invariants=("network.responsive",),
        observations={"rss_gb": 2.0, "latency_ms": 30000.0},
    ),
    "FAULT-INF-003": dict(
        subsystem="inference", kind="ValueError",
        broken_invariants=("inference.shape",),
        observations={"rss_gb": 3.0, "latency_ms": 120.0},
    ),
}


@pytest.fixture
def ontology():
    known = FailureOntology()
    for fault_id, spec in CATALOGUE.items():
        for index in range(MIN_INSTANCES + 1):
            known.observe(
                fault_id,
                Signature(
                    **{
                        **spec,
                        "observations": {
                            k: v * (1 + 0.02 * index)
                            for k, v in spec["observations"].items()
                        },
                    }
                ),
            )
    return known


NOVEL = Signature(
    subsystem="consciousness",
    kind="SilentDriftError",
    broken_invariants=("self.coherence.monotone",),
    observations={"rss_gb": 3.1, "latency_ms": 110.0},
)


# ── step 1, and its null ─────────────────────────────────────────────────


def test_a_repeat_of_a_known_failure_is_recognised(ontology):
    """The null. A recogniser that calls everything novel is useless."""
    repeat = Signature(
        subsystem="memory", kind="OutOfMemory",
        broken_invariants=("memory.bounded",),
        observations={"rss_gb": 14.3, "latency_ms": 920.0},
    )
    verdict = ontology.recognise(repeat)
    assert verdict.recognition is Recognition.KNOWN
    assert verdict.nearest == "FAULT-MEM-001"


def test_a_failure_outside_the_ontology_is_called_novel(ontology):
    verdict = ontology.recognise(NOVEL)
    assert verdict.recognition is Recognition.NOVEL
    assert verdict.needs_a_new_concept is True


def test_the_same_kind_of_error_somewhere_new_is_novel(ontology):
    """An OutOfMemory in storage is not the memory subsystem's fault mode."""
    elsewhere = Signature(
        subsystem="storage", kind="OutOfMemory",
        broken_invariants=("storage.bounded",),
        observations={"rss_gb": 13.9, "latency_ms": 880.0},
    )
    assert ontology.recognise(elsewhere).recognition is Recognition.NOVEL


def test_numeric_channels_are_compared_on_their_own_scale(ontology):
    """A latency in milliseconds otherwise swamps every other difference.

    On the first run a consciousness drift error came back as a known
    inference fault, because its distance of 2.5 looked small beside a
    typical separation of 5144 that was entirely made of latency.
    """
    verdict = ontology.recognise(NOVEL)
    assert verdict.distance is not None and verdict.distance <= 1.0


def test_too_small_an_ontology_cannot_say_not_any_of_these():
    known = FailureOntology()
    for index in range(MIN_INSTANCES + 1):
        known.observe("FAULT-A", Signature(subsystem="a", kind="X", observations={"v": 1.0}))
    verdict = known.recognise(Signature(subsystem="z", kind="Q"))
    assert verdict.recognition is Recognition.UNDECIDABLE
    assert str(MIN_KNOWN_FAULTS) in verdict.because


def test_one_instance_is_a_point_not_a_signature():
    known = FailureOntology()
    for fault_id in ("a", "b", "c", "d"):
        known.observe(fault_id, Signature(subsystem=fault_id, kind="X"))
    assert known.known_faults == ()
    assert known.recognise(Signature(subsystem="z", kind="Q")).recognition is (
        Recognition.UNDECIDABLE
    )


# ── steps 2 and 3 ────────────────────────────────────────────────────────


def test_a_known_failure_reports_the_invariant_its_instances_broke(ontology):
    repeat = Signature(
        subsystem="memory", kind="OutOfMemory",
        broken_invariants=("memory.bounded",),
        observations={"rss_gb": 14.1, "latency_ms": 905.0},
    )
    assert ontology.recognise(repeat).broken_invariant == "memory.bounded"


def test_a_novel_failure_reports_the_invariant_it_actually_broke(ontology):
    assert ontology.recognise(NOVEL).broken_invariant == "self.coherence.monotone"


def test_an_invariant_is_not_invented_for_a_failure_that_broke_none(ontology):
    """Guessing one gets the repair verified against something else."""
    quiet = Signature(subsystem="consciousness", kind="SilentDriftError")
    assert ontology.recognise(quiet).broken_invariant == ""


def test_the_locus_is_where_it_was_reported(ontology):
    assert ontology.recognise(NOVEL).locus == "consciousness"


# ── step 4 ───────────────────────────────────────────────────────────────


def test_repairs_are_offered_cheapest_and_least_irreversible_first(ontology):
    repairs = propose_repairs(ontology.recognise(NOVEL))
    costs = [r.irreversibility for r in repairs]
    assert costs == sorted(costs)
    assert costs[0] == 0.0


def test_every_repair_carries_the_invariant_it_has_to_restore(ontology):
    verdict = ontology.recognise(NOVEL)
    for repair in propose_repairs(verdict):
        assert repair.restores == verdict.broken_invariant


# ── step 5 ───────────────────────────────────────────────────────────────


def test_a_repair_needs_both_halves_to_count():
    repair = Repair(action="restart", rationale="r", restores="i")
    assert RepairOutcome(repair, invariant_holds=True, signature_recurred=False).worked
    assert not RepairOutcome(repair, invariant_holds=True, signature_recurred=True).worked
    assert not RepairOutcome(repair, invariant_holds=False, signature_recurred=False).worked


def test_an_invariant_that_holds_while_the_failure_recurs_means_the_wrong_invariant():
    repair = Repair(action="restart", rationale="r", restores="i")
    outcome = RepairOutcome(repair, invariant_holds=True, signature_recurred=True)
    assert outcome.worked is False


# ── step 6 ───────────────────────────────────────────────────────────────


def test_integrating_a_novel_failure_makes_it_known_next_time(ontology):
    assert ontology.recognise(NOVEL).recognition is Recognition.NOVEL
    fault_id = ontology.integrate("silent_drift", NOVEL)
    for _ in range(MIN_INSTANCES):
        ontology.observe(fault_id, NOVEL)
    assert ontology.recognise(NOVEL).recognition is Recognition.KNOWN
    assert fault_id in ontology.invented


def test_what_was_invented_is_kept_separate_from_what_was_catalogued(ontology):
    ontology.integrate("silent_drift", NOVEL)
    assert ontology.invented == ("FAULT-NOVEL-silent_drift",)
    assert "FAULT-MEM-001" not in ontology.invented


def test_the_snapshot_counts_what_has_a_signature(ontology):
    snapshot = ontology.snapshot()
    assert snapshot["faults_with_signatures"] == len(CATALOGUE)
    assert snapshot["instances"] == len(CATALOGUE) * (MIN_INSTANCES + 1)
