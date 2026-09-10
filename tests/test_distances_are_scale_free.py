"""The four places a distance measured whichever channel was largest.

Each of these was written carefully, passed its own tests, and was wrong the
same way: a distance or a step taken over quantities whose scales come from
callers, with no normalisation on the way. The result measures whichever
channel happens to be biggest and looks exactly like a working instrument.

    model_horizon        an absolute neighbour radius in a feature space whose
                         scale is a per-model fact
    unknown_failure      a latency in milliseconds swamping every categorical
                         difference
    manipulable_learning a fixed 0.2 step on a value of 2 that rounds to an
                         integer
    narrative_provenance a duration in seconds swamping two values in [0, 1]

I tried to write a detector for the class and it caught two of the four while
reporting four sites I had to read and dismiss. Tuning it further would have
been fitting it to these four examples, which is the failure the epistemic
independence work in this same pass exists to prevent. So this is a
regression test for the four rather than a gate for the class: precise about
what is known to have gone wrong, silent about what is not.
"""

from __future__ import annotations

import random

import pytest


def test_the_model_horizon_radius_is_relative_to_the_record():
    """An absolute radius made a query between two clusters well supported by
    cases that were nowhere near it."""
    from core.verify.model_horizon import ModelHorizon, Standing

    def cluster(model, centre, spread, accurate, n=40, seed=5):
        rng = random.Random(seed)
        for _ in range(n):
            point = [rng.uniform(centre - spread, centre + spread) for _ in range(2)]
            error = 0.05 if accurate else 0.9
            model.observe(point, 0.5, 0.5 + rng.uniform(-error, error))

    tight = ModelHorizon("tight")
    cluster(tight, 0.15, 0.15, True)
    cluster(tight, 0.85, 0.15, False, seed=6)

    # The same shape, a thousand times wider. A radius fixed in absolute units
    # would call everything a neighbour here.
    wide = ModelHorizon("wide")
    cluster(wide, 150.0, 150.0, True)
    cluster(wide, 850.0, 150.0, False, seed=6)

    assert tight.standing([0.15, 0.15]).standing is Standing.INSIDE
    assert wide.standing([150.0, 150.0]).standing is Standing.INSIDE
    assert tight.standing([0.5, 0.5]).standing is Standing.UNSUPPORTED
    assert wide.standing([500.0, 500.0]).standing is Standing.UNSUPPORTED


def test_a_failure_signature_is_not_decided_by_its_largest_channel():
    """A consciousness drift error came back as a known inference fault
    because a latency channel spanning 120-30000 swamped the rest."""
    from core.resilience.unknown_failure import (
        MIN_INSTANCES,
        FailureOntology,
        Recognition,
        Signature,
    )

    ontology = FailureOntology()
    catalogue = {
        "MEM": dict(subsystem="memory", kind="OutOfMemory",
                    broken_invariants=("memory.bounded",),
                    observations={"rss_gb": 14.0, "latency_ms": 900.0}),
        "NET": dict(subsystem="network", kind="TimeoutError",
                    broken_invariants=("network.responsive",),
                    observations={"rss_gb": 2.0, "latency_ms": 30000.0}),
        "INF": dict(subsystem="inference", kind="ValueError",
                    broken_invariants=("inference.shape",),
                    observations={"rss_gb": 3.0, "latency_ms": 120.0}),
    }
    for fault_id, spec in catalogue.items():
        for index in range(MIN_INSTANCES + 1):
            ontology.observe(fault_id, Signature(**{
                **spec,
                "observations": {
                    k: v * (1 + 0.02 * index) for k, v in spec["observations"].items()
                },
            }))
    novel = Signature(
        subsystem="consciousness", kind="SilentDriftError",
        broken_invariants=("self.coherence.monotone",),
        observations={"rss_gb": 3.1, "latency_ms": 110.0},
    )
    verdict = ontology.recognise(novel)
    assert verdict.recognition is Recognition.NOVEL
    assert verdict.distance <= 1.0, "the distance is not on a bounded scale"


def test_a_search_step_scales_with_the_parameter_it_moves():
    """A fixed step on a breadth of 2 that rounds to an integer landed back
    on 2 every time, and the meta-search reported that it had looked."""
    from core.learning.manipulable_learning import Program, mutate

    small = Program("p", moves=("m",), params={"x": 1.0})
    large = Program("p", moves=("m",), params={"x": 1000.0})
    small_moves = [
        abs(mutate(small, (), random.Random(s)).params["x"] - 1.0) for s in range(40)
    ]
    large_moves = [
        abs(mutate(large, (), random.Random(s)).params["x"] - 1000.0) for s in range(40)
    ]
    assert max(large_moves) > max(small_moves) * 10


def test_the_introspective_calibration_is_not_decided_by_a_clock():
    """A duration in seconds beside two values in [0, 1] took a generator
    whose words tracked valence perfectly from 0.90 to 0.03."""
    from core.consciousness.narrative_provenance import fidelity

    rng = random.Random(5)
    words = {
        "calm": "quiet settled still",
        "tense": "tight urgent pressing",
        "bright": "lit clear vivid",
        "heavy": "slow dim weighted",
    }
    pairs = []
    for index in range(16):
        valence, arousal = rng.choice([0.1, 0.9]), rng.choice([0.1, 0.9])
        key = (
            "calm" if valence < 0.5 and arousal < 0.5
            else "tense" if arousal > 0.5 and valence < 0.5
            else "bright" if arousal > 0.5
            else "heavy"
        )
        pairs.append((words[key], {
            "valence": valence, "arousal": arousal,
            "duration": index * 60.0, "moments_recorded": float(index),
        }))
    assert fidelity(pairs).informative is True
