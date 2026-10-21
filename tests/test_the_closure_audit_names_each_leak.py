"""The closure test names what each leak is, and finds a cause planted outside K.

The pair of 21 October failed closure on leaks of three kinds: a learned weight
matrix of an organ K reads, a counter, and running medians of the phi
estimators. Those are K's blindness, a clock and a memory of K's own past, and
none is a cause from outside. The audit says which, and its positive control
shows the test can find a real outside cause when there is one.
"""
from __future__ import annotations

import numpy as np

from core.subject.closure import audit_leaks


def _world(rows: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    hidden = np.zeros(rows + 1)
    k = np.zeros((rows + 1, 3))
    for t in range(rows):
        hidden[t + 1] = 0.8 * hidden[t] + rng.normal(0, 1.0)
        k[t + 1] = 0.5 * k[t] + np.array([0.9, 0.0, 0.0]) * hidden[t] + rng.normal(0, 0.3, 3)
    # A summary of K's own past, and the hidden driver itself.
    memory = np.concatenate([[0.0], 0.5 * k[:-1, 0]])
    return k[:-1], k[1:], np.column_stack([memory[:-1], hidden[:-1]])


def test_a_summary_of_the_core_past_is_named_as_one_and_a_hidden_driver_is_not():
    now, nxt, outside = _world()
    names = ("PhiPhase._running_medians.p0", "service.weather.instance.pressure")
    audit = audit_leaks(now, nxt, outside, names, [(names[0], 0.01), (names[1], 0.2)], ())
    kinds = {row["variable"]: row["kind"] for row in audit["leaks"]}
    assert kinds[names[0]] == "a summary of the core's own past"
    assert kinds[names[1]] == "an outside cause or unread state"


def test_a_leak_from_an_organ_k_reads_is_named_as_k_not_taking_it_in():
    now, nxt, outside = _world(seed=1)
    names = ("service.memory_echo.instance.p0", "service.unified_field.instance._W_input_batched.p2")
    audit = audit_leaks(now, nxt, outside, names, [(names[1], 0.2)], ())
    assert audit["leaks"][0]["organ"] == "field"
    assert audit["leaks"][0]["kind"] == "state of an organ K reads, which K does not take in"


def test_the_planted_cause_is_found():
    now, nxt, outside = _world(seed=2)
    audit = audit_leaks(now, nxt, outside, ("a", "b"), [], ())
    assert audit["planted_cause"]["detected"] is True


def test_a_counter_is_named_as_a_clock():
    now, nxt, outside = _world(seed=3)
    audit = audit_leaks(now, nxt, outside, ("service.self_prediction.instance._confirmation_count", "b"),
                        [("service.self_prediction.instance._confirmation_count", 0.01)],
                        ("service.self_prediction.instance._confirmation_count",))
    assert audit["leaks"][0]["kind"] == "a clock, entered as its increment"
