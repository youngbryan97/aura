"""A gate that measures something weaker than it declares.

The whole protocol rests on writing pass conditions down before running
anything, and the reason is this: gate one declared human-range accuracy AND
no confident wrong answers, checked only the share, and five wrong answers in
two hundred passed a condition written to exclude them. Reading the number
would never have found it. Reading the declaration against the code did.

So the declarations are checked against the implementations here, for every
gate that runs.
"""

from __future__ import annotations

import inspect

from tools.agi_gauntlet.gates import THE_GATES


def test_every_gate_declares_all_four_things():
    for gate in THE_GATES:
        assert gate.measures, f"{gate.name} does not say what it measures"
        assert gate.passes_when, f"{gate.name} does not say what passing is"
        assert gate.control, f"{gate.name} has no control"
        assert gate.runnable or gate.if_not_here, (
            f"{gate.name} neither runs nor says what it would take"
        )


def test_a_gate_that_names_a_refusal_checks_for_one():
    """Gate one's condition is about calibration and its code has to be."""

    gate = next(one for one in THE_GATES if one.number == 1)
    said = gate.passes_when.lower()
    assert "refus" in said
    source = inspect.getsource(gate.run)
    assert "refused" in source
    assert "wrong" in source


def test_every_runnable_gate_returns_a_verdict_and_its_evidence():
    from tools.agi_gauntlet.protocol import take_the_freeze

    freeze = take_the_freeze()
    small = {
        "instances": 6, "worlds": 3, "trajectories": 3, "episodes": 2,
        "pairs": 8, "questions": 3, "lives": 3,
    }
    for gate in THE_GATES:
        if not gate.runnable:
            continue
        found = gate.run(freeze, small)
        assert "passed" in found, f"{gate.name} returns no verdict"
        assert isinstance(found["passed"], bool)
        assert "trajectories" in found, f"{gate.name} returns no evidence"


def test_a_gate_that_cannot_run_here_returns_no_number():
    """A harness that substitutes a proxy for the thing it names is how a
    system gets credited with a capability nobody measured."""

    from tools.agi_gauntlet.gates import run_a_gate
    from tools.agi_gauntlet.protocol import take_the_freeze

    freeze = take_the_freeze()
    for gate in THE_GATES:
        if gate.runnable:
            continue
        receipt = run_a_gate(gate, freeze, {})
        assert receipt.ran is False
        assert receipt.passed is None
        assert receipt.measurements == {}
        assert receipt.why_not


def test_a_gate_that_raises_has_not_passed():
    from tools.agi_gauntlet.gates import Gate, run_a_gate
    from tools.agi_gauntlet.protocol import take_the_freeze

    def blows_up(_freeze, _options):
        raise RuntimeError("no")

    receipt = run_a_gate(
        Gate(0, "one that raises", "x", "y", "z", run=blows_up),
        take_the_freeze(),
        {},
    )
    assert receipt.passed is False
    assert "RuntimeError" in receipt.why_not
