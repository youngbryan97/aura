"""Two arms of an intervention must not read two different machines.

The state's body readings are held for the duration of a trial, because what
the host is doing is the environment and not the organism, and letting it
drift between arms puts the machine's own load into the floor every edge has
to clear — measured once at ten units of temperature, larger than anything the
experiment was looking for.

Every layer that reads the machine through the shared observer went round that
hold. Embodied interoception samples it once a second of the organism's life,
straight from `psutil`, which made it a fourth body beside the proprioceptive
loop, the soma timer and the resilience engine. The hold now happens at the
observer's own seam, and interoception reads the shared observer.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_a_held_observer_answers_every_call_the_same_way():
    from core.subject.driver import _HeldObserver

    class _Machine:
        def __init__(self) -> None:
            self.reads = 0

        def compute(self):
            self.reads += 1
            return self.reads

        def memory(self, *, root_pid=None, include_process_tree=True):
            self.reads += 1
            return ("memory", self.reads)

    machine = _Machine()
    held = _HeldObserver(machine)
    assert held.compute() == held.compute() == held.compute()
    assert held.memory() == held.memory()
    assert machine.reads == 2, "one read of each shape, not one per call"


def test_the_hold_distinguishes_calls_by_their_arguments():
    from core.subject.driver import _HeldObserver

    class _Machine:
        def disk(self, path="/"):
            return f"disk:{path}"

    held = _HeldObserver(_Machine())
    assert held.disk("/") == "disk:/"
    assert held.disk("/tmp") == "disk:/tmp"


def test_freezing_the_host_holds_the_observer_and_thawing_releases_it(tmp_path):
    from core.runtime.resource_observation import get_resource_observer
    from core.subject.driver import _HeldObserver, build_runtime

    runtime = build_runtime(tmp_path / "runtime", seed=3)
    before = get_resource_observer()
    runtime.freeze_host()
    try:
        assert isinstance(get_resource_observer(), _HeldObserver)
    finally:
        runtime.thaw_host()
    assert get_resource_observer() is before


def test_interoception_reads_the_shared_observer_rather_than_psutil():
    """Read out of the source, because the defect was a direct call."""
    source = (ROOT / "core" / "consciousness" / "embodied_interoception.py").read_text()
    tree = ast.parse(source)
    sampler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_sample_hardware"
    )
    body = ast.get_source_segment(source, sampler) or ""
    assert "_resource_observer()" in body
    # The fallback stays: a caller that cannot reach the shared reading uses
    # its own rather than reporting a body of zeros.
    assert "psutil.cpu_percent" in body
    cpu_line = next(
        line for line in body.splitlines() if "observer.compute()" in line
    )
    assert "cpu_percent" in cpu_line


def test_every_priced_kind_of_work_has_something_that_reports_it():
    """A cost nothing writes makes the felt total wrong in two ways.

    `EffortLedger.exertion` sums each reported amount against its unit cost and
    divides by how many kinds are priced. `phases` and `tool_calls` were priced
    and never reported, so thinking through a turn and acting on the world both
    cost her nothing, and the divisor counted two channels that could not be
    written. A reader with no writer, in the module written to give her a sense
    of her own exertion.
    """
    import subprocess

    from core.soma.effort import UNIT_COST

    found = subprocess.run(
        ["grep", "-rn", "note_effort(", "--include=*.py", "core/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    reported = {
        line.split('note_effort("', 1)[1].split('"', 1)[0]
        for line in found.splitlines()
        if 'note_effort("' in line
    }
    missing = sorted(set(UNIT_COST) - reported)
    assert not missing, f"priced but never reported: {missing}"


def test_an_unremarkable_turn_does_not_read_as_maximum_exertion():
    """The aggregate contradicted its own calibration.

    Each unit cost is defined as the amount an unremarkable turn produces, so
    an unremarkable turn makes every ratio one and the mean one — and the mean
    was clipped at one. Measured across four rounds of the eight ordinary
    conditions, exertion's median came back at exactly 1.000 and its minimum at
    0.848: her sense of her own exertion was a constant pinned at the top, and
    it was the term that won the workspace competition on nineteen turns in
    twenty-four.
    """
    from core.soma.effort import UNIT_COST, EffortLedger

    unremarkable = dict(UNIT_COST)
    reading = EffortLedger.exertion(unremarkable)
    assert reading == pytest.approx(0.5), reading

    twice = {kind: value * 2.0 for kind, value in UNIT_COST.items()}
    assert EffortLedger.exertion(twice) > reading
    half = {kind: value * 0.5 for kind, value in UNIT_COST.items()}
    assert EffortLedger.exertion(half) < reading


def test_exertion_stays_inside_its_bounds_however_hard_the_turn():
    from core.soma.effort import UNIT_COST, EffortLedger

    assert EffortLedger.exertion({}) == pytest.approx(0.0)
    enormous = {kind: value * 10_000.0 for kind, value in UNIT_COST.items()}
    assert 0.0 <= EffortLedger.exertion(enormous) <= 1.0


def test_how_hard_she_may_think_reaches_the_state():
    """Three of four cognitive-modifier columns were constants for a whole life.

    `HomeostaticCoupling` blends the continuous substrate into felt state and
    computes from it how creative, how focused, how urgent and how vital the
    moment is. Those readings lived on the coupling object, where the
    capability engine, the phantom browser, time dilation and the heartbeat
    find them by reference — and `cognition.modifiers`, which every phase reads
    and the subject schema measures, carried three of the four as constants.

    A reading computed and kept out of the state is a reading the cognition
    cannot use.
    """
    import ast

    source = (ROOT / "core" / "phases" / "proprioceptive_loop.py").read_text()
    tree = ast.parse(source)
    publisher = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_publish_modifiers"
    )
    body = ast.get_source_segment(source, publisher) or ""
    for key in ("creativity_mod", "focus_mod", "overall_vitality", "urgency_flag"):
        assert key in body, key
    assert "cognition" in body and "modifiers" in body
