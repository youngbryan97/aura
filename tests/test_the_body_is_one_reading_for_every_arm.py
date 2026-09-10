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
