"""Her soma read the machine the sweep was loading, and re-read it every turn.

A paired arm forks one anchor twice and compares where the two rollouts end, so
the machine underneath has to be held still or the comparison is between two
machines. `freeze_host` does that, and `turn_once` calls it per turn and thaws it
again at the turn's end — which holds the host within a turn and re-reads it for
the next one. Over a sweep that is the machine's load, turn by turn.

The v25 carrier sweep never declared a host of its own. MEASURED on a sixty-round
baseline of 25 October: sixty "THERMAL RESONANCE: Hardware is overheating.
Emergency throttling." records and thirteen metabolic-pressure mitigations,
inside the run, caused by the run. Thermal strain multiplies her depth by 0.4,
her creativity by 0.5 and her temperature by 0.7, so two arms minutes apart were
thinking at different depths. The late untouched fork of the seed-23 campaign
separated from the cached one at a lower bound of 0.011444 against a weakest cut
of 0.00126, and both partition criteria failed on that control.

One `freeze_host` before the anchors holds all of it: the hardware and latency
dicts, and every answer of the shared resource observer — which is where the
thermal probe, the process memory and the cpu reading all come from.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _Machine:
    """An observer whose answers change every time it is asked."""

    def __init__(self) -> None:
        self.asked = 0
        self.provenance = "test"

    def thermal(self):
        self.asked += 1
        return type("R", (), {"level": self.asked, "provider": "test"})()

    def memory(self):
        self.asked += 1
        return type("M", (), {"process_rss_bytes": self.asked * 1_000_000})()


def test_the_hold_covers_the_thermal_probe_and_the_process_memory():
    """The two inputs that bypassed the hardware dict go through the observer."""
    from core.subject.snapshot import _HeldObserver

    machine = _Machine()
    held = _HeldObserver(machine)
    first = held.thermal().level
    assert held.thermal().level == first, "a second reading moved under the hold"
    assert held.thermal().level == first
    rss = held.memory().process_rss_bytes
    assert held.memory().process_rss_bytes == rss
    assert machine.asked == 2, "the real machine was asked once per shape, not per call"


def test_the_sweep_declares_a_host_before_it_takes_any_anchor():
    """Everything after the anchors forks paired arms off them."""
    import inspect

    from tools import run_subject_core_v25 as sweep

    body = inspect.getsource(sweep)
    assert "_hold_the_machine_still(runtime)" in body
    declared = body.index("_hold_the_machine_still(runtime)")
    collected = body.index("collecting {args.anchors} anchors")
    assert declared < collected, "the host was declared after the anchors were taken"


def test_a_runtime_that_cannot_declare_one_says_so(monkeypatch):
    from tools import run_subject_core_v25 as sweep

    said: list[tuple] = []
    monkeypatch.setattr(
        "core.runtime.errors.record_degradation",
        lambda *a, **k: said.append((a, k)),
    )

    class _NoFreeze:
        pass

    assert sweep._hold_the_machine_still(_NoFreeze()) is None
    assert said == [], "a runtime with no freeze at all is not a degradation"

    class _Raises:
        def freeze_host(self):
            raise RuntimeError("no soma")

    assert sweep._hold_the_machine_still(_Raises()) is None
    assert said, "a freeze that failed has to be recorded"
    assert "real machine" in said[0][1].get("action", "")


def test_the_declared_reading_comes_back_for_the_record():
    from tools import run_subject_core_v25 as sweep

    class _Runtime:
        def freeze_host(self):
            return {"cpu_usage": 41.0, "ram_usage": 68.0, "temperature": 52.0}

    assert sweep._hold_the_machine_still(_Runtime())["cpu_usage"] == 41.0
