"""The gate's memory reading goes stale on the machine's clock, not on hers.

Under a run's experiment clock the gate's module reads her monotonic, which a
restore rewinds. On seed 97's report run, 29 October, the reading taken when
her cortex had just been stopped (21.2 GB free) was served for every later
admission, and her cortex was refused a reload while the machine had 32 GB.
"""

from __future__ import annotations

import time

import pytest

from core.brain import inference_gate_cortex_warmup as module
from core.brain.inference_gate import InferenceGate
from core.subject.clock import ExperimentClock

pytestmark = pytest.mark.unit


@pytest.fixture
def machine(monkeypatch):
    box = {"free": 21.2}
    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: box["free"])
    monkeypatch.setattr(InferenceGate, "_virtual_memory_memo", None)
    monkeypatch.setattr(InferenceGate, "_VIRTUAL_MEMORY_MEMO_TTL_S", 0.05)
    return box


def test_a_rewound_clock_does_not_keep_an_old_reading(machine) -> None:
    clock = ExperimentClock(1.0 / 33.0, start=1_000_000.0)
    clock.install()
    try:
        assert InferenceGate._recent_virtual_memory() == 21.2
        machine["free"] = 32.1
        # A restore puts her clock back before the reading was stamped.
        clock.set(999_000.0)
        time.sleep(0.1)
        # As it was: 21.2, for as long as her clock stayed behind the stamp.
        assert InferenceGate._recent_virtual_memory() == 32.1
    finally:
        clock.uninstall()


def test_a_reading_is_still_shared_inside_its_window(machine) -> None:
    InferenceGate._VIRTUAL_MEMORY_MEMO_TTL_S = 60.0
    assert InferenceGate._recent_virtual_memory() == 21.2
    machine["free"] = 32.1
    assert InferenceGate._recent_virtual_memory() == 21.2
