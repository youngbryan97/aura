"""The closure test finds a broker that keeps the system's memory outside K.

The hidden broker routes every path between the ten domains through eight
variables that are none of them and that remember nearly all of their own
past. The graph measures pass it: every domain reaches every other. Only
closure can say that K's future depends on something no reading of K contains,
so closure has to say it, and has to name the broker as the leak.
"""

from __future__ import annotations

from core.subject.closure import closure_gain
from core.subject.nulls import architecture, toy_periphery, toy_recording


def _report(name: str, *, seed: int = 5, steps: int = 1500, draws: int = 8):
    system = architecture(name, seed=seed)
    return closure_gain(
        toy_recording(system, steps=steps, seed=seed),
        toy_periphery(system, steps=steps, seed=seed),
        tuple(f"broker.{index}" for index in range(system.hub_width)),
        seed=seed,
        draws=draws,
    )


def test_a_hidden_broker_leaves_k_open() -> None:
    report = _report("hidden_broker")
    assert not report.closed
    assert report.leak > report.floor_high


def test_the_leak_is_named_as_the_broker() -> None:
    report = _report("hidden_broker")
    assert report.top_leaks, "an open core with no variable named as the leak"
    assert all(name.startswith("broker.") for name, _ in report.top_leaks[:3])


def test_the_broker_is_found_on_other_seeds() -> None:
    assert all(not _report("hidden_broker", seed=seed).closed for seed in (6, 7, 8))
