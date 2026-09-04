"""A channel that reads absent forever is indistinguishable from one that
does not exist, and nothing was measuring the difference.

Measured on 1,729 live turns, 2026-08-25: of 74 named dimensions, 47 were
never present and 9 more were pinned at one value. Coverage reported 0.365,
which was kinder than the truth, because a constant dimension pads it while
carrying nothing and a duplicated one is counted twice.

Every cause was the same shape — a reader naming an organ or a key that no
writer publishes:

    substrate    -> continuous_substrate / liquid_state, registered nowhere;
                    and get_state_vector(), which the live organ lacks
    uncertainty  -> calibration_tracker, confidence_calibrator, epistemics,
                    uncertainty_engine, none registered
    self-state   -> ghost, soul, watchdog, none registered
    memory       -> semantic_density, contradiction_rate; the facade
                    publishes which stores exist and last_commit
    affect       -> curiosity through get_state_summary, which this build
                    does not have, while `current` holds it

None of it failed loudly. Each probe is fail-open by design, which is right —
a turn that cannot read an organ should still generate — and it means the only
thing that can catch a dead channel is a test that drives the probes with
organs shaped like the real ones and counts what comes back.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from core.brain.llm.endogenous_state import FEATURES, assemble_state

#: Dimensions the runtime genuinely cannot supply from a bare organ set, with
#: the reason. Every other declared dimension must be reachable. This list may
#: only shrink: an entry is a dimension nothing publishes yet, not a licence.
KNOWN_UNREACHABLE = {
    "attention.novelty": "needs a history of the salient item across turns",
    "recurrence.convergence": "belongs to a running recurrent turn",
    "recurrence.delta": "belongs to a running recurrent turn",
}


class _Substrate:
    """`current` is a PROPERTY on the live organ, and the snapshot is the
    non-blocking accessor. Both shapes are what this stub exists to hold."""

    def __init__(self, step: int) -> None:
        self._step = step

    @property
    def current(self):
        step = self._step

        class _Axes:
            frustration = 0.10 + 0.01 * step
            curiosity = 0.30 + 0.02 * step
            energy = 0.50 + 0.01 * step
            focus = 0.20 + 0.03 * step

        return _Axes()

    def _state_snapshot_nowait(self, max_wait_s: float = 0.05) -> dict:
        rng = np.random.default_rng(self._step)
        return {
            "x": rng.normal(size=64),
            "phi": 0.20 + 0.01 * self._step,
            "snapshot_age_s": 0.0,
            "freshness_threshold_s": 5.0,
        }


class _HealthMonitor:
    max_consecutive_errors = 5

    def __init__(self, step: int) -> None:
        self.consecutive_errors = step % 4
        self.healthy = True


class _MemoryFacade:
    def __init__(self, step: int) -> None:
        self._step = step

    def get_status(self) -> dict:
        stamp = datetime.now() - timedelta(seconds=10 * self._step + 1)
        return {"episodic": True, "last_commit": stamp.isoformat()}


@pytest.fixture
def organ_set(monkeypatch):
    """Assemble the state repeatedly against organs that move."""

    def _assemble(count: int = 12):
        rows = []
        for step in range(count):
            organs = {
                "conscious_substrate": _Substrate(step),
                "liquid_state": _Substrate(step),
                "health_monitor": _HealthMonitor(step),
                "memory_facade": _MemoryFacade(step),
            }
            monkeypatch.setattr(
                "core.brain.llm.endogenous_state._service",
                lambda name, d=organs: d.get(name),
            )
            state = assemble_state()
            rows.append((state.values.copy(), state.present.copy()))
        return (
            np.stack([row[0] for row in rows]),
            np.stack([row[1] for row in rows]),
        )

    return _assemble


def test_the_substrate_channel_is_not_one_dimension_of_thirty_four(organ_set):
    """It read 1 of 34 present and 0 varying on every live turn."""
    values, present = organ_set()
    bands = [
        index
        for index, feature in enumerate(FEATURES)
        if feature.name.startswith("substrate.band_")
    ]

    alive = [index for index in bands if present[:, index].all()]
    moving = [index for index in alive if values[present[:, index], index].std() > 0]

    assert len(alive) >= 12, f"only {len(alive)} of {len(bands)} bands present"
    assert len(moving) >= 12, f"only {len(moving)} bands carried any variance"


def test_the_dimensions_that_had_no_reachable_source_now_have_one(organ_set):
    """Each of these read absent on all 1,729 recorded turns."""
    values, present = organ_set()
    index = {feature.name: position for position, feature in enumerate(FEATURES)}

    for name in (
        "affect.curiosity",
        "memory.episodic_recency",
        "self.integrity",
        "substrate.energy",
        "substrate.phi",
    ):
        assert present[:, index[name]].all(), f"{name} is still unreachable"


def test_a_bare_organ_set_reaches_most_of_the_declared_state(organ_set):
    """Coverage from organs alone, before any of the richer ones are up."""
    _values, present = organ_set()
    reachable = sum(1 for i in range(len(FEATURES)) if present[:, i].all())

    assert reachable >= 45, (
        f"only {reachable} of {len(FEATURES)} dimensions are reachable from a "
        "bare organ set; a channel has gone silent again"
    )


def test_the_unreachable_list_names_a_reason_for_every_entry():
    """It may only shrink, and an entry is a gap rather than a licence."""
    names = {feature.name for feature in FEATURES}
    for name, reason in KNOWN_UNREACHABLE.items():
        assert name in names, f"{name} is not a declared dimension"
        assert reason.strip(), name
    assert len(KNOWN_UNREACHABLE) <= 3


def test_the_bands_track_the_real_organ_not_a_randomised_stub():
    """The stub above varies every band on every read. The real one does not.

    `_Substrate` returns `rng.normal(size=64)` seeded per step, so every band
    moves and the liveness assertions above pass on plumbing alone. Measured
    against a real LiquidSubstrate on 2026-09-02: 34 dimensions present and 32
    distinct band values — the probe reads it correctly — but only the band
    holding a pushed axis moves, because the 64-neuron reservoir is integrated
    by a 20Hz loop that is not running in a test process.

    So the honest claim is what this checks: the probe RECOVERS the organ's
    state and TRACKS a change to it. How many bands vary per turn live is a
    property of the substrate's own loop, not of this probe, and asserting it
    here against a stub would be measuring the stub.
    """
    import asyncio

    from core.consciousness.liquid_substrate import LiquidSubstrate
    from core.brain.llm import endogenous_state as module

    real = LiquidSubstrate()

    def _only_real(name):
        return real if name in ("conscious_substrate", "liquid_state") else None

    original = module._service
    module._service = _only_real
    try:
        first = module._probe_substrate() or {}
        bands = {k: v for k, v in first.items() if k.startswith("substrate.band_")}
        assert len(bands) == 32, f"only {len(bands)} bands recovered from the real organ"
        assert len(set(round(v, 9) for v in bands.values())) > 16, (
            "the bands carry one repeated number; the pooling has collapsed"
        )

        asyncio.run(real.update(delta_curiosity=0.4))
        second = module._probe_substrate() or {}
        moved = [
            k
            for k in bands
            if abs(second.get(k, 0.0) - bands[k]) > 1e-9
        ]
        assert moved, (
            "pushing a named axis of the substrate moved no band; the probe is "
            "reading something that is not the live state"
        )
    finally:
        module._service = original
