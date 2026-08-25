"""A reader whose keys no writer publishes is a channel that cannot fire.

Measured on 1,729 live turns, 2026-08-25: the substrate channel reported 1 of
its 34 dimensions present and NONE of them varying. Two causes, both silent.

The probe resolved `continuous_substrate` or `liquid_state`. Neither name is
registered anywhere in this tree — the live container registers
`conscious_substrate` — so the lookup returned None on every turn and the
channel read absent, which is indistinguishable from a runtime that has no
substrate.

Then the accessor: it asked for `get_state_vector()`, which the live
LiquidSubstrate does not have. Its non-blocking accessor is
`_state_snapshot_nowait()`, whose "x" is the continuous state and whose
"snapshot_age_s"/"freshness_threshold_s" say whether that state still
describes now.

So both halves of this file check the reader against what the WRITER actually
publishes, rather than against a stub written from the same assumption that
caused the defect.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.brain.llm.endogenous_state import (
    _probe_substrate,
    _substrate_snapshot,
    _substrate_vector,
)


class _LiveShapedSubstrate:
    """The accessors and key names the real LiquidSubstrate publishes."""

    def __init__(self, *, age: float = 0.0, threshold: float = 5.0) -> None:
        self._age = age
        self._threshold = threshold

    def _state_snapshot_nowait(self, max_wait_s: float = 0.05) -> dict:
        return {
            "x": np.linspace(-1.0, 1.0, 64),
            "v": np.zeros(64),
            "phi": 0.42,
            "last_update": time.time(),
            "update_rate_hz": 8.0,
            "snapshot_age_s": self._age,
            "freshness_threshold_s": self._threshold,
            "coherence": 0.7,
            "em_field": 0.1,
        }

    def current(self):
        class _Vector:
            frustration = 0.1
            curiosity = 0.6
            energy = 0.83
            focus = 0.5

        return _Vector()


def test_the_registered_service_name_is_the_one_the_probe_asks_for():
    """The whole first half of the defect, stated as a name check."""
    import inspect

    from core.brain.llm import endogenous_state

    source = inspect.getsource(endogenous_state._probe_substrate)
    assert "conscious_substrate" in source, (
        "the live container registers conscious_substrate; a probe that does "
        "not ask for it reads absent forever"
    )


def test_the_state_vector_comes_back_from_the_live_accessor():
    substrate = _LiveShapedSubstrate()
    vector = _substrate_vector(substrate)
    assert vector is not None
    assert len(vector) == 64


def test_a_stale_snapshot_reads_absent_rather_than_stale():
    """A vector from an earlier moment would be fitted against words it did
    not precede, which is worse than having no vector."""
    fresh = _LiveShapedSubstrate(age=1.0, threshold=5.0)
    stale = _LiveShapedSubstrate(age=9.0, threshold=5.0)

    assert _substrate_snapshot(fresh) is not None
    assert _substrate_snapshot(stale) is None
    assert _substrate_vector(stale) is None


def test_the_probe_fills_the_bands_energy_and_phi(monkeypatch):
    substrate = _LiveShapedSubstrate()
    monkeypatch.setattr(
        "core.brain.llm.endogenous_state._service",
        lambda name: substrate if name == "conscious_substrate" else None,
    )

    read = _probe_substrate()

    assert read is not None
    bands = {key for key in read if key.startswith("substrate.band_")}
    assert len(bands) >= 12, sorted(bands)
    assert read["substrate.phi"] == pytest.approx(0.42)
    assert read["substrate.energy"] == pytest.approx(0.83)


def test_no_substrate_still_reads_absent(monkeypatch):
    monkeypatch.setattr(
        "core.brain.llm.endogenous_state._service", lambda name: None
    )
    assert _probe_substrate() is None
