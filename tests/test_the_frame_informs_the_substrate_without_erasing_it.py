"""A perceptual frame is an input to the substrate, not a replacement for it.

Two defects sat in `LiquidSubstrate.inject_perceptual_frame`, both from the
same mistaken belief that dimensions 0-15 were a free telemetry block.

The band starts at zero, and dimensions 0-6 are the named psychological state:
valence, arousal, dominance, frustration, curiosity, energy, focus. So every
proprioceptive tick added CPU load to her valence, thermal to her arousal,
memory to her dominance and CPU-plus-memory to her curiosity, with a sign
nobody chose and around every appraisal path the body already has.

And the frame was blended over the whole vector against a delta that was zero
everywhere the frame said nothing, so twenty numbers pulled five hundred and
twelve dimensions a quarter of the way to zero on every tick. The substrate's
recurrent state was erased by the input meant to inform it.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig


@pytest.fixture()
def substrate() -> LiquidSubstrate:
    return LiquidSubstrate(SubstrateConfig(neuron_count=128))


#: A frame with every channel loud, so nothing here passes by saying nothing.
LOUD = {
    "cpu_percent": 90.0,
    "memory_percent": 80.0,
    "thermal": 0.9,
    "valence": 0.4,
    "arousal": 0.6,
    "user_presence": 0.8,
    "voice_activity": 1.0,
    "screen_changed": 0.7,
    "social": 0.6,
    "threat": 0.5,
    "novelty": 0.3,
}


def test_the_frame_does_not_write_the_named_psychological_state(substrate) -> None:
    named = (
        substrate.idx_valence,
        substrate.idx_arousal,
        substrate.idx_dominance,
        substrate.idx_frustration,
        substrate.idx_curiosity,
        substrate.idx_energy,
        substrate.idx_focus,
    )
    with substrate.sync_lock:
        substrate.x[:] = 0.0
        for index in named:
            substrate.x[index] = 0.5
    substrate.inject_perceptual_frame(dict(LOUD))
    for index in named:
        assert substrate.x[index] == pytest.approx(0.5), index


def test_a_dimension_the_frame_says_nothing_about_keeps_its_value(substrate) -> None:
    """The recurrence lives in the dimensions no frame addresses."""
    with substrate.sync_lock:
        substrate.x[:] = 0.5
    before = substrate.x.copy()
    substrate.inject_perceptual_frame(dict(LOUD))
    moved = np.flatnonzero(np.abs(substrate.x - before) > 1e-12)
    assert moved.size <= 20, f"{moved.size} dimensions moved for a frame of twenty"
    untouched = np.setdiff1d(np.arange(substrate.x.size), moved)
    assert substrate.x[untouched] == pytest.approx(0.5)


def test_an_empty_frame_leaves_everything_outside_the_bands(substrate) -> None:
    """A frame that says nothing still addresses its bands, and nothing else.

    Absent readings default to zero, so the bands move toward zero — which is
    the frame reporting silence. What must not happen is the rest of the state
    moving with them.
    """
    with substrate.sync_lock:
        substrate.x[:] = 0.3
    before = substrate.x.copy()
    substrate.inject_perceptual_frame({})
    moved = set(np.flatnonzero(np.abs(substrate.x - before) > 1e-12).tolist())
    bands = set(range(substrate._TELEMETRY_BASE, substrate._TELEMETRY_BASE + 6))
    bands |= set(range(16, 20)) | set(range(32, 36)) | set(range(48, 52)) | {64, 65, 66}
    assert moved <= bands, sorted(moved - bands)
    assert substrate.x[100:] == pytest.approx(before[100:])


def test_the_frame_still_reaches_the_substrate(substrate) -> None:
    """Moving the band off the psych state must not disconnect it."""
    with substrate.sync_lock:
        substrate.x[:] = 0.0
    substrate.inject_perceptual_frame(dict(LOUD))
    band = substrate._TELEMETRY_BASE
    assert substrate.x[band] > 0.0, "system load reaches its own dimension"
    assert substrate.x[16] > 0.0, "the user's presence reaches its own dimension"
    assert substrate.x[32] > 0.0, "the screen reaches its own dimension"
    assert substrate.x[48] > 0.0, "audio reaches its own dimension"


def test_the_telemetry_band_clears_the_named_state(substrate) -> None:
    """Stated as a bound rather than a number, so widening one moves the other."""
    highest_named = max(
        substrate.idx_valence,
        substrate.idx_arousal,
        substrate.idx_dominance,
        substrate.idx_frustration,
        substrate.idx_curiosity,
        substrate.idx_energy,
        substrate.idx_focus,
    )
    assert substrate._TELEMETRY_BASE > highest_named
