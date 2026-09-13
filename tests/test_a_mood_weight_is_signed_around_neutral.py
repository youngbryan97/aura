"""Below neutral has to steer the other way.

Every mood dimension is a 0..1 activation — `_neutral_reference_state` says so
— and the weight read the raw value as though it were already signed. Neutral
weighed +0.46 and a low state pushed the SAME way as a high one, only less
hard, so the channel carried a constant offset whose magnitude wobbled rather
than her state.

The fusion certificate is what caught it: two opposing states moved the
distribution apart by 0.171 of how far either moved at all, against a floor of
0.5, and `carries_content` refused.
"""

from __future__ import annotations

import pytest

from core.consciousness.affective_steering import (
    AFFECTIVE_DIMENSIONS,
    _NEUTRAL_MOOD,
    _signed_weight,
)
from core.consciousness.fusion_probe import STATE_HIGH, STATE_LOW

_MOOD_OF = {
    "valence_positive": "valence",
    "arousal": "arousal",
    "curiosity": "motivation",
    "frustration": "stress",
    "energy": "energy",
}


@pytest.mark.parametrize("substrate_fn", ["tanh", "linear_half"])
def test_neutral_affect_steers_nowhere(substrate_fn: str) -> None:
    assert _signed_weight(_NEUTRAL_MOOD, substrate_fn) == 0.0


@pytest.mark.parametrize("substrate_fn", ["tanh", "linear_half"])
def test_either_side_of_neutral_steers_opposite_ways(substrate_fn: str) -> None:
    low = _signed_weight(0.1, substrate_fn)
    high = _signed_weight(0.9, substrate_fn)
    assert low < 0.0 < high
    assert low == pytest.approx(-high, abs=1e-6)


@pytest.mark.parametrize("dimension", AFFECTIVE_DIMENSIONS, ids=lambda d: d["key"])
def test_the_probe_states_are_opposite_on_every_dimension(dimension) -> None:
    """What `carries_content` needs: two states that are actually opposed.

    Both were positive on all five before this — frustration included, because
    a high-stress low state read as a weak push in the same direction as a
    contented one.
    """
    mood = _MOOD_OF[str(dimension["key"])]
    substrate_fn = str(dimension["substrate_fn"])
    high = _signed_weight(STATE_HIGH[mood], substrate_fn)
    low = _signed_weight(STATE_LOW[mood], substrate_fn)
    assert high * low < 0.0, f"{dimension['key']}: {high:+.4f} and {low:+.4f}"


def test_a_vector_reads_the_same_weight_from_moods_and_from_the_substrate() -> None:
    """Two entry points, one mapping. They fed the same channel and differed."""
    import numpy as np

    from core.consciousness.affective_steering import SteeringVector

    vector = SteeringVector(
        key="valence_positive",
        layer_idx=3,
        d_model=4,
        v=np.zeros(4, dtype=np.float32),
        substrate_idx=0,
        substrate_fn="tanh",
    )
    state = np.full(64, _NEUTRAL_MOOD, dtype=np.float32)
    state[0] = 0.9
    assert vector.compute_weight({"valence": 0.9}) == pytest.approx(
        vector.compute_weight_from_state(state), abs=1e-6
    )
