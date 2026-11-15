"""The steering hooks read her felt state, not one neuron that is a term of it.

`HomeostaticCoupling` calls the continuous substrate the ground truth for her
felt state and blends it into affect at `SUBSTRATE_SHARE`; the affect phase
pushes the fused result back down once per turn. The channel that carries her
state to the hooks in her cortex published the raw neuron, and the neuron's own
dynamics are wider within a turn than anything the appraisal puts into it.

Measured on the stub organism, 28 September: three arms whose fused valence was
+0.456, +0.143 and +0.428 published neuron activations of 0.6154, 0.6077 and
0.6122 — the manipulation reached the hooks at eight thousandths of the spread
it had in her.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness import steering_channel


class _Substrate:
    """Enough of the substrate for the channel: a state vector and its two indices."""

    idx_valence = 0
    idx_arousal = 1
    idx_frustration = 2

    def __init__(self, x: list[float], *, snapshot_age_s: float = 0.0) -> None:
        self._x = np.array(x, dtype=np.float64)
        self._age = snapshot_age_s

    def _state_snapshot_nowait(self) -> dict[str, object]:
        return {"x": self._x}

    def get_state_summary_nowait(self) -> dict[str, float]:
        return {"snapshot_age_s": self._age}


@pytest.fixture(autouse=True)
def _forget_the_felt_state():
    steering_channel._FELT.clear()
    yield
    steering_channel._FELT.clear()


def test_without_a_felt_reading_the_neuron_is_published() -> None:
    substrate = _Substrate([0.4, 0.3, 0.1, 0.9])
    state = steering_channel.activation_state(substrate)
    assert state[0] == pytest.approx(0.7)  # (0.4 + 1) / 2
    assert state[1] == pytest.approx(0.65)


def test_the_felt_reading_is_what_reaches_the_hooks() -> None:
    substrate = _Substrate([0.4, 0.3, 0.1, 0.9])
    steering_channel.note_felt(-0.5, 0.8)
    state = steering_channel.activation_state(substrate)
    assert state[0] == pytest.approx(0.25)  # (-0.5 + 1) / 2
    # Her arousal is already an activation where the neuron's is signed.
    assert state[1] == pytest.approx(0.8)
    # Everything else is the substrate's, untouched.
    assert state[3] == pytest.approx(0.9)


def test_a_displacement_of_her_feeling_reaches_the_hooks_at_its_own_size() -> None:
    """The reading that failed: the arms differed in her by 0.31 and at the hooks by 0.008."""
    substrate = _Substrate([0.239, 0.3, 0.1])
    published = []
    for valence in (0.456, 0.143, 0.428):
        steering_channel.note_felt(valence, 0.5)
        published.append(float(steering_channel.activation_state(substrate)[0]))
    spread = max(published) - min(published)
    assert spread == pytest.approx((0.456 - 0.143) / 2.0, abs=1e-6), published
    assert spread > 0.1, published


def test_a_feeling_is_hers_until_her_affect_phase_fuses_a_newer_one() -> None:
    """It does not expire: her feeling is not cancelled by nobody asking her anything."""
    substrate = _Substrate([0.4, 0.3, 0.1])
    steering_channel.note_felt(-0.9, 0.1)
    assert steering_channel.activation_state(substrate)[0] == pytest.approx(0.05)
    steering_channel.note_felt(0.6, 0.1)
    assert steering_channel.activation_state(substrate)[0] == pytest.approx(0.8)


def test_before_her_affect_phase_has_run_the_neuron_is_still_what_steers_her() -> None:
    substrate = _Substrate([0.4, 0.3, 0.1])
    assert steering_channel.activation_state(substrate)[0] == pytest.approx(0.7)


def test_the_governor_reads_the_arousal_the_hooks_read() -> None:
    substrate = _Substrate([0.4, 0.3, 0.1])
    steering_channel.note_felt(0.0, 0.77)
    state = steering_channel.activation_state(substrate)
    arousal, coherence = steering_channel.governor_inputs(state, {})
    assert arousal == pytest.approx(0.77)
    assert coherence == pytest.approx(1.0)


def test_the_affect_phase_is_what_notes_it() -> None:
    """What the affect phase fuses is what the steering hooks then read: valence, arousal and the felt reading."""
    from core.phases.affect_update import VALENCE_FELT, AffectUpdatePhase
    from core.state.aura_state import AffectVector, AuraState

    steering_channel._FELT.clear()
    affect = AffectVector(valence=0.37, arousal=0.81)
    affect.markers = {VALENCE_FELT: 0.42}
    state = AuraState.default()
    AffectUpdatePhase(None)._release_chemistry(affect, state)
    assert steering_channel._FELT == pytest.approx({"valence": 0.37, "arousal": 0.81, "felt": 0.42})
    assert state.response_modifiers["felt_state_steering_valence"] == pytest.approx(0.37)
