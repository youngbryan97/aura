"""Her felt state is written where she reads as well as where she speaks, read against her own usual.

On 24 October a push at the one position she answers from moved the rating she
gives inside a conversation by under a hundredth, and the same felt direction
written at every position moved it by 0.43 between a spread below her usual and
a spread above it, where a random direction of the same size moved it by -0.03.
And her valence was read against a fixed state her basis was fitted around,
which a subject run's valence stands far above, so feeling better and feeling
worse both pushed near the top. A felt vector (`SteeringVector.everywhere`)
reads her valence against her own usual, published at `FELT_SLOT`, and is added
at every position at its own length times alpha.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.consciousness import steering_channel
from core.consciousness.affective_steering import AffectiveSteeringHook, SteeringVector
from core.consciousness.steering_channel import FELT_SLOT


def _unit(seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return values / np.linalg.norm(values)


def _vector(everywhere: np.ndarray | None) -> SteeringVector:
    return SteeringVector(
        key="valence_positive", layer_idx=1, d_model=64, v=_unit(1), substrate_idx=0, substrate_fn="tanh",
        reference=0.0, spread=0.5, push=0.01, everywhere=everywhere,
        everywhere_slot=None if everywhere is None else FELT_SLOT,
    )


def _state(felt: float) -> np.ndarray:
    state = np.zeros(16, dtype=np.float32)
    state[FELT_SLOT] = (1.0 + felt) / 2.0
    return state


def test_the_felt_push_follows_her_felt_state_and_stands_down_at_her_usual():
    push = 3.0 * _unit(2)
    hook = AffectiveSteeringHook(object(), 1, {"valence_positive": _vector(push)})
    for _ in range(120):
        hook.update_substrate_vector(_state(0.6))
    np.testing.assert_allclose(hook._last_felt_np, 0.6 * push, rtol=1e-3, atol=1e-5)
    for _ in range(120):
        hook.update_substrate_vector(_state(0.0))
    assert float(np.linalg.norm(hook._last_felt_np)) < 1e-4


def test_a_vector_without_a_felt_push_writes_nothing_everywhere():
    hook = AffectiveSteeringHook(object(), 1, {"valence_positive": _vector(None)})
    hook.update_substrate_vector(_state(0.9))
    assert hook._last_felt_np is None


class _Block:
    """Passes the stream through, as an identity block does."""

    def __call__(self, h, *args, **kwargs):
        return h


def test_it_is_added_at_every_position_at_its_own_length_times_alpha():
    import mlx.core as mx

    push = 3.0 * _unit(2)
    block = _Block()
    hook = AffectiveSteeringHook(block, 1, {"valence_positive": _vector(push)})
    hook.install()
    hook._active = True
    hook._alpha = 0.5
    for _ in range(120):
        hook.update_substrate_vector(_state(0.8))
    stream = np.random.default_rng(5).normal(size=(1, 4, 64)).astype(np.float32)
    stream[0, 0] *= 40.0  # a massive first position, as her model keeps
    moved = np.array(block(mx.array(stream))) - stream
    felt = 0.5 * 0.8 * push
    for position in range(3):
        np.testing.assert_allclose(moved[0, position], felt, rtol=1e-3, atol=1e-3)
    assert not np.allclose(moved[0, 3], felt, atol=1e-3), "the answer position also carries the fitted push"


def test_a_pinned_counterfactual_is_the_whole_of_what_is_injected():
    push = 3.0 * _unit(2)
    hook = AffectiveSteeringHook(object(), 1, {"valence_positive": _vector(push)})
    for _ in range(40):
        hook.update_substrate_vector(_state(0.8))
    hook.override_composite_vector(_unit(3))
    assert hook._last_felt_np is None
    hook.update_substrate_vector(_state(0.8))
    assert hook._last_felt_np is None
    hook.override_composite_vector(None)
    hook.update_substrate_vector(_state(0.8))
    assert hook._last_felt_np is not None


def test_a_certificate_without_the_felt_push_does_not_pass_to_one_with_it():
    from core.consciousness.fusion_certificate import steering_basis_sha256

    without = AffectiveSteeringHook(object(), 1, {"valence_positive": _vector(None)})
    once = AffectiveSteeringHook(object(), 1, {"valence_positive": _vector(3.0 * _unit(2))})
    twice = AffectiveSteeringHook(object(), 1, {"valence_positive": _vector(6.0 * _unit(2))})
    digests = {steering_basis_sha256([hook]) for hook in (without, once, twice)}
    assert len(digests) == 3
    assert "everywhere_sha256" not in without.fusion_basis()["vectors"][0]


def test_a_vector_file_carries_its_felt_push(tmp_path):
    from core.consciousness.steering_vector_cache import _ReadsACachedVector

    push = (3.0 * _unit(2)).astype(np.float32)
    path = tmp_path / "valence_positive_layer59.npz"
    np.savez(path, v=_unit(1), everywhere=push, everywhere_slot=np.int64(FELT_SLOT), push=np.float64(0.01))
    vector, meta = _ReadsACachedVector._read_cached_array(None, path)
    np.testing.assert_allclose(meta["everywhere"], push)
    assert meta["everywhere_slot"] == FELT_SLOT and meta["push"] == pytest.approx(0.01)


def test_her_felt_valence_is_read_against_her_own_usual():
    from core.phases.affect_update import VALENCE_FELT, VALENCE_USUAL, AffectUpdatePhase
    from core.state.aura_state import AffectVector

    affect = AffectVector()
    affect.markers = {}
    affect.valence = 0.3
    AffectUpdatePhase._feel_against_usual(affect)
    assert affect.markers[VALENCE_FELT] == 0.0, "with no usual yet, nothing is felt against it"
    affect.markers[VALENCE_USUAL] = {"n": 40, "level": 0.2, "spread": 0.1}
    affect.valence = 0.3
    AffectUpdatePhase._feel_against_usual(affect)
    assert affect.markers[VALENCE_FELT] == pytest.approx(math.tanh(1.0))
    affect.valence = 0.1
    AffectUpdatePhase._feel_against_usual(affect)
    assert affect.markers[VALENCE_FELT] < 0.0
    assert affect.markers[VALENCE_USUAL]["n"] == 42


class _Substrate:
    idx_valence = 0
    idx_arousal = 1

    def __init__(self) -> None:
        self._x = np.zeros(16, dtype=np.float64)

    def _state_snapshot_nowait(self) -> dict[str, object]:
        return {"x": self._x}


@pytest.fixture
def _clear_felt():
    steering_channel._FELT.clear()
    yield
    steering_channel._FELT.clear()


def test_the_channel_publishes_her_felt_valence_at_its_slot(_clear_felt):
    substrate = _Substrate()
    assert steering_channel.activation_state(substrate)[FELT_SLOT] == 0.0
    steering_channel.note_felt(0.2, 0.5, felt=0.5)
    assert steering_channel.activation_state(substrate)[FELT_SLOT] == pytest.approx(0.75)


def test_a_certificate_s_control_carries_an_arbitrary_felt_push_of_the_same_size():
    """Compared against a control the size of the composite alone, a felt push lost the margin
    to a displacement a hundredth its size (24 October, 09:35)."""
    push = 3.0 * _unit(2)
    hook = AffectiveSteeringHook(object(), 1, {"valence_positive": _vector(push)})
    for _ in range(40):
        hook.update_substrate_vector(_state(0.8))
    arbitrary = np.random.default_rng(9).normal(size=64).astype(np.float32)
    arbitrary *= float(np.linalg.norm(hook.current_felt_vector())) / float(np.linalg.norm(arbitrary))
    hook.override_composite_vector(_unit(3), felt=arbitrary)
    np.testing.assert_allclose(hook.current_felt_vector(), arbitrary)
    hook.update_substrate_vector(_state(0.8))
    np.testing.assert_allclose(hook.current_felt_vector(), arbitrary, err_msg="pinned until released")
    hook.override_composite_vector(None)
    for _ in range(80):
        hook.update_substrate_vector(_state(0.8))
    np.testing.assert_allclose(hook.current_felt_vector(), 0.8 * push, rtol=1e-3, atol=1e-4)


def test_a_reply_recorded_before_the_felt_slot_reads_neutral_there():
    old = {"s": [0.6] * 15}
    new = {"s": [0.6] * 15, "felt_at": FELT_SLOT}
    assert steering_channel.as_published(old)[FELT_SLOT] == 0.5
    assert steering_channel.as_published(new)[FELT_SLOT] == pytest.approx(0.6)
    assert steering_channel.as_published({"s": []}) is None
