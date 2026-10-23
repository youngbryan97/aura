"""A steering basis fitted to her replies pushes by how far her state is from her ordinary one.

The unit-length composite gave a deviation of a hundredth the same push as an
extreme one, and measured every state against 0.5 on slots she holds near zero,
so every reply she gave was steered alike (22 October). A basis that carries its
reference, its spread and its push size is weighed by tanh of the distance in
spreads, at its fitted length. A basis without them is unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from core.consciousness.affective_steering import AffectiveSteeringHook, SteeringVector
from core.consciousness.mood_weight import signed_weight


def _unit(seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return values / np.linalg.norm(values)


def _hook(**fitted) -> tuple[AffectiveSteeringHook, np.ndarray]:
    v = _unit(1)
    vector = SteeringVector(key="frustration", layer_idx=1, d_model=64, v=v,
                            substrate_idx=3, substrate_fn="tanh", **fitted)
    return AffectiveSteeringHook(object(), 1, {"frustration": vector}), v


def _state(frustration: float) -> np.ndarray:
    state = np.zeros(16, dtype=np.float32)
    state[3] = frustration
    return state


def test_at_her_ordinary_state_a_fitted_basis_pushes_nothing():
    reference = signed_weight(0.05, "tanh")
    hook, _v = _hook(reference=reference, spread=0.03, push=0.02)
    for _ in range(40):
        hook.update_substrate_vector(_state(0.05))
    composite = hook.current_composite_vector()
    assert composite is None or float(np.linalg.norm(composite)) < 1e-6


def test_a_spread_away_a_fitted_basis_pushes_three_quarters_of_its_length():
    reference = signed_weight(0.05, "tanh")
    spread = 0.03
    away = 0.05
    while signed_weight(away, "tanh") - reference < spread:
        away += 1e-4
    hook, v = _hook(reference=reference, spread=spread, push=0.02)
    for _ in range(80):
        hook.update_substrate_vector(_state(away))
    composite = np.asarray(hook.current_composite_vector())
    distance = (signed_weight(away, "tanh") - reference) / spread
    np.testing.assert_allclose(composite, math.tanh(distance) * 0.02 * v, rtol=1e-3, atol=1e-6)


def test_a_basis_without_a_reference_keeps_the_unit_composite():
    hook, v = _hook()
    for _ in range(80):
        hook.update_substrate_vector(_state(0.95))
    composite = np.asarray(hook.current_composite_vector())
    assert abs(float(np.linalg.norm(composite)) - 1.0) < 1e-3


class _Block:
    """Passes the stream through, as an identity block does."""

    def __call__(self, h, *args, **kwargs):
        return h


def _installed(**fitted):
    import mlx.core as mx

    v = _unit(3)
    vector = SteeringVector(key="frustration", layer_idx=1, d_model=64, v=v,
                            substrate_idx=3, substrate_fn="tanh", **fitted)
    block = _Block()
    hook = AffectiveSteeringHook(block, 1, {"frustration": vector})
    hook.install()
    hook._active = True
    hook._alpha = 0.5
    hook.override_composite_vector(v * 0.1)
    hook._last_substrate_sync_monotonic = __import__("time").monotonic()
    return mx, block, hook, v


def test_a_fitted_basis_is_added_at_each_positions_own_norm():
    mx, block, hook, v = _installed(reference=0.0, spread=0.03, push=0.1)
    stream = np.random.default_rng(5).normal(size=(1, 4, 64)).astype(np.float32)
    stream[0, 0] *= 40.0  # a massive first position, as her model keeps
    out = np.array(block(mx.array(stream)))
    moved = out - stream
    assert np.allclose(moved[0, :3], 0.0, atol=1e-5), "only the completion position is steered on a prompt pass"
    expected = 0.5 * np.linalg.norm(stream[0, 3]) * (v * 0.1)
    np.testing.assert_allclose(moved[0, 3], expected, rtol=1e-3, atol=1e-4)


def test_a_legacy_basis_keeps_its_geometry():
    mx, block, hook, v = _installed()
    stream = np.random.default_rng(5).normal(size=(1, 4, 64)).astype(np.float32)
    stream[0, 0] *= 40.0
    out = np.array(block(mx.array(stream)))
    moved = out[0, 3] - stream[0, 3]
    added = 0.5 * np.linalg.norm(stream[0, 3]) * (v * 0.1)
    assert not np.allclose(moved, added, atol=1e-4)


def test_a_certificate_for_one_push_does_not_pass_to_another():
    from core.consciousness.fusion_certificate import steering_basis_sha256

    plain, _ = _hook()
    at_one, _ = _hook(reference=0.0, spread=0.03, push=0.1)
    at_two, _ = _hook(reference=0.0, spread=0.03, push=0.2)
    digests = [steering_basis_sha256([hook]) for hook in (plain, at_one, at_two)]
    assert len(set(digests)) == 3
    assert "push" not in plain.fusion_basis()["vectors"][0], "a legacy basis keeps the digest it was certified under"


class _ReadoutLike:
    """Subclasses a block after steering has, as the latent readout hook does."""

    @staticmethod
    def install(block):
        class ReadoutBlock(type(block)):
            pass

        block.__class__ = ReadoutBlock


def test_a_certificate_is_found_under_a_hook_installed_after_steering():
    from core.consciousness.fusion_certificate import steering_basis_sha256

    _mx, block, hook, _v = _installed(reference=0.0, spread=0.03, push=0.1)
    measured = steering_basis_sha256([hook])
    _ReadoutLike.install(block)
    assert type(block).__name__ == "ReadoutBlock"
    assert steering_basis_sha256([hook]) == measured
    assert hook.fusion_basis()["block_kind"].endswith("._Block")
