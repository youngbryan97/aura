"""A nudge that turns the stream must not also lengthen it.

The reason these exist: sixteen layers of `h + 0.2*s*v` on the 27B returned
eight characters. The vector was right and the model had stopped writing. So
the property under test is not "the state moved" -- translation moves it too --
but "the state moved and is still the length the next block expects".
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.consciousness.residual_injection_geometry import (  # noqa: E402
    CLIP,
    CLIP_GROWTH_CEILING,
    LIFT,
    PROJECT,
    ROTATE,
    TRANSLATE,
    inject,
    injection_mode,
)


def _norms(h):
    return mx.linalg.norm(mx.astype(h, mx.float32), axis=-1)


@pytest.fixture
def stream():
    mx.random.seed(11)
    return mx.random.normal(shape=(1, 4, 64)) * mx.array(3.0)


@pytest.fixture
def delta(stream):
    direction = mx.random.normal(shape=(64,))
    direction = direction / mx.linalg.norm(direction)
    reference = float(mx.mean(_norms(stream)))
    return mx.array(0.2 * reference) * direction


def test_translate_lengthens_the_stream(stream, delta):
    """Today's geometry, and the reason for the other two."""
    moved = inject(stream, delta, mode=TRANSLATE)
    assert float(mx.max(_norms(moved) - _norms(stream))) > 0.1


def test_rotate_returns_the_stream_at_its_own_length(stream, delta):
    moved = inject(stream, delta, mode=ROTATE)
    before, after = _norms(stream), _norms(moved)
    assert float(mx.max(mx.abs(after - before))) < 1e-2 * float(mx.mean(before))


def test_rotate_still_turns_the_stream(stream, delta):
    """Preserving the length must not have preserved the direction."""
    moved = inject(stream, delta, mode=ROTATE)
    cosine = mx.sum(stream * moved, axis=-1) / (_norms(stream) * _norms(moved))
    assert float(mx.max(cosine)) < 0.9999


def test_rotate_turns_toward_the_vector_not_away(stream, delta):
    unit = delta / mx.linalg.norm(delta)
    moved = inject(stream, delta, mode=ROTATE)
    before = mx.sum(stream * unit, axis=-1) / _norms(stream)
    after = mx.sum(moved * unit, axis=-1) / _norms(moved)
    assert float(mx.min(after - before)) > 0.0


def test_clip_allows_growth_up_to_its_ceiling(stream, delta):
    """A small nudge keeps the magnitude that made it a nudge."""
    small = delta * mx.array(0.05)
    moved = inject(stream, small, mode=CLIP)
    ratio = _norms(moved) / _norms(stream)
    assert float(mx.min(ratio)) > 1.0
    assert float(mx.max(ratio)) <= CLIP_GROWTH_CEILING + 1e-3


def test_clip_pulls_back_a_large_nudge(stream, delta):
    huge = delta * mx.array(20.0)
    moved = inject(stream, huge, mode=CLIP)
    ratio = _norms(moved) / _norms(stream)
    assert float(mx.max(ratio)) <= CLIP_GROWTH_CEILING + 1e-3


def test_the_mask_leaves_every_other_position_untouched(stream, delta):
    mask = mx.array([[[0.0], [0.0], [0.0], [1.0]]])
    moved = inject(stream, delta, mask=mask, mode=ROTATE)
    # A mask that multiplied the ROTATED state instead of the DIFFERENCE would
    # zero these rows rather than leave them alone.
    assert float(mx.max(mx.abs(moved[:, :3, :] - stream[:, :3, :]))) == 0.0
    assert float(mx.max(mx.abs(moved[:, 3, :] - stream[:, 3, :]))) > 0.0


def test_the_masked_position_keeps_its_own_length(stream, delta):
    mask = mx.array([[[0.0], [0.0], [0.0], [1.0]]])
    moved = inject(stream, delta, mask=mask, mode=ROTATE)
    before = float(_norms(stream)[0, 3])
    after = float(_norms(moved)[0, 3])
    assert abs(after - before) < 1e-2 * before


def test_an_unknown_mode_name_is_not_silently_a_new_geometry(monkeypatch):
    monkeypatch.setenv("AURA_STEERING_INJECTION", "sideways")
    assert injection_mode() == TRANSLATE


@pytest.mark.parametrize("name", [TRANSLATE, ROTATE, CLIP, PROJECT, LIFT])
def test_the_environment_names_the_geometry(monkeypatch, name):
    monkeypatch.setenv("AURA_STEERING_INJECTION", name.upper())
    assert injection_mode() == name


# ── projection: the mode that survives sixteen layers ─────────────────────

def test_projection_sets_the_component_rather_than_adding_to_it(stream, delta):
    unit = delta / mx.linalg.norm(delta)
    target = float(mx.linalg.norm(delta))
    moved = inject(stream, delta, mode=PROJECT)
    component = mx.sum(moved * unit, axis=-1)
    assert float(mx.max(mx.abs(component - target))) < 1e-2 * target


def test_projection_applied_twice_lands_where_it_landed_once(stream, delta):
    """The whole reason it survives sixteen layers: it does not compound."""
    once = inject(stream, delta, mode=PROJECT)
    twice = inject(once, delta, mode=PROJECT)
    assert float(mx.max(mx.abs(twice - once))) < 1e-2 * float(mx.mean(_norms(stream)))


def test_projection_leaves_the_orthogonal_content_alone(stream, delta):
    unit = delta / mx.linalg.norm(delta)

    def orthogonal(h):
        return h - mx.sum(h * unit, axis=-1, keepdims=True) * unit

    moved = inject(stream, delta, mode=PROJECT)
    assert float(mx.max(mx.abs(orthogonal(moved) - orthogonal(stream)))) < 1e-2


def test_lift_never_pulls_a_leaning_sample_back(stream, delta):
    """A sample already past the target keeps what it had."""
    unit = delta / mx.linalg.norm(delta)
    target = mx.linalg.norm(delta)
    already = stream + mx.array(4.0) * target * unit
    moved = inject(already, delta, mode=LIFT)
    assert float(mx.max(mx.abs(moved - already))) == 0.0
    # And projection, which is the comparison, does pull it back.
    pulled = inject(already, delta, mode=PROJECT)
    assert float(mx.max(mx.abs(pulled - already))) > 0.0


def test_lift_still_raises_a_sample_below_the_target(stream, delta):
    unit = delta / mx.linalg.norm(delta)
    target = float(mx.linalg.norm(delta))
    below = stream - mx.sum(stream * unit, axis=-1, keepdims=True) * unit
    moved = inject(below, delta, mode=LIFT)
    component = mx.sum(moved * unit, axis=-1)
    assert float(mx.min(component)) > 0.9 * target


def test_the_mask_still_chooses_positions_under_projection(stream, delta):
    mask = mx.array([[[0.0], [0.0], [0.0], [1.0]]])
    moved = inject(stream, delta, mask=mask, mode=PROJECT)
    assert float(mx.max(mx.abs(moved[:, :3, :] - stream[:, :3, :]))) == 0.0
    assert float(mx.max(mx.abs(moved[:, 3, :] - stream[:, 3, :]))) > 0.0
