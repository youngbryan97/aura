"""Public sampling and raster receipts qualify only one image-versioned draw."""
import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.perception.a_texture_copy_with_public_state import (
    CopyFragmentSource,
    CopyGeometry,
    CopyImageWrite,
    CopyRasterState,
    PublicDraw,
    SamplerParameters,
    TextureSamplingState,
    _effective_texture_copy_invariant,
    admit_texture_copy,
)
from core.perception.graphics_presented_on_canvas import (
    FramebufferCopy,
    ImageVersion,
    PresentedCanvas,
    presented_hints,
)


def _case():
    source = ("#version 300 es\nuniform sampler2D picture;in vec2 coordinate;out vec4 pixel;"
                              "void main(){pixel=texture(picture,coordinate);}")
    draw = PublicDraw(1, 2, 20, 47, 3)
    program = CopyFragmentSource(draw, source, hashlib.sha256(source.encode()).hexdigest(), (('pixel', 0),))
    source = ImageVersion(1, 2, 45, 19, 2560, 1522)
    before = ImageVersion(1, 2, 49, 7, 2560, 1522)
    write = CopyImageWrite(draw, source, before, replace(before, content=8), 36064)
    # These conflicting filter and wrap values match the recorded public
    # snapshot. Comparison and sample-coverage state must be supplied too.
    texture = SamplerParameters(9986, 9729, 10497, 10497, -1000., 1000., 0)
    sampler = SamplerParameters(9984, 9728, 33071, 33071, 0., 100., 0)
    sampling = TextureSamplingState(draw, "picture", 0, source, 3553, 0, 0, texture, 13, sampler)
    raster = CopyRasterState(draw, (0, 0, 2560, 1522), True, (0, 0, 2560, 1522), True,
                            1, 0, 1, 0, 32774, 32774, (True,) * 4,
                            False, False, False, False, False, False, draw_buffers=(36064,))
    geometry = CopyGeometry(draw, "coordinate", ((-1., -1., 0., 1.), (1., -1., 0., 1.),
                                                (1., 1., 0., 1.), (-1., 1., 0., 1.)),
                            ((0., 0.), (1., 0.), (1., 1.), (0., 1.)), (0, 1, 2, 0, 2, 3))
    return program, write, sampling, raster, geometry


def test_bound_sampler_overrides_conflicting_texture_values_in_a_replacement_draw():
    program, write, sampling, raster, geometry = _case()
    admitted = admit_texture_copy(program, write, sampling, raster, geometry)
    assert admitted.reason is None and admitted.projection is not None
    projection = admitted.projection
    assert projection.source is write.source and projection.destination is write.destination_after
    assert projection.sequence == write.draw.sequence
    assert projection.scissor == (0., 0., 2560., 1522.)
    region = (((10., 20.), (30., 20.), (30., 40.), (10., 40.)),)
    mapped = projection.mapped(region)
    assert mapped is not None
    assert sum(len(polygon) for polygon in mapped) >= 4
    assert {tuple(round(v, 8) for v in point) for polygon in mapped for point in polygon}.issuperset(region[0])


def test_admitted_projection_follows_the_existing_presentation_route_without_material_authority():
    args = _case()
    projection = admit_texture_copy(*args).projection
    source, canvas = projection.source, ImageVersion(1, 2, 0, 12, 2560, 1522)
    blit = FramebufferCopy(projection.destination, canvas, (0., 1522., 2560., 0.),
                           (0., 0., 2560., 1522.), 21)
    at = MeasuredFrameTime(1., lower=1., upper=1.01, arrived=1.02,
                           source="browser-monotonic", source_at=10., epoch=2)
    region = (((10., 20.), (30., 20.), (30., 40.), (10., 40.)),)
    result = presented_hints(region, source, 19, (projection, blit), PresentedCanvas(canvas, 22, at))
    assert result.reason is None and result.hints.coverage == "unverified"
    assert result.hints.copies == (20, 21)
    assert not hasattr(result.hints, "body")


@pytest.mark.parametrize("change", [
    {"sampler_parameters": None}, {"sampler": None}, {"unit": None}, {"unit": True},
    {"uniform": "another"}, {"target": 34067}, {"base_level": None}, {"base_level": 1},
    {"max_level": None}, {"max_level": 1}, {"base_level": False},
])
def test_incomplete_or_unsupported_sampling_never_uses_texture_state_as_a_fallback(change):
    program, write, sampling, raster, geometry = _case()
    result = admit_texture_copy(program, write, replace(sampling, **change), raster, geometry)
    assert result.projection is None and result.reason == "sampling_state"


@pytest.mark.parametrize("field,value", [
    ("min_filter", 9729), ("min_filter", None), ("mag_filter", 9729), ("wrap_s", 10497),
    ("wrap_t", 10497), ("compare_mode", None), ("compare_mode", 34894),
    ("min_lod", None), ("min_lod", 1.), ("max_lod", -1.), ("max_lod", float("nan")),
])
def test_bad_bound_sampler_is_not_rescued_by_good_texture_parameters(field, value):
    program, write, sampling, raster, geometry = _case()
    good = sampling.sampler_parameters
    sampling = replace(sampling, texture_parameters=good, sampler_parameters=replace(good, **{field: value}))
    assert admit_texture_copy(program, write, sampling, raster, geometry).reason == "sampling_state"


@pytest.mark.parametrize("min_filter", [9728, 9984])
def test_observed_unbound_sampler_uses_complete_texture_parameters(min_filter):
    program, write, sampling, raster, geometry = _case()
    sampling = replace(sampling, sampler=0, sampler_parameters=None,
                       texture_parameters=replace(sampling.sampler_parameters, min_filter=min_filter))
    assert admit_texture_copy(program, write, sampling, raster, geometry).projection is not None
    assert admit_texture_copy(program, write, replace(sampling, texture_parameters=None), raster, geometry).projection is None
    assert admit_texture_copy(program, write, replace(sampling, sampler_parameters=sampling.texture_parameters),
                              raster, geometry).projection is None


@pytest.mark.parametrize("change", [
    {"blend_enabled": None}, {"source_rgb": 770}, {"destination_rgb": 771},
    {"source_alpha": 770}, {"destination_alpha": 771}, {"equation_rgb": 32778},
    {"equation_alpha": 32779}, {"equation_alpha": None}, {"source_rgb": True},
    {"color_mask": (True, True, True, False)}, {"color_mask": (1, 1, 1, 1)},
    {"depth_test": True}, {"stencil_test": True}, {"cull_face": True},
    {"rasterizer_discard": True}, {"sample_alpha_to_coverage": True}, {"sample_coverage": True},
    {"sample_coverage": None}, {"scissor_enabled": None}, {"scissor_box": None},
    {"scissor_box": (0, 0, 0, 1522)}, {"viewport": None}, {"viewport": (0, 0, 0, 1522)},
    {"viewport": (0., 0., 2560., 1522.)},
])
def test_source_alpha_blends_and_incomplete_or_restrictive_raster_state_are_refused(change):
    program, write, sampling, raster, geometry = _case()
    result = admit_texture_copy(program, write, sampling, replace(raster, **change), geometry)
    assert result.projection is None and result.reason == "raster_state"


def test_disabled_blending_has_no_dependence_on_its_measured_unused_factors():
    program, write, sampling, raster, geometry = _case()
    raster = replace(raster, blend_enabled=False, source_rgb=770, destination_rgb=771,
                     source_alpha=770, destination_alpha=771, equation_rgb=32778, equation_alpha=32779)
    assert admit_texture_copy(program, write, sampling, raster, geometry).projection is not None
    assert admit_texture_copy(program, write, sampling, replace(raster, source_rgb=None), geometry).projection is None


def test_public_scissor_xy_width_height_becomes_the_existing_endpoint_rectangle():
    program, write, sampling, raster, geometry = _case()
    raster = replace(raster, viewport=(10, 20, 200, 100), scissor_box=(15, 25, 30, 40))
    projection = admit_texture_copy(program, write, sampling, raster, geometry).projection
    assert projection.viewport == (10., 20., 200., 100.)
    assert projection.scissor == (15., 25., 45., 65.)
    disabled = admit_texture_copy(program, write, sampling, replace(raster, scissor_enabled=False), geometry).projection
    assert disabled.scissor is None


@pytest.mark.parametrize("field,value", [
    ("context", 9), ("epoch", 9), ("sequence", 19), ("program", 48), ("link", 2),
])
@pytest.mark.parametrize("receipt", ["sampling", "raster", "geometry"])
def test_state_from_a_different_context_epoch_draw_or_link_is_not_borrowed(field, value, receipt):
    program, write, sampling, raster, geometry = _case()
    changed = replace(write.draw, **{field: value})
    values = dict(sampling=sampling, raster=raster, geometry=geometry)
    values[receipt] = replace(values[receipt], draw=changed)
    result = admit_texture_copy(program, write, **values)
    assert result.reason == "draw_or_image_custody"


@pytest.mark.parametrize("part,change", [
    ("source", {"epoch": 3}), ("source", {"content": 0}),
    ("destination_before", {"image": 50}), ("destination_before", {"content": 6}),
    ("destination_after", {"content": 7}), ("destination_after", {"content": 9}),
    ("destination_after", {"width": 100}), ("destination_after", {"context": 2}),
])
def test_attachment_reuse_changed_content_or_nonadjacent_write_generations_are_refused(part, change):
    program, write, sampling, raster, geometry = _case()
    write = replace(write, **{part: replace(getattr(write, part), **change)})
    assert admit_texture_copy(program, write, sampling, raster, geometry).projection is None


def test_a_stale_texture_version_cannot_supply_the_current_sampled_image():
    program, write, sampling, raster, geometry = _case()
    stale = replace(sampling, image=replace(write.source, content=write.source.content - 1))
    assert admit_texture_copy(program, write, stale, raster, geometry).reason == "sampling_state"
    feedback = replace(write, source=write.destination_before)
    assert admit_texture_copy(program, feedback, replace(sampling, image=feedback.source), raster, geometry).projection is None


@pytest.mark.parametrize("change", [
    {"coordinates_name": "another"}, {"positions": ((0., 0., 0., 1.),) * 65},
    {"coordinates": ((0., 0.),) * 4}, {"coordinates": ((0., 0.), (2., 0.), (1., 1.), (0., 1.))},
    {"indices": (0, 1, 4)}, {"indices": (0, 1, 2) * 65},
    {"positions": ((0., 0., float("nan"), 1.),) * 4},
])
def test_coordinate_names_singular_mapping_nonfinite_values_and_geometry_budgets_are_checked(change):
    program, write, sampling, raster, geometry = _case()
    assert admit_texture_copy(program, write, sampling, raster, replace(geometry, **change)).projection is None


def test_changed_fragment_arithmetic_is_not_admitted_by_good_public_state():
    program, write, sampling, raster, geometry = _case()
    source = ("#version 300 es\nuniform sampler2D picture;in vec2 coordinate;out vec4 pixel;"
                              "void main(){pixel=texture(picture,coordinate)*vec4(0.5);}")
    changed = replace(program, source=source, digest=hashlib.sha256(source.encode()).hexdigest())
    assert admit_texture_copy(changed, write, sampling, raster, geometry).reason == "shader_dataflow"
    assert admit_texture_copy(None, write, sampling, raster, geometry).reason == "copy_contract"


@pytest.mark.parametrize('change', [
    {'digest': '0' * 64}, {'source': ''}, {'source': 'x' * 65537},
    {'source': '\ud800'}, {'source': 'void main(){discard;}'},
])
def test_incomplete_or_corrupted_fragment_source_cannot_supply_a_copy(change):
    program, write, sampling, raster, geometry = _case()
    assert admit_texture_copy(replace(program, **change), write, sampling, raster, geometry).projection is None


def test_a_fragment_source_from_another_program_link_is_not_borrowed():
    program, write, sampling, raster, geometry = _case()
    stale = replace(program, draw=replace(program.draw, link=program.draw.link-1))
    assert admit_texture_copy(stale, write, sampling, raster, geometry).reason == 'draw_or_image_custody'


@pytest.mark.parametrize('part,change', [
    ('fragment', {'output_locations': ()}), ('fragment', {'output_locations': (('pixel', 1),)}),
    ('fragment', {'output_locations': (('other', 0),)}), ('fragment', {'output_locations': (('pixel', True),)}),
    ('raster', {'draw_buffers': None}), ('raster', {'draw_buffers': (0,)}),
    ('raster', {'draw_buffers': (36065,)}), ('write', {'attachment': None}),
])
def test_the_fragment_output_must_write_the_recorded_attachment(part, change):
    fragment, write, sampling, raster, geometry = _case()
    values = dict(fragment=fragment, write=write, sampling=sampling, raster=raster, geometry=geometry)
    values[part] = replace(values[part], **change)
    assert admit_texture_copy(**values).reason == 'output_attachment'


def test_extreme_numeric_inputs_are_refused_without_unbounded_arithmetic():
    fragment, write, sampling, raster, geometry = _case()
    bad = replace(geometry, positions=((10**1000, 0., 0., 1.),) * 4)
    assert admit_texture_copy(fragment, write, sampling, raster, bad).reason == 'copy_geometry'
    sampling = replace(sampling, sampler_parameters=replace(sampling.sampler_parameters, min_lod=10**1000))
    assert admit_texture_copy(fragment, write, sampling, raster, geometry).reason == 'sampling_state'


def test_receipts_are_immutable_and_the_registered_invariant_runs():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    args = _case()
    with pytest.raises(FrozenInstanceError):
        args[2].sampler = 0
    assert _effective_texture_copy_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'texture_copy_uses_effective_public_draw_state')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
