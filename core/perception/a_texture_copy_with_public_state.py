"""Admit a texture projection from one complete public draw snapshot.

The admission preserves image generations and checks sampling and replacement
arithmetic. Its output remains geometry with unverified fragment coverage.
Collectors must obtain the receipts from the same successful linked draw.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from core.perception.a_texture_copied_by_its_shader import texture_copy_form
from core.perception.graphics_presented_on_canvas import ImageVersion
from core.perception.shader_position_source import UnsupportedPositionError, parse_positions
from core.perception.texture_coordinates_to_a_surface import TextureProjection
from core.verify import invariant

_NEAREST, _NEAREST_MIPMAP_NEAREST, _CLAMP_TO_EDGE = 9728, 9984, 33071
_TEXTURE_2D, _FUNC_ADD = 3553, 32774


@dataclass(frozen=True)
class PublicDraw:
    context: int | None
    epoch: int | None
    sequence: int | None
    program: int | None
    link: int | None

    def valid(self) -> bool:
        return all(type(v) is int and 0 < v <= 2**53 - 1
                   for v in (self.context, self.epoch, self.sequence, self.program, self.link))


@dataclass(frozen=True)
class SamplerParameters:
    min_filter: int | None = None
    mag_filter: int | None = None
    wrap_s: int | None = None
    wrap_t: int | None = None
    min_lod: float | None = None
    max_lod: float | None = None
    compare_mode: int | None = None


@dataclass(frozen=True)
class TextureSamplingState:
    draw: PublicDraw
    uniform: str | None
    unit: int | None
    image: ImageVersion | None
    target: int | None
    base_level: int | None
    max_level: int | None
    texture_parameters: SamplerParameters | None
    sampler: int | None
    sampler_parameters: SamplerParameters | None


@dataclass(frozen=True)
class CopyRasterState:
    draw: PublicDraw
    viewport: tuple | None = None
    scissor_enabled: bool | None = None
    scissor_box: tuple | None = None
    blend_enabled: bool | None = None
    source_rgb: int | None = None
    destination_rgb: int | None = None
    source_alpha: int | None = None
    destination_alpha: int | None = None
    equation_rgb: int | None = None
    equation_alpha: int | None = None
    color_mask: tuple | None = None
    depth_test: bool | None = None
    stencil_test: bool | None = None
    cull_face: bool | None = None
    rasterizer_discard: bool | None = None
    sample_alpha_to_coverage: bool | None = None
    sample_coverage: bool | None = None
    draw_buffers: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CopyImageWrite:
    draw: PublicDraw
    source: ImageVersion
    destination_before: ImageVersion
    destination_after: ImageVersion
    attachment: int | None = None


@dataclass(frozen=True)
class CopyFragmentSource:
    """The fragment source attached to this successful program link."""

    draw: PublicDraw
    source: str
    digest: str
    output_locations: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class CopyGeometry:
    """Evaluated public vertex outputs, owned by the same linked draw."""

    draw: PublicDraw
    coordinates_name: str | None
    positions: tuple
    coordinates: tuple
    indices: tuple


@dataclass(frozen=True)
class CopyAdmission:
    projection: TextureProjection | None
    reason: str | None = None


def _same_draw(draw: PublicDraw, *receipts) -> bool:
    return (isinstance(draw, PublicDraw) and draw.valid()
            and all(getattr(receipt, "draw", None) == draw for receipt in receipts))


def _finite(value) -> bool:
    return type(value) in (float, int) and abs(value) <= 1e12 and math.isfinite(value)


def _box(value) -> bool:
    return (type(value) is tuple and len(value) == 4
            and all(type(v) is int and abs(v) <= 2**20 for v in value)
            and 0 < value[2] <= 32768 and 0 < value[3] <= 32768)


def _image_write(write: CopyImageWrite) -> bool:
    images = (write.source, write.destination_before, write.destination_after)
    if any(not isinstance(image, ImageVersion) or not image.valid() for image in images):
        return False
    source, before, after = images
    return (all((image.context, image.epoch) == (write.draw.context, write.draw.epoch) for image in images)
            and all(0 <= value <= 2**53-1 for image in images for value in (image.image, image.content))
            and source.content >= 1 and before.content >= 0
            and source.image != after.image and before.image == after.image
            and (before.width, before.height) == (after.width, after.height)
            and after.content == before.content + 1)


def _output_attachment(fragment: CopyFragmentSource, raster: CopyRasterState,
                       write: CopyImageWrite, name: str) -> bool:
    locations, buffers = fragment.output_locations, raster.draw_buffers
    if (type(locations) is not tuple or len(locations) != 1 or type(locations[0]) is not tuple
            or len(locations[0]) != 2 or locations[0][0] != name
            or type(locations[0][1]) is not int or not 0 <= locations[0][1] < 32
            or type(buffers) is not tuple or not 0 < len(buffers) <= 32
            or any(type(value) is not int for value in buffers)
            or locations[0][1] >= len(buffers) or type(write.attachment) is not int):
        return False
    expected = buffers[locations[0][1]]
    attachment_valid = (expected == 1029 if write.destination_after.image == 0 else 36064 <= expected < 36096)
    return attachment_valid and write.attachment == expected


def _bounded_geometry(geometry: CopyGeometry) -> bool:
    return (type(geometry.positions) is tuple and 3 <= len(geometry.positions) <= 64
            and type(geometry.coordinates) is tuple and len(geometry.coordinates) == len(geometry.positions)
            and all(type(point) is tuple and len(point) == 4 and all(_finite(v) for v in point)
                    for point in geometry.positions)
            and all(type(point) is tuple and len(point) == 2 and all(_finite(v) and 0 <= v <= 1 for v in point)
                    for point in geometry.coordinates)
            and type(geometry.indices) is tuple and 0 < len(geometry.indices) <= 192
            and len(geometry.indices) % 3 == 0
            and all(type(i) is int and 0 <= i < len(geometry.positions) for i in geometry.indices))


def _nearest_sampling(state: TextureSamplingState, source: ImageVersion, sampler_name: str) -> bool:
    if (state.uniform != sampler_name or type(state.unit) is not int or not 0 <= state.unit < 256
            or state.image != source or type(state.target) is not int or state.target != _TEXTURE_2D
            or type(state.base_level) is not int or state.base_level != 0
            or type(state.max_level) is not int or state.max_level != 0
            or type(state.sampler) is not int or not 0 <= state.sampler <= 2**53 - 1):
        return False
    # Zero is an observed unbound sampler; None is missing evidence. A bound
    # sampler replaces the texture's sampling parameters, including unknowns.
    if state.sampler == 0:
        if state.sampler_parameters is not None:
            return False
        effective = state.texture_parameters
    else:
        effective = state.sampler_parameters
    if not isinstance(effective, SamplerParameters):
        return False
    enums = (effective.min_filter, effective.mag_filter, effective.wrap_s,
             effective.wrap_t, effective.compare_mode)
    return (all(type(v) is int for v in enums)
            and effective.min_filter in (_NEAREST, _NEAREST_MIPMAP_NEAREST)
            and effective.mag_filter == _NEAREST
            and effective.wrap_s == effective.wrap_t == _CLAMP_TO_EDGE
            and effective.compare_mode == 0
            and _finite(effective.min_lod) and _finite(effective.max_lod)
            and effective.min_lod <= 0 <= effective.max_lod)


def _replacement_raster(state: CopyRasterState) -> bool:
    flags = (state.depth_test, state.stencil_test, state.cull_face, state.rasterizer_discard,
             state.sample_alpha_to_coverage, state.sample_coverage)
    if (any(flag is not False for flag in flags) or type(state.color_mask) is not tuple
            or len(state.color_mask) != 4 or any(channel is not True for channel in state.color_mask)
            or type(state.blend_enabled) is not bool or type(state.scissor_enabled) is not bool
            or not _box(state.viewport) or not _box(state.scissor_box)):
        return False
    values = (state.source_rgb, state.destination_rgb, state.source_alpha,
              state.destination_alpha, state.equation_rgb, state.equation_alpha)
    if any(type(value) is not int or not 0 <= value <= 65535 for value in values):
        return False
    return (state.blend_enabled is False
            or values == (1, 0, 1, 0, _FUNC_ADD, _FUNC_ADD))


def admit_texture_copy(fragment: CopyFragmentSource, write: CopyImageWrite, sampling: TextureSamplingState,
                       raster: CopyRasterState, geometry: CopyGeometry) -> CopyAdmission:
    """Produce an existing projection only after every public-state check passes.

    Image identifiers name attachment images, including their storage and
    subresource. The before/after generations describe this single write.
    No source opacity, clear state, material identity or control is inferred.
    """
    if (not isinstance(fragment, CopyFragmentSource) or not isinstance(write, CopyImageWrite)
            or not isinstance(sampling, TextureSamplingState) or not isinstance(raster, CopyRasterState)
            or not isinstance(geometry, CopyGeometry)):
        return CopyAdmission(None, "copy_contract")
    if not _same_draw(write.draw, fragment, sampling, raster, geometry) or not _image_write(write):
        return CopyAdmission(None, "draw_or_image_custody")
    if (type(fragment.source) is not str or not 0 < len(fragment.source) <= 65536
            or type(fragment.digest) is not str):
        return CopyAdmission(None, "shader_source")
    try:
        if hashlib.sha256(fragment.source.encode('utf-8')).hexdigest() != fragment.digest:
            return CopyAdmission(None, "shader_source")
        program = parse_positions(fragment.source)
    except (UnicodeError, UnsupportedPositionError, ValueError, TypeError):
        return CopyAdmission(None, "shader_source")
    form = texture_copy_form(program)
    if form is None or geometry.coordinates_name != form.coordinates:
        return CopyAdmission(None, "shader_dataflow")
    if not _output_attachment(fragment, raster, write, form.output):
        return CopyAdmission(None, "output_attachment")
    if not _nearest_sampling(sampling, write.source, form.sampler):
        return CopyAdmission(None, "sampling_state")
    if not _replacement_raster(raster):
        return CopyAdmission(None, "raster_state")
    if not _bounded_geometry(geometry):
        return CopyAdmission(None, "copy_geometry")
    scissor = None
    if raster.scissor_enabled:
        x, y, width, height = raster.scissor_box
        scissor = (float(x), float(y), float(x + width), float(y + height))
    projection = TextureProjection(write.source, write.destination_after, geometry.positions,
                                   geometry.coordinates, geometry.indices, tuple(float(v) for v in raster.viewport),
                                   write.draw.sequence, scissor=scissor, smooth=True, base_level_nearest=True)
    if not projection.valid():
        return CopyAdmission(None, "copy_geometry")
    bounds = (((0., 0.), (float(write.source.width), 0.),
               (float(write.source.width), float(write.source.height)), (0., float(write.source.height))),)
    if projection.mapped(bounds) is None:
        return CopyAdmission(None, "copy_geometry")
    return CopyAdmission(projection)


@invariant("perception.texture_copy_uses_effective_public_draw_state", scope="perception",
           owner="core/perception/a_texture_copy_with_public_state.py", observational=False)
def _effective_texture_copy_invariant() -> tuple:
    from dataclasses import replace

    draw = PublicDraw(1, 2, 7, 8, 1)
    source = ImageVersion(1, 2, 3, 4, 32, 24)
    before = ImageVersion(1, 2, 5, 6, 32, 24)
    write = CopyImageWrite(draw, source, before, replace(before, content=7), 36064)
    parameters = SamplerParameters(_NEAREST_MIPMAP_NEAREST, _NEAREST, _CLAMP_TO_EDGE, _CLAMP_TO_EDGE, 0., 100., 0)
    sampling = TextureSamplingState(draw, "image", 0, source, _TEXTURE_2D, 0, 0, None, 9, parameters)
    raster = CopyRasterState(draw, (0, 0, 32, 24), True, (0, 0, 32, 24), True,
                            1, 0, 1, 0, _FUNC_ADD, _FUNC_ADD, (True,) * 4, False, False, False, False, False, False,
                            draw_buffers=(36064,))
    geometry = CopyGeometry(draw, "where", ((-1., -1., 0., 1.), (1., -1., 0., 1.), (-1., 1., 0., 1.)),
                            ((0., 0.), (1., 0.), (0., 1.)), (0, 1, 2))
    source = "#version 300 es\nuniform sampler2D image;in vec2 where;out vec4 pixel;void main(){pixel=texture(image,where);}"
    fragment = CopyFragmentSource(draw, source, hashlib.sha256(source.encode()).hexdigest(), (('pixel', 0),))
    assert admit_texture_copy(fragment, write, sampling, raster, geometry).projection is not None
    assert admit_texture_copy(fragment, write, replace(sampling, sampler_parameters=None), raster, geometry).projection is None
    assert admit_texture_copy(fragment, write, sampling, replace(raster, destination_alpha=771), geometry).projection is None
    return ()
