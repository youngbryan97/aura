"""Build bounded layered search pictures from registered source artwork.

Source placements propose a raster. Ordinary correspondence and current
material checks must still establish its presence. Component provenance is
retained, with execution, frames, roles and contact geometry unresolved.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
from core.perception.source_layers_with_copied_positions import SourceLayerProposal
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from core.verify import invariant


@dataclass(frozen=True)
class ComposedVisualReference(RegisteredVisualReference):
    components: tuple[RegisteredVisualReference, ...] = field(default=(), repr=False, compare=False)
    placement: SourceLayerProposal | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (len(self.components) != 2 or not isinstance(self.placement, SourceLayerProposal)
                or any(not isinstance(c, RegisteredVisualReference) or isinstance(c, ComposedVisualReference)
                       or c.source != self.source for c in self.components)
                or self.placement.source != self.source
                or tuple(c.declaration.character_id for c in self.components)
                   != (self.placement.base_character, self.placement.over_character)
                or _digest(self.pixels) != self.coordinates.reference_sha256):
            raise ValueError('a composed search reference needs both current source components')

    @property
    def composition_identity(self) -> str:
        payload = (tuple((c.variant_key, c.coordinates.linear, c.coordinates.offset) for c in self.components), self.placement.source_delta,
                   self.placement.depths, self.placement.instruction_source, self.placement.offsets)
        return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()

    @property
    def variant_key(self) -> tuple:
        # A composition is no additional declared source timeline frame.
        return (*self.key, 'composition:'+self.composition_identity)


def compose_source_layers(base: RegisteredVisualReference, over: RegisteredVisualReference,
                          placement: SourceLayerProposal) -> ComposedVisualReference | None:
    if (not isinstance(placement, SourceLayerProposal)
            or not all(isinstance(r, RegisteredVisualReference) and not isinstance(r, ComposedVisualReference) for r in (base, over))
            or base.source != over.source or base.source != placement.source
            or (base.declaration.character_id, over.declaration.character_id) != (placement.base_character, placement.over_character)
            or placement.depths[0] >= placement.depths[1]
            or any(_digest(r.pixels) != r.coordinates.reference_sha256 for r in (base, over))
            or len(placement.source_delta) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                or abs(v) > 4096 for v in placement.source_delta)):
        return None
    cbase, cover = base.coordinates, over.coordinates
    matrix = np.asarray(cbase.linear) @ np.linalg.inv(np.asarray(cover.linear))
    offset = np.asarray(cbase.offset)+np.asarray(cbase.linear) @ placement.source_delta-matrix @ cover.offset
    # Independent static registrations can differ by floating round-off.
    if np.max(abs(matrix-np.round(matrix))) < 1e-4:
        matrix = np.round(matrix)
    if np.max(abs(offset-np.round(offset))) < 1e-4:
        offset = np.round(offset)
    if abs(np.linalg.det(matrix)) < 1e-8 or np.linalg.cond(matrix) > 100:
        return None
    height, width = over.pixels.shape[:2]
    corners = np.array(((0., 0.), (width-1., 0.), (0., height-1.), (width-1., height-1.))) @ matrix.T+offset
    low = np.floor(np.minimum((0., 0.), corners.min(axis=0))).astype(int)
    high = np.ceil(np.maximum(np.array(base.pixels.shape[1::-1])-1, corners.max(axis=0))).astype(int)+1
    size = high-low
    if np.any(size < 16) or np.any(size > 512):
        return None
    yy, xx = np.mgrid[low[1]:high[1], low[0]:high[0]]
    positions = np.stack((xx, yy), axis=-1)

    def sampled(reference, xy):
        rgba = reference.pixels.astype(float)/255.
        rgba[..., :3] *= rgba[..., 3:]
        return np.stack([ndimage.map_coordinates(rgba[..., c], (xy[..., 1], xy[..., 0]), order=1,
            mode='constant', cval=0., prefilter=False) for c in range(4)], axis=-1)

    below = sampled(base, positions)
    above = sampled(over, (positions-offset) @ np.linalg.inv(matrix).T)
    rgba = above+below*(1-above[..., 3:])
    rgba[..., :3] = np.divide(rgba[..., :3], rgba[..., 3:], out=np.zeros_like(rgba[..., :3]), where=rgba[..., 3:] > 0)
    pixels = np.rint(np.clip(rgba, 0, 1)*255).astype(np.uint8)
    pixels.setflags(write=False)
    digest = hashlib.sha256(json.dumps((cbase.svg_sha256, cover.svg_sha256, placement.source_delta,
        placement.depths, placement.offsets), separators=(',', ':')).encode()).hexdigest()
    coordinates = SourceImageCoordinates(base.source, _digest(pixels), digest, tuple(map(int, size)),
        cbase.linear, tuple(float(v) for v in np.asarray(cbase.offset)-low),
        max(cbase.calibration_error, cover.calibration_error))
    return ComposedVisualReference(base.source, base.declaration, coordinates, pixels,
        base.frame_number, base.declared_frame_count, (base, over), placement)


def with_source_layer_proposals(references: tuple, proposals: tuple, *, capacity: int = 32) -> tuple[tuple, tuple]:
    """Keep raw references and static-frame coverage distinct from compositions."""
    if type(capacity) is not int or not 1 <= capacity <= 32 or not isinstance(proposals, tuple) or len(proposals) > 32:
        return references, ('source_layer_proposal_capacity_unavailable',)
    result, refused, seen = list(references), [], set()
    for placement in proposals:
        if not isinstance(placement, SourceLayerProposal):
            refused.append('source_layer_proposal_unavailable')
            continue
        first = [r for r in references if r.source == placement.source and r.declaration.character_id == placement.base_character]
        second = [r for r in references if r.source == placement.source and r.declaration.character_id == placement.over_character]
        if not first or not second:
            refused.append('source_layer_reference_unavailable')
            continue
        if len(first)*len(second) > 16:
            refused.append('source_layer_variant_capacity_unavailable')
            continue
        for base in first:
            for over in second:
                if len(result) >= capacity:
                    refused.append('composed_reference_capacity_unavailable')
                    return tuple(result), tuple(dict.fromkeys(refused))
                composed = compose_source_layers(base, over, placement)
                if composed is None:
                    refused.append('source_layer_transform_unavailable')
                elif composed.variant_key not in seen:
                    seen.add(composed.variant_key)
                    result.append(composed)
    return tuple(result), tuple(dict.fromkeys(refused))


def composition_record(reference: RegisteredVisualReference) -> dict | None:
    if not isinstance(reference, ComposedVisualReference):
        return None
    placement = reference.placement
    return dict(source=reference.source, character_id=reference.declaration.character_id,
        reference_sha256=reference.coordinates.reference_sha256, composition_identity=reference.composition_identity,
        components=tuple(dict(character_id=c.declaration.character_id, frame_number=c.frame_number,
                              reference_sha256=c.coordinates.reference_sha256) for c in reference.components),
        instruction_source=placement.instruction_source, method=placement.method,
        offsets=placement.offsets, source_delta=placement.source_delta, depths=placement.depths,
        uncertainties=placement.uncertainties, execution_verified=False, geometry_authority=False)


@invariant('perception.composed_source_artwork_remains_a_pixel_search_proposal', scope='perception',
           owner='core/perception/composed_source_artwork_for_pixel_search.py', observational=False)
def _composed_source_artwork_invariant() -> tuple:
    assert with_source_layer_proposals((), ()) == ((), ())
    assert compose_source_layers(None, None, None) is None
    return ()
