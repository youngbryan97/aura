"""Map a measured source pose into registered sibling artwork for fresh search.

Static source-coordinate registrations relate frames of one declared asset.
This supplies possible search shapes. It supplies no current pixels, active
animation frame, continued object identity or material correspondence.
"""
from __future__ import annotations

import numpy as np

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance, SourceFramePoseHint, _digest
from core.perception.material_samples_in_a_reference_fit import material_samples
from core.perception.source_reference_proposals import SourceReferenceProposal
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from core.verify import invariant


def hint_for_source_variant(donor: RegisteredVisualReference, target: RegisteredVisualReference,
        pose: ReferenceAppearance) -> SourceFramePoseHint | None:
    """Transfer only a static coordinate hypothesis between the same source asset."""
    if (not isinstance(donor, RegisteredVisualReference) or not isinstance(target, RegisteredVisualReference)
            or donor.key[:2] != target.key[:2] or donor.variant_key == target.variant_key
            or getattr(donor, 'components', ()) or getattr(target, 'components', ())
            or donor.coordinates.source != donor.source or target.coordinates.source != target.source
            or not isinstance(pose, ReferenceAppearance) or pose.reference_sha256 != donor.key[2]
            or material_samples(pose) is None or _digest(donor.pixels) != donor.key[2]
            or _digest(target.pixels) != target.key[2]):
        return None
    try:
        measured = np.asarray(pose.linear)
        source_to_picture = measured @ np.asarray(donor.coordinates.linear)
        source_origin = measured @ np.asarray(donor.coordinates.offset)+pose.offset
        linear = source_to_picture @ np.linalg.inv(np.asarray(target.coordinates.linear))
        offset = source_origin-linear @ np.asarray(target.coordinates.offset)
        return SourceFramePoseHint(target.key[2], pose.picture_sha256, tuple(map(tuple, linear)), tuple(offset), donor.variant_key)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None


def with_source_variant_hints(proposals: tuple, references: tuple) -> tuple:
    """Keep actual matches first and derive bounded unmeasured variants once."""
    if not isinstance(proposals, tuple) or len(proposals) > 32 or not isinstance(references, tuple) or len(references) > 32:
        return proposals
    actual = tuple(p for p in proposals if isinstance(p, SourceReferenceProposal)
                   and all(isinstance(pose, ReferenceAppearance) for pose in p.appearances))
    result = list(actual)
    keys = {p.reference.variant_key for p in actual}
    for reference in references:
        if not isinstance(reference, RegisteredVisualReference) or reference.variant_key in keys or len(result) >= 32:
            continue
        donors = [p for p in actual if p.reference.key[:2] == reference.key[:2]]
        donors.sort(key=lambda p: (p.capture.at, p.revision), reverse=True)
        for donor in donors:
            hints = tuple(hint for pose in donor.appearances[:4]
                          if (hint := hint_for_source_variant(donor.reference, reference, pose)) is not None)
            if hints:
                result.append(SourceReferenceProposal(reference, donor.capture, hints, donor.revision))
                keys.add(reference.variant_key)
                break
    return tuple(result)


@invariant('perception.source_frame_hints_cannot_be_material_measurements', scope='perception',
           owner='core/perception/search_hints_for_source_frame_variants.py', observational=False)
def _source_variant_hint_invariant() -> tuple:
    hint = SourceFramePoseHint('0'*64, '1'*64, ((1., 0.), (0., 1.)), (2., 3.), ('source', 7, '2'*64, 1))
    assert material_samples(hint) is None and not isinstance(hint, ReferenceAppearance)
    assert with_source_variant_hints((), ()) == ()
    return ()
