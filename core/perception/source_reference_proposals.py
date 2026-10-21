"""Captured reference geometry offered for fresh measurement, without input authority."""
from __future__ import annotations

from dataclasses import dataclass

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance, SourceFramePoseHint
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.perception.visual_references_for_current_sources import RegisteredVisualReference


@dataclass(frozen=True)
class SourceReferenceProposal:
    """Captured matches or explicitly unmeasured sibling-frame search hints."""
    reference: RegisteredVisualReference
    capture: PixelsConsumed
    appearances: tuple[ReferenceAppearance | SourceFramePoseHint, ...]
    revision: int

    def __post_init__(self) -> None:
        if (not isinstance(self.reference, RegisteredVisualReference) or not isinstance(self.capture, PixelsConsumed)
                or type(self.revision) is not int or self.revision < 1
                or not isinstance(self.appearances, tuple) or not 1 <= len(self.appearances) <= 8
                or any(not isinstance(pose, (ReferenceAppearance, SourceFramePoseHint))
                       or pose.reference_sha256 != self.reference.coordinates.reference_sha256
                       or pose.picture_sha256 != self.capture.sha256
                       or isinstance(pose, SourceFramePoseHint) and pose.donor_variant[:2] != self.reference.key[:2]
                       for pose in self.appearances)):
            raise ValueError('a reference proposal needs bounded captured matches or source-frame search hints')
