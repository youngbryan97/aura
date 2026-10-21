"""Carry an owned missing-extent request into bounded current measurement.

Requests can order existing optional refits. Receipts report the current
extent measurement, including refusal; neither supplies control authority.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.perception.source_material_groups_for_visual_attention import SourceMaterialGroup
from core.verify import invariant


@dataclass(frozen=True)
class SourceExtentRequest:
    material: SourceMaterialGroup

    def __post_init__(self) -> None:
        if not isinstance(self.material, SourceMaterialGroup):
            raise ValueError('an extent request needs a current source-material observation')


def request_source_extent(observer: Any, need: Any, at: float) -> bool:
    """Queue only the exact current candidate named by the geometric refusal."""
    if (getattr(need, 'operation', None) != 'measure_whole_extent'
            or getattr(need, 'at', None) != at or getattr(need, 'candidate_count', None) != 1):
        return False
    candidates = getattr(need, 'candidates', ())
    if not isinstance(candidates, tuple) or len(candidates) != 1:
        return False
    owner = getattr(candidates[0], 'owner', None)
    groups = getattr(observer, 'source_material_groups', {})
    if not isinstance(groups, dict):
        return False
    matches = [g for g in groups.values() if isinstance(g, SourceMaterialGroup)
               and (g.number, g.born, g.kind) == owner and g.seen == at
               and g.pixels == getattr(observer, 'pixels_consumed', None)]
    if len(matches) != 1:
        return False
    observer.source_extent_requests = (SourceExtentRequest(matches[0]),)
    return True


def current_extent_requests(requests: Any, records: dict, at: float, tables: tuple) -> tuple[tuple, tuple]:
    """A worker validates requests against its own preceding material records."""
    active = {getattr(table, 'source', None) for table in tables}
    if not isinstance(requests, tuple) or len(requests) > 4:
        return (), ({'status': 'extent_request_capacity_unavailable'},)
    qualified, refused = [], []
    for request in requests:
        if not isinstance(request, SourceExtentRequest):
            refused.append({'status': 'extent_request_owner_unavailable'})
            continue
        material = request.material
        old = records.get(material.number)
        status = None
        if old != material:
            status = 'extent_request_owner_unavailable'
        elif material.source not in active:
            status = 'extent_request_source_changed'
        elif type(at) not in (int, float) or not math.isfinite(at) or not 0 < at-material.seen <= .75:
            status = 'extent_request_capture_expired'
        if status is not None:
            refused.append(_receipt(material, at, status))
        elif request not in qualified:
            qualified.append(request)
    return tuple(qualified), tuple(refused)


def _receipt(material: SourceMaterialGroup, at: float, status: str, **fields: Any) -> dict:
    return dict(owner=(material.number, material.born, material.kind), requested_at=material.seen,
                measured_at=at, source=material.source, character_id=material.character_id,
                status=status, geometry_authority=False, **fields)


def measured_extent_receipts(requests: tuple, readings: tuple, wholes: dict, at: float,
                            pixels: PixelsConsumed | None) -> tuple:
    """Reuse owned measurements; do not run another matcher or discard a refusal."""
    receipts = []
    for request in requests:
        material = request.material
        if not isinstance(pixels, PixelsConsumed) or pixels.at != at:
            receipts.append(_receipt(material, at, 'current_capture_unavailable', measurements=0, reasons=(), observed_owner=None))
            continue
        attempts = [reading for reading in readings if reading.get('source') == material.source
                    and reading.get('character_id') == material.character_id
                    and reading.get('mark_epoch') == material.mark_epoch
                    and reading.get('captured_at') == at and reading.get('picture_sha256') == pixels.sha256
                    and len(frozenset(reading.get('material_anchors', ())) & material.anchors) >= 3]
        observed = [whole for whole in wholes.values() if whole.source == material.source
                    and whole.seen == at and whole.pixels == pixels
                    and whole.character_id == material.character_id and whole.mark_epoch == material.mark_epoch
                    and len(whole.anchors & material.anchors) >= 3]
        status = ('current_extent_measured' if len(observed) == 1 else 'ambiguous_current_extent'
                  if len(observed) > 1 else 'current_extent_refused' if attempts else 'current_material_unavailable')
        receipts.append(_receipt(material, at, status, measurements=len(attempts),
            reasons=tuple(dict.fromkeys(a['reason'] for a in attempts)),
            observed_owner=(observed[0].number, observed[0].born, observed[0].kind) if len(observed) == 1 else None))
    return tuple(receipts)


def extent_reference_to_measure(keys: tuple, covered: set, cursor: int, priority: tuple,
                                ordinary: Any) -> Any:
    """One of four existing optional slots can serve the requested family.

    Ordinary turns use a compact counter so reserving a slot cannot repeatedly
    skip the same reference. Frame variants retain their ordinary ordering.
    """
    preferred = tuple(key for key in keys if key[:2] in priority)
    if not preferred:
        return ordinary(keys, covered, cursor)
    if cursor % 4 == 0:
        return preferred[(cursor//4) % len(preferred)]
    return ordinary(keys, covered, cursor-(cursor+3)//4)


@invariant('perception.extent_demand_keeps_other_references_observable', scope='perception',
           owner='core/perception/a_source_extent_requested_in_play.py', observational=False)
def _extent_demand_invariant() -> tuple:
    keys = tuple(('source', n, 'image', 1) for n in range(7))
    ordinary = lambda all_keys, _covered, cursor: all_keys[cursor % len(all_keys)]
    chosen = [extent_reference_to_measure(keys, set(), n, (('source', 5),), ordinary) for n in range(32)]
    assert all(chosen[n] == keys[5] for n in range(0, 32, 4)) and set(chosen) == set(keys)
    assert extent_reference_to_measure(keys, set(), 3, (), ordinary) == keys[3]
    return ()
