"""Keep candidate referents, missing evidence and geometric regions distinct.

An observed named extent can supply a geometric approach region. It cannot
certify a collider, terminal speed, angle or successful contact. Partial
names remain observation candidates and never grant movement authority.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


@dataclass(frozen=True)
class ReferentCandidate:
    owner: tuple
    seen: float | None
    label: str
    matched_words: tuple[str, ...]


@dataclass(frozen=True)
class ReferentEvidenceNeed:
    operation: str
    reason: str
    at: float | None
    candidates: tuple[ReferentCandidate, ...]
    candidate_count: int
    missing_words: tuple[str, ...]


@dataclass(frozen=True)
class ObservedRelationRegion:
    """A current box-derived region for the body centre, with acceptance unknowns."""

    body_owner: tuple
    target_owner: tuple
    at: float
    relation: str
    bounds: tuple[float, float, float, float]
    basis: tuple | None = None
    evidence: str = 'current_named_extent'
    contact_verified: bool | None = None
    maximum_speed: float | None = None
    maximum_angle: float | None = None
    unknowns: tuple[str, ...] = ('collision_surface', 'arrival_speed', 'arrival_angle', 'observed_contact', 'coordinate_basis')


def _owner(thing: Any) -> tuple:
    return tuple(getattr(thing, field, None) for field in ('number', 'born', 'kind'))


def observation_time(at: Any) -> float | None:
    if isinstance(at, bool) or not isinstance(at, (int, float)):
        return None
    try:
        value = float(at)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def referent_evidence_needed(status: str, wanted: tuple, matches: list, partial: list, at: Any) -> ReferentEvidenceNeed:
    """Diagnostic truncation bounds questions; all rivals still compete in binding."""
    from core.agency.a_requested_relation_in_play import _name_words

    operations = {'body_or_request_unavailable': 'measure_current_body',
                  'no_named_referent': 'identify_whole_referent',
                  'incomplete_name': 'resolve_missing_properties',
                  'ambiguous_named_referent': 'distinguish_referents',
                  'referent_not_current': 'reacquire_referent',
                  'referent_motion_unavailable': 'measure_reference_motion',
                  'relation_geometry_unavailable': 'measure_whole_extent'}
    if status == 'body_or_request_unavailable':
        operations[status] = 'read_motion_request' if not wanted else 'recapture_current_scene' if observation_time(at) is None else 'measure_current_body'
    entries = matches or partial
    candidates = tuple(ReferentCandidate(_owner(thing), getattr(thing, 'seen', None), str(name)[:160],
                       tuple(sorted(set(wanted) & set(_name_words(name))))) for thing, name in entries[:32])
    covered = set(candidates[0].matched_words) if len(entries) == 1 else set()
    return ReferentEvidenceNeed(operations[status], status, observation_time(at),
        candidates, len(entries), tuple(sorted(set(wanted)-covered)))


def observed_relation_region(relation: str, mine: Any, target: Any, at: float) -> ObservedRelationRegion | None:
    """Publish the finite centre region from the same extents used by binding."""
    from core.agency.a_requested_relation_in_play import _box

    own, other = _box(mine), _box(target)
    owners = _owner(mine), _owner(target)
    if (observation_time(at) is None or own is None or other is None or any(type(o[0]) is not int or type(o[2]) is not int
            or not isinstance(o[1], (int, float)) or isinstance(o[1], bool) or not math.isfinite(o[1])
            or not 0 <= o[1] <= at for o in owners)
            or mine.seen != at or target.seen != at):
        return None
    x, y, w, h = other
    mw, mh = own[2:]
    if relation == 'on' and mw <= w:
        bounds = x-(w-mw)/2, y-(h+mh)/2, x+(w-mw)/2, y-(h+mh)/2
    elif relation == 'inside' and mw <= w and mh <= h:
        bounds = x-(w-mw)/2, y-(h-mh)/2, x+(w-mw)/2, y+(h-mh)/2
    else:
        return None
    return ObservedRelationRegion(*owners, float(at), relation, bounds)


def relation_region_in_its_scene(region: ObservedRelationRegion, moves: Any, at: Any) -> ObservedRelationRegion:
    """Attach the actual observer and sample basis; no camera transform is inferred."""
    from dataclasses import replace

    shape, scale = tuple(getattr(moves, 'shape', ())), getattr(moves, 'scale', None)
    if (len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
            or not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0):
        return region
    marks = getattr(moves, 'material_marks', None)
    basis = (id(moves), getattr(moves, '_instance_name_epoch', None), shape, float(scale),
             getattr(marks, 'epoch', None), getattr(at, 'source', None), getattr(at, 'surface', None), getattr(at, 'epoch', None))
    return replace(region, basis=basis, unknowns=tuple(v for v in region.unknowns if v != 'coordinate_basis'))


@invariant('agency.referent_candidates_cannot_certify_contact', scope='agency',
           owner='core/agency/evidence_for_a_requested_relation.py', observational=False)
def _referent_evidence_invariant() -> tuple:
    from types import SimpleNamespace

    mine = SimpleNamespace(number=1, born=0., kind=1, seen=3., x=10., y=10., w=10., h=8.)
    target = SimpleNamespace(number=2, born=0., kind=2, seen=3., x=60., y=70., w=40., h=10.)
    need = referent_evidence_needed('incomplete_name', ('blue', 'platform'), [(target, 'platform')], [], 3.)
    assert need.operation == 'resolve_missing_properties' and need.missing_words == ('blue',)
    region = observed_relation_region('on', mine, target, 3.)
    assert region.bounds == (45., 61., 75., 61.) and region.contact_verified is None
    assert region.maximum_speed is None and region.maximum_angle is None
    return ()
