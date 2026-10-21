"""Use a current bound relation as geometry, with physical acceptance unknown."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.agency.evidence_for_a_requested_relation import (
    ObservedRelationRegion, observed_relation_region, relation_region_in_its_scene,
)
from core.verify import invariant


def current_requested_region(choosing: Any, goal: tuple | None = None) -> ObservedRelationRegion | None:
    """Recheck owners, extents and sample basis before a plan consumes a region."""
    from core.agency.a_requested_relation_in_play import BoundDestination
    from core.perception.whole_referents_from_source_material import scene_referents

    binding = getattr(choosing, 'goal_binding', None)
    request = getattr(getattr(choosing, 'meeting', None), 'spatial_request', None)
    if (not isinstance(binding, BoundDestination) or binding.request != request
            or getattr(choosing, 'goal_stance', None) != 'meet'
            or goal is not None and tuple(goal) != binding.point
            or not isinstance(binding.region, ObservedRelationRegion)):
        return None
    moves, mine, at = choosing.moves, choosing.mine, choosing.at
    if not any(thing is binding.thing for thing in scene_referents(moves)):
        return None
    expected = observed_relation_region(request.relation, mine, binding.thing, at)
    region = binding.region
    if (expected is None or region.body_owner != expected.body_owner or region.target_owner != expected.target_owner
            or region.at != expected.at or region.relation != expected.relation or region.bounds != expected.bounds):
        return None
    current = relation_region_in_its_scene(expected, moves, at)
    if region.basis is not None and region.basis != current.basis:
        return None
    # Only the box-derived geometry is consumed. Optional acceptance fields
    # cannot turn this objective into a witnessed collider or outcome.
    return current


@dataclass(frozen=True)
class RelationInPlan:
    region: ObservedRelationRegion
    target_start: tuple[float, float]
    path: Callable[[float], tuple[float, float] | None]

    def bounds_after(self, after: float) -> tuple[float, float, float, float] | None:
        """Translate the centre region along the existing world forecast."""
        where = self.path(after)
        if where is None or len(where) != 2 or not all(math.isfinite(v) for v in where):
            return None
        dx, dy = where[0]-self.target_start[0], where[1]-self.target_start[1]
        left, top, right, bottom = self.region.bounds
        return left+dx, top+dy, right+dx, bottom+dy

    def nearest(self, position: tuple[float, float], after: float) -> tuple[float, float] | None:
        bounds = self.bounds_after(after)
        if bounds is None or not all(math.isfinite(v) for v in position):
            return None
        left, top, right, bottom = bounds
        return min(max(position[0], left), right), min(max(position[1], top), bottom)

    def distance(self, position: tuple[float, float], after: float) -> float | None:
        nearest = self.nearest(position, after)
        return math.dist(position, nearest) if nearest is not None else None


@dataclass(frozen=True)
class RelationForecast:
    """Predicted geometric errors; these fields record no observed contact."""

    region: ObservedRelationRegion
    terminal_error: float
    path_error: float
    terminal_relative_speed: float
    status: str = 'predicted_geometry'
    observed_contact: None = None
    observed_success: None = None


@invariant('agency.requested_regions_keep_current_owners_and_acceptance_unknown', scope='agency',
           owner='core/agency/a_requested_region_in_a_plan.py', observational=False)
def _planned_relation_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.a_requested_relation_in_play import SpatialRequest, read_destination

    mine = SimpleNamespace(number=1, born=0., kind=1, seen=3., x=90., y=60., w=10., h=10.)
    target = SimpleNamespace(number=2, born=0., kind=2, seen=3., x=140., y=60., w=100., h=40.)
    request = SpatialRequest('inside', 'panel', '')
    reading = read_destination(request, mine, [target], {2: 'panel'}, 3.)
    choosing = SimpleNamespace(mine=mine, moves=SimpleNamespace(things={1: mine, 2: target}), at=3.,
        meeting=SimpleNamespace(spatial_request=request), goal_binding=reading.binding, goal_stance='meet')
    region = current_requested_region(choosing, reading.binding.point)
    assert region is not None and region.contact_verified is None and region.maximum_speed is None
    objective = RelationInPlan(region, (140., 60.), lambda _after: (140., 60.))
    assert objective.distance((90., 60.), 0.) == 5.
    assert objective.distance((100., 60.), 0.) == 0.
    choosing.at = 4.
    assert current_requested_region(choosing) is None
    return ()
