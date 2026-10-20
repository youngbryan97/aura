"""Turn a missing destination into current instance observations to request.

Geometry orders questions; it does not identify a destination. The existing
binder still needs a unique current referent and its requested relation.
"""
from __future__ import annotations

from typing import Any

from core.agency.a_requested_relation_in_play import (
    position_for_relation, read_destination,
)
from core.perception.names_for_observed_instances import NamesInScene, query_for
from core.verify import invariant


def referents_to_observe(request: Any, moves: Any, mine: Any, names: Any, at: float) -> tuple[tuple, ...]:
    """Prioritize unresolved current instances without granting a target or control role."""
    if request is None:
        return ()
    things = list(getattr(moves, 'things', {}).values())
    reading = read_destination(request, mine, things, names, at)
    if reading.binding is not None:
        return ()
    candidates = []
    for thing in things:
        if mine is not None and getattr(thing, 'number', None) == mine.number:
            continue
        query = query_for(thing, moves, at)
        if query is None:
            continue
        name = names.for_thing(thing) if isinstance(names, NamesInScene) else names.get(thing.kind, '')
        if name:
            continue
        fits = mine is not None and position_for_relation(request.relation, mine, thing) is not None
        candidates.append((not fits, bool(getattr(thing, 'moved', False)), -thing.w * thing.h, query.key))
    # Bounded attention, without discarding rivals from the binder itself.
    return tuple(item[-1] for item in sorted(candidates)[:32])


@invariant('agency.missing_referents_request_current_instance_observation', scope='agency',
           owner='core/agency/looking_for_an_unbound_referent.py', observational=False)
def _referent_attention_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.a_requested_relation_in_play import SpatialRequest

    def body(number, x, width):
        return SimpleNamespace(number=number, born=0., kind=number, seen=3., x=x, y=30., w=width, h=10.,
                               box=lambda: (x-width/2, 25., x+width/2, 35.), moved=False)
    mine, target = body(1, 20., 10.), body(2, 50., 30.)
    moves = SimpleNamespace(things={1: mine, 2: target})
    request = SpatialRequest('on', 'platform', '')
    attention = referents_to_observe(request, moves, mine, {}, 3.)
    assert attention == (query_for(target, moves, 3.).key,)
    assert read_destination(request, mine, [target], {}, 3.).binding is None
    target.seen = 2.
    assert not referents_to_observe(request, moves, mine, {}, 3.)
    return ()
