"""Bind a requested spatial relation to current observed entities.

Names describe referents; they do not give an arbitrary object that role.
The result is a geometric destination, not evidence of contact or success.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.verify import invariant
from core.perception.names_for_observed_instances import NamesInScene
from core.agency.evidence_for_a_requested_relation import (
    ObservedRelationRegion, ReferentEvidenceNeed, observed_relation_region, referent_evidence_needed,
    observation_time, relation_region_in_its_scene,
)


@dataclass(frozen=True)
class SpatialRequest:
    """Where the directly controlled body is asked to go, in the place's words."""

    relation: str
    destination: str
    source: str


@dataclass(frozen=True)
class BoundDestination:
    """One current referent and a position for the controlled body's centre."""

    request: SpatialRequest
    thing: Any
    point: tuple[float, float]
    region: ObservedRelationRegion | None = None


@dataclass(frozen=True)
class DestinationReading:
    """The binding result and the branch that produced it, from one scene."""

    status: str
    binding: BoundDestination | None = None
    named: tuple[tuple[int | None, str], ...] = ()
    named_count: int = 0
    candidates: tuple[int | None, ...] = ()
    candidate_count: int = 0
    need: ReferentEvidenceNeed | None = None


_HOW = r'(?:safely|softly|gently|slowly|carefully)'
_REL = r'(?P<relation>on top of|onto|inside|into|on|in|above|below|beside|next to|at|to)'
_DIRECT = re.compile(
    r'\b(?:use|using)\s+(?:the\s+)?(?:(?:[\w-]+\s+){0,3}keys|arrows|mouse|pointer|controls|keyboard)\s+to\s+'
    r'(?:guide|move|steer|drive|fly|bring|lead)\s+(?P<mover>[\w\x27’ -]{1,80}?)\s+(?:' + _HOW + r'\s+)?'
    + _REL + r'\s+(?P<name>.+)$', re.I)
_SELF = re.compile(
    r'^(?:(?:please|now|then)\s+)?(?:land|dock|walk|run|fly|go|head|arrive|stand|rest|move|drive|steer)\s+'
    r'(?:' + _HOW + r'\s+)?' + _REL + r'\s+(?P<name>.+)$', re.I)
_NEGATED = re.compile(r"\b(?:not|never|don't|cannot|can't|avoid|if|unless|when|could|would|might|may)\b", re.I)
_AFTER = re.compile(r'\s+(?:and|while|before|after|without|using|with|to)\b', re.I)
_COMMON = frozenset('the a an your my its this that to of our their it them there here somewhere something'.split())
_AN_ACTION = re.compile(r'^(?:find|get|collect|win|lose|avoid|shoot|move|open|close|build|make|try|help|catch|rescue|'
                        r'protect|save|complete|finish|land|dock|jump|pick|push|pull|use)\b', re.I)


def _directions(words: str) -> set[tuple[str, str, str]]:
    found = set()
    for clause in re.split(r'[.!?;\n]+', str(words or '')):
        clause = ' '.join(clause.split()).strip()
        if not clause or len(clause) > 300 or _NEGATED.search(clause):
            continue
        match = _DIRECT.search(clause) or _SELF.fullmatch(clause)
        if match is None:
            continue
        if 'mover' in match.re.groupindex and re.search(r'\b(?:and|or|then|while|before|after|with)\b', match['mover'], re.I):
            continue
        name = _AFTER.split(match['name'], maxsplit=1)[0].strip(' ,:')
        if (not _name_words(name) or len(name) > 80
                or match['relation'].lower() == 'to' and _AN_ACTION.match(name)):
            continue
        relation = {'onto': 'on', 'on top of': 'on', 'into': 'inside', 'in': 'inside', 'to': 'at', 'next to': 'beside'}.get(
            match['relation'].lower(), match['relation'].lower())
        found.add((relation, name.casefold(), clause))
    return found


def requested_relation(words: str) -> SpatialRequest | None:
    """Read explicit motion directions; conflicting or indirect requests abstain.

    Moving another object is a task-planning problem. This reader accepts
    controls explicitly moving their body, or an imperative self-motion verb.
    An infinitive action after "to" is not a named destination.
    """
    found = _directions(words)
    meanings = {(relation, name) for relation, name, _ in found}
    if len(meanings) != 1:
        return None
    relation, name = next(iter(meanings))
    return SpatialRequest(relation, name, min(clause for _, _, clause in found))


def _name_words(name: str) -> tuple[str, ...]:
    return tuple(word for word in re.findall(r'[a-z]+', str(name or '').casefold()) if word not in _COMMON)


def _box(thing: Any) -> tuple[float, float, float, float] | None:
    try:
        raw = tuple(getattr(thing, field) for field in ('x', 'y', 'w', 'h'))
        if any(isinstance(value, bool) for value in raw):
            return None
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None
    return values if all(math.isfinite(v) for v in values) and min(values[2:]) > 0 else None


def position_for_relation(relation: str, mine: Any, target: Any) -> tuple[float, float] | None:
    """A centre destination from finite extents; containment requires enough room.

    The support and containment geometry is provisional. It cannot certify
    collision shape, orientation, safe arrival speed, or a completed task.
    """
    own, other = _box(mine), _box(target)
    if own is None or other is None:
        return None
    x, y, w, h = other
    _, _, mw, mh = own
    if relation == 'inside':
        return (x, y) if mw <= w and mh <= h else None
    if relation == 'on':
        return (x, y - (h + mh) / 2) if mw <= w else None
    if relation == 'above':
        return x, y - h / 2 - mh
    if relation == 'below':
        return x, y + h / 2 + mh
    if relation == 'beside':
        return x + math.copysign((w + mw) / 2, own[0] - x), y
    return (x, y) if relation == 'at' else None


def bind_destination(request: SpatialRequest, mine: Any, things: Sequence[Any], names: Mapping[int, str] | NamesInScene,
                     at: float | None) -> BoundDestination | None:
    """The uniquely named current object, never the nearest unnamed substitute.

    An unmeasured, stale or duplicate candidate still competes for the name.
    Dropping it could turn an ambiguous request into a false unique match.
    """
    return read_destination(request, mine, things, names, at).binding


def read_destination(request: SpatialRequest, mine: Any, things: Sequence[Any], names: Mapping[int, str] | NamesInScene,
                     at: float | None) -> DestinationReading:
    """Preserve the actual refusal without changing the binding criteria or selecting again."""
    wanted = _name_words(request.destination)
    if (not wanted or mine is None or observation_time(at) is None
            or getattr(mine, 'seen', None) != at):
        return DestinationReading('body_or_request_unavailable',
            need=referent_evidence_needed('body_or_request_unavailable', wanted, [], [], at))
    candidates, best, named, named_count, partial = [], 0, [], 0, []
    for thing in things:
        if getattr(thing, 'number', None) == getattr(mine, 'number', None):
            continue
        name = names.for_thing(thing) if isinstance(names, NamesInScene) else names.get(getattr(thing, 'kind', -1), '')
        if name:
            named_count += 1
            if len(named) < 32:
                named.append((getattr(thing, 'number', None), str(name)[:160]))
        words = _name_words(name)
        if set(wanted) & set(words):
            partial.append((thing, name))
        if wanted[-1] not in words:
            continue
        score = len(set(wanted) & set(words))
        if score < best:
            continue
        if score > best:
            candidates, best = [], score
        candidates.append(thing)
    facts = dict(named=tuple(named), named_count=named_count,
                 candidates=tuple(getattr(thing, 'number', None) for thing in candidates[:32]), candidate_count=len(candidates))
    matched = [(thing, names.for_thing(thing) if isinstance(names, NamesInScene) else names.get(getattr(thing, 'kind', -1), ''))
               for thing in candidates]
    def refused(status):
        return DestinationReading(status, need=referent_evidence_needed(status, wanted, matched, partial, at), **facts)
    if not candidates:
        return refused('no_named_referent')
    if len(candidates) != 1:
        return refused('ambiguous_named_referent')
    if best != len(set(wanted)):
        return refused('incomplete_name')
    target = candidates[0]
    if getattr(target, 'seen', None) != at:
        return refused('referent_not_current')
    point = position_for_relation(request.relation, mine, target)
    if point is None:
        return refused('relation_geometry_unavailable')
    return DestinationReading('bound', BoundDestination(request, target, point,
        observed_relation_region(request.relation, mine, target, at)), **facts)


@invariant('agency.destination_refusals_keep_the_deciding_branch', scope='agency',
           owner='core/agency/a_requested_relation_in_play.py', observational=False)
def _destination_reading_invariant() -> tuple:
    from types import SimpleNamespace

    body = SimpleNamespace(number=1, kind=1, seen=3., x=20., y=20., w=10., h=10.)
    target = SimpleNamespace(number=2, kind=2, seen=3., x=50., y=60., w=30., h=10.)
    request = SpatialRequest('on', 'blue platform', '')
    missing = read_destination(request, body, [target], {2: 'base'}, 3.)
    partial = read_destination(request, body, [target], {2: 'platform'}, 3.)
    bound = read_destination(request, body, [target], {2: 'blue platform'}, 3.)
    assert missing.status == 'no_named_referent' and missing.binding is None
    assert partial.status == 'incomplete_name' and partial.binding is None
    assert bound.status == 'bound' and bound.binding.thing is target
    return ()


def observed_names(moves: Any) -> dict[int, str] | NamesInScene:
    """Current instance names, or legacy kind names before scoped naming begins."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    seen = getattr(guide, 'seen', None)
    legacy = {kind.number: seen.as_seen(kind.number) for kind in moves.kinds} if seen is not None else {}
    return seen.instances.scene(moves, legacy) if seen is not None else legacy


def take_motion_directions(meeting: Any, words: str) -> None:
    """Retain explicit directions beside the existing arrival-speed instruction."""
    from core.agency.what_meeting_things_does import says_to_come_gently

    if words:
        meeting.spatial_request = requested_relation(words)
    meeting.told_gently = meeting.told_gently or says_to_come_gently(words)


def update_motion_directions(meeting: Any, words: str) -> None:
    """A new explicit direction replaces the old one; counters supply no revision."""
    if _directions(words):
        take_motion_directions(meeting, words)


def _scene_referents(moves: Any) -> list[Any]:
    """Keep published instances before interaction heuristics remove any."""
    observed = getattr(moves, 'things', None)
    return list(observed.values()) if isinstance(observed, Mapping) else []


def requested_target(choosing: Any) -> tuple | None:
    """Give an explicit request priority; an unbound referent waits for observation."""
    request = getattr(getattr(choosing, 'meeting', None), 'spatial_request', None)
    choosing.goal_reading = choosing.goal_stance = None
    if request is None or choosing.mine is None:
        return None
    # A spatial referent can be scenery or share her visual kind. Interaction
    # heuristics must not remove it, or remove a rival that keeps its name ambiguous.
    reading = read_destination(request, choosing.mine, _scene_referents(choosing.moves), observed_names(choosing.moves), choosing.at)
    if reading.binding is not None and reading.binding.region is not None:
        from dataclasses import replace

        reading = replace(reading, binding=replace(reading.binding,
            region=relation_region_in_its_scene(reading.binding.region, choosing.moves, choosing.at)))
    choosing.goal_reading = reading
    binding = reading.binding
    choosing.goal_binding = binding
    if binding is not None:
        choosing.goal_stance = choosing.stance(binding.thing)
    if binding is not None and choosing.goal_stance not in ('avoid', 'shoot', 'ignore'):
        return binding.point, 'meet', binding.thing
    # Stabilize where she is while the referent is missing. Unknown scenery
    # cannot become a replacement destination or a predicted contact reward.
    return (float(choosing.mine.x), float(choosing.mine.y)), 'wait', None


def requested_contact_stance(choosing: Any, thing: Any, stance: str) -> str:
    """Unknown contacts have no task reward when an explicit destination is requested."""
    if stance != 'meet' or getattr(choosing.meeting, 'spatial_request', None) is None:
        return stance
    bound = getattr(choosing, 'goal_binding', None)
    if bound is not None and bound.thing.number == thing.number:
        return stance
    if choosing.meeting.known(thing.kind) or thing.kind in choosing.meeting.told:
        return stance
    return 'ignore'


@invariant('agency.requested_destinations_require_current_unique_referents', scope='agency',
           owner='core/agency/a_requested_relation_in_play.py', observational=False)
def _requested_destination_invariant() -> tuple:
    from types import SimpleNamespace

    request = requested_relation('Use the arrow keys to guide the vessel safely onto the platform.')
    assert request is not None and request.relation == 'on'
    body = SimpleNamespace(number=1, kind=0, x=10., y=10., w=4., h=4., seen=3.)
    target = SimpleNamespace(number=2, kind=1, x=50., y=60., w=20., h=10., seen=3.)
    assert _scene_referents(SimpleNamespace(things={1: body, 2: target})) == [body, target]
    assert _scene_referents(SimpleNamespace()) == []
    assert bind_destination(request, body, [body, target], {1: 'platform'}, 3.).point == (50., 53.)
    assert bind_destination(request, body, [body, target], {}, 3.) is None
    assert bind_destination(request, body, [body, target, target], {1: 'platform'}, 3.) is None
    assert bind_destination(request, body, [body, target], {1: 'platform'}, 4.) is None
    assert requested_relation('Put the crate onto the platform') is None
    return ()
