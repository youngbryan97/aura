"""Keep a visual name with the tracked instance actually shown to a provider."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import numpy as np

from core.verify import invariant


@dataclass(frozen=True)
class InstanceQuery:
    observer: int
    epoch: str
    number: int
    born: float
    kind: int
    at: float
    box: tuple[float, float, float, float]
    picture_sha256: str = ''

    def __post_init__(self) -> None:
        if (type(self.observer) is not int or self.observer <= 0 or not isinstance(self.epoch, str)
                or not re.fullmatch('[0-9a-f]{32}', self.epoch)
                or type(self.number) is not int or self.number < 0 or type(self.kind) is not int or self.kind < 0
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.born, self.at))
                or not 0 <= self.born <= self.at or not isinstance(self.box, (tuple, list)) or len(self.box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.box)
                or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]
                or not isinstance(self.picture_sha256, str)
                or self.picture_sha256 and not re.fullmatch('[0-9a-f]{64}', self.picture_sha256)):
            raise ValueError('an instance query needs a finite capture, tracked birth and coordinate box')
        object.__setattr__(self, 'box', tuple(self.box))

    @property
    def key(self) -> tuple:
        return self.observer, self.epoch, self.number, self.born, self.kind

    def with_picture(self, picture: np.ndarray) -> InstanceQuery:
        if (not isinstance(picture, np.ndarray) or picture.dtype != np.uint8 or picture.ndim != 3
                or picture.shape[2] != 3 or min(picture.shape[:2]) < 1 or max(picture.shape[:2]) > 4096):
            raise ValueError('an instance query needs bounded measured RGB bytes')
        digest = hashlib.sha256(str(picture.shape).encode()+picture.tobytes()).hexdigest()
        return replace(self, picture_sha256=digest)


def _scope(observer: Any) -> tuple[int, str]:
    epoch = getattr(observer, '_instance_name_epoch', None)
    if not isinstance(epoch, str) or not re.fullmatch('[0-9a-f]{32}', epoch):
        epoch = uuid4().hex
        setattr(observer, '_instance_name_epoch', epoch)
    return id(observer), epoch


def query_for(thing: Any, observer: Any, at: float) -> InstanceQuery | None:
    values = tuple(getattr(thing, key, None) for key in ('number', 'born', 'kind', 'seen'))
    number, born, kind, seen = values
    if (type(number) is not int or number < 0 or type(kind) is not int or kind < 0
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (born, seen, at))
            or not 0 <= born <= at or seen != at):
        return None
    if not callable(getattr(thing, 'box', None)):
        return None
    box = tuple(thing.box())
    if (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        return None
    return InstanceQuery(*_scope(observer), number, born, kind, at, box)


@dataclass(frozen=True)
class InstanceName:
    query: InstanceQuery
    what: str
    odd: str = ''
    who: str = ''


@dataclass(frozen=True)
class NamesInScene:
    """Names resolved for current instances, or legacy class names before scoping."""

    scoped: bool
    names: tuple[tuple[int, float, int, str], ...]
    legacy: tuple[tuple[int, str], ...] = ()

    def for_thing(self, thing: Any) -> str:
        if self.scoped:
            if (type(getattr(thing, 'number', None)) is not int or type(getattr(thing, 'kind', None)) is not int
                    or type(getattr(thing, 'born', None)) not in (int, float)):
                return ''
            key = thing.number, getattr(thing, 'born', None), thing.kind
            return next((name for number, born, kind, name in self.names if (number, born, kind) == key), '')
        return dict(self.legacy).get(thing.kind, '')


class InstanceNames:
    """Bounded semantic guesses; no name can select a controlled body."""

    def __init__(self) -> None:
        self.records: dict[tuple, InstanceName] = {}
        self.observers: list[tuple[int, str]] = []
        self._owner = uuid4().hex

    def begin(self, observer: Any) -> None:
        scope = _scope(observer)
        setattr(observer, '_instance_names_scoped', self._owner)
        if scope not in self.observers:
            self.observers.append(scope)
        while len(self.observers) > 16:
            retired = self.observers.pop(0)
            self.records = {key: value for key, value in self.records.items() if key[:2] != retired}

    def active(self, observer: Any) -> bool:
        # Eviction removes labels, not the requirement for instance ownership.
        return getattr(observer, '_instance_names_scoped', None) == self._owner

    def record_for(self, observer: Any, query: InstanceQuery, what: str, odd: str = '', who: str = '') -> bool:
        if not isinstance(query, InstanceQuery) or (query.observer, query.epoch) != _scope(observer):
            return False
        current = getattr(observer, 'things', {}).get(query.number)
        if (current is None or getattr(current, 'born', None) != query.born
                or getattr(current, 'kind', None) != query.kind
                or type(getattr(current, 'seen', None)) not in (int, float)
                or not math.isfinite(current.seen) or current.seen < query.at):
            return False
        return self.record(query, what, odd, who)

    def record(self, query: InstanceQuery, what: str, odd: str = '', who: str = '') -> bool:
        if (not isinstance(query, InstanceQuery) or (query.observer, query.epoch) not in self.observers
                or not re.fullmatch('[0-9a-f]{64}', query.picture_sha256)
                or not isinstance(what, str) or not what or len(what) > 160
                or not isinstance(odd, str) or not isinstance(who, str) or len(who) > 80 or len(odd) > 160):
            return False
        key = query.observer, query.epoch, query.number, query.born
        old = self.records.get(key)
        if old is not None and old.query.at > query.at:
            return False
        if old is not None and old.query.at == query.at and (old.what, old.who) != (what, who):
            what = ''  # Conflicting labels of one capture remain unresolved.
        self.records[key] = InstanceName(query, what, odd, who)
        while len(self.records) > 256:
            self.records.pop(next(iter(self.records)))
        return bool(what)

    def of(self, observer: Any, thing: Any) -> InstanceName | None:
        if (type(getattr(thing, 'number', None)) is not int or type(getattr(thing, 'kind', None)) is not int
                or type(getattr(thing, 'born', None)) not in (int, float)):
            return None
        found = self.records.get((*_scope(observer), thing.number, thing.born))
        at = getattr(thing, 'seen', None)
        return (found if found is not None and found.what and found.query.kind == thing.kind
                and type(at) in (int, float) and math.isfinite(at) and at >= found.query.at else None)

    def scene(self, observer: Any, legacy: dict[int, str]) -> NamesInScene:
        names = []
        for thing in (getattr(observer, 'things', {}) or {}).values():
            found = self.of(observer, thing)
            if found is not None:
                names.append((thing.number, thing.born, thing.kind, found.who or found.what))
        return NamesInScene(self.active(observer), tuple(names), tuple(legacy.items()))

    def consensus(self, observer: Any, kind: int) -> InstanceName | None:
        peers = [thing for thing in observer.things.values() if thing.kind == kind]
        labels = [self.of(observer, thing) for thing in peers]
        if not labels or any(label is None for label in labels):
            return None
        return labels[0] if len({(label.what, label.who) for label in labels}) == 1 else None


@invariant('perception.visual_names_keep_the_queried_instance', scope='perception',
           owner='core/perception/names_for_observed_instances.py', observational=False)
def _instance_name_invariant() -> tuple:
    from types import SimpleNamespace

    first = SimpleNamespace(number=1, born=1., kind=0, seen=2., box=lambda: (0., 0., 10., 10.))
    other = SimpleNamespace(number=2, born=1., kind=0, seen=2., box=lambda: (20., 0., 30., 10.))
    observer = SimpleNamespace(things={1: first, 2: other})
    names = InstanceNames()
    names.begin(observer)
    query = query_for(first, observer, 2.)
    assert query is not None and not names.record(query, 'vessel')
    assert names.record(query.with_picture(np.zeros((16, 16, 3), np.uint8)), 'vessel')
    assert names.scene(observer, {0: 'vessel'}).for_thing(first) == 'vessel'
    assert names.scene(observer, {0: 'vessel'}).for_thing(other) == ''
    assert names.consensus(observer, 0) is None
    return ()
