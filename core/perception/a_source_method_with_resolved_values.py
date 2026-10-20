"""Evaluate a bounded source method against explicitly owned primitive values.

The existing symbolic reader owns instruction order and arithmetic semantics.
This resolves its references and selects one path. Unresolved calls, property
deletions and unknown guards remain evidence requests. A resolved arithmetic
step grants neither visual identity nor live control or outcome authority.
"""
from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from core.perception.symbolic_motion_in_flash_methods import (
    Arithmetic, BranchEvidence, FlashMethodReading, SourceInvocation, SourceWrite,
    _boolean, _calculation, _expression_is_bounded, _literal,
)
from core.verify import invariant


@dataclass(frozen=True)
class SourceValues:
    """Supplied primitive bindings; these do not certify a game's heap or pixels."""
    source: str
    owner: Hashable
    at: float
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (not isinstance(self.source, str) or not 1 <= len(self.source) <= 512 or self.owner is None
                or not _finite_number(self.at)
                or self.at < 0 or not isinstance(self.values, Mapping) or len(self.values) > 1024):
            raise ValueError('source values need a bounded source, owner, time and primitive mapping')
        try:
            hash(self.owner)
        except TypeError as why:
            raise ValueError('source values need a hashable owner') from why
        normalized = {}
        for name, value in self.values.items():
            if (not isinstance(name, str) or len(name) > 512
                    or re.fullmatch(r'[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*', name) is None):
                raise ValueError('source values need explicit qualified field names')
            if isinstance(value, str) and len(value) > 4096:
                raise ValueError('source strings exceed the expression budget')
            try:
                normalized[name] = _literal(value)
            except (ValueError, OverflowError) as why:
                raise ValueError('source values require finite primitive values') from why
        object.__setattr__(self, 'values', MappingProxyType(normalized))


@dataclass(frozen=True)
class SourceValueNeed:
    operation: str
    target: str
    source: str
    offset: int | None = None


@dataclass(frozen=True)
class ResolvedSourceStep:
    source: str
    method: str
    owner: Hashable
    at: float
    writes: tuple[SourceWrite, ...] = ()
    returned: Arithmetic | None = None
    needs: tuple[SourceValueNeed, ...] = ()
    invocation: SourceInvocation | None = None
    conditions: tuple[BranchEvidence, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.returned is not None and not self.needs


class _ValueUnavailable(ValueError):
    pass


def _finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


_ARITY = {**dict.fromkeys(('not', 'neg', 'cos', 'sin', 'sqrt', 'abs', 'round', 'int'), 1),
          **dict.fromkeys(('add', 'sub', 'mul', 'div', 'mod', 'lt', 'gt', 'atan2',
                           'eq', 'strict eq', 'legacy eq'), 2)}


def _primitive(expression: Arithmetic) -> bool:
    args = expression.arguments
    if expression.operation in ('null', 'undefined'):
        return not args
    if len(args) != 1:
        return False
    return {'number': _finite_number(args[0]), 'boolean': type(args[0]) is bool,
            'string': isinstance(args[0], str) and len(args[0]) <= 4096}.get(expression.operation, False)


def _bounded(expression: Arithmetic) -> bool:
    try:
        return _expression_is_bounded(expression)
    except OverflowError:
        return False


class _Resolve:
    def __init__(self, values: SourceValues) -> None:
        self.values = values
        self.memo: dict[int, Arithmetic] = {}
        self.visited = 0

    def value(self, expression: Arithmetic, depth: int = 0) -> Arithmetic:
        if not isinstance(expression, Arithmetic) or depth > 32:
            raise _ValueUnavailable('bounded_expression_required')
        cached = self.memo.get(id(expression))
        if cached is not None:
            return cached
        self.visited += 1
        if self.visited > 4096:
            raise _ValueUnavailable('expression_work_budget_exhausted')
        operation = expression.operation
        if operation in ('number', 'boolean', 'string', 'null', 'undefined'):
            if not _primitive(expression):
                raise _ValueUnavailable('finite_primitive_required')
            value = expression
        elif operation == 'reference':
            if len(expression.arguments) != 1 or not isinstance(expression.arguments[0], str):
                raise _ValueUnavailable('qualified_reference_required')
            name = expression.arguments[0]
            if name not in self.values.values:
                raise _ValueUnavailable(str(name))
            value = self.values.values[name]
        elif (_ARITY.get(operation) == len(expression.arguments)
              and all(isinstance(a, Arithmetic) for a in expression.arguments)):
            value = _calculation(operation, *(self.value(a, depth+1) for a in expression.arguments))
            if not _primitive(value):
                raise _ValueUnavailable('resolve_arithmetic:'+operation)
        else:
            raise _ValueUnavailable('resolve_arithmetic:'+operation)
        self.memo[id(expression)] = value
        return value


def resolve_source_method(reading: FlashMethodReading, values: SourceValues) -> ResolvedSourceStep:
    """Return one complete conditional arithmetic step or the exact missing evidence."""
    base = dict(source=values.source, method=reading.target, owner=values.owner, at=float(values.at),
                invocation=reading.invocation)
    if not reading.complete:
        return ResolvedSourceStep(**base, needs=(SourceValueNeed('complete_source_method', reading.target, values.source),))
    evaluator = _Resolve(values)
    possible, unresolved = [], []
    for path in reading.paths:
        mismatched, unknown = False, []
        for condition in path.conditions:
            if not _same_source(condition.source, values.source):
                return ResolvedSourceStep(**base, needs=(SourceValueNeed('bind_current_source', reading.target, condition.source),))
            try:
                truth = _boolean(evaluator.value(condition.condition))
                if truth is None:
                    raise _ValueUnavailable('resolve_guard_truth')
                mismatched |= truth != condition.taken
            except _ValueUnavailable as why:
                unknown.append(SourceValueNeed('observe_source_value', str(why), condition.source, condition.offset))
        if mismatched:
            continue
        if unknown:
            unresolved.extend(unknown)
        else:
            possible.append(path)
    if unresolved or len(possible) != 1:
        needs = unresolved or [SourceValueNeed('distinguish_source_path', reading.target, values.source)]
        return ResolvedSourceStep(**base, needs=tuple(dict.fromkeys(needs))[:64])
    path = possible[0]
    needs = [SourceValueNeed('resolve_call_effects', c.target, c.source, c.offset) for c in path.calls]
    needs.extend(SourceValueNeed('resolve_property_deletion', d.target, d.source, d.offset) for d in path.deletions)
    for name, expected in path.assumptions:
        supplied = values.values.get(name)
        if supplied is None or supplied != _literal(expected):
            needs.append(SourceValueNeed('verify_assumed_value', name, values.source))
    if needs:
        return ResolvedSourceStep(**base, needs=tuple(needs[:64]))
    writes = []
    for write in path.writes:
        if not _same_source(write.source, values.source):
            needs.append(SourceValueNeed('bind_current_source', write.target, write.source, write.offset))
            continue
        try:
            if not _bounded(write.value):
                raise _ValueUnavailable('bounded_expression_required')
            writes.append(replace(write, value=evaluator.value(write.value)))
        except _ValueUnavailable as why:
            needs.append(SourceValueNeed('observe_source_value', str(why), write.source, write.offset))
    try:
        if not _bounded(path.returned):
            raise _ValueUnavailable('bounded_expression_required')
        returned = evaluator.value(path.returned)
    except _ValueUnavailable as why:
        needs.append(SourceValueNeed('observe_source_value', str(why), values.source))
        returned = None
    return ResolvedSourceStep(**base, writes=tuple(writes) if not needs else (),
                              returned=returned if not needs else None, needs=tuple(dict.fromkeys(needs))[:64],
                              conditions=path.conditions)


def _same_source(stream: str, source: str) -> bool:
    return stream == source or stream.startswith(source+'/')


@invariant('perception.resolved_source_steps_keep_unknown_values_explicit', scope='perception',
           owner='core/perception/a_source_method_with_resolved_values.py', observational=False)
def _resolved_source_step_invariant() -> tuple:
    from core.perception.symbolic_motion_in_flash_methods import MethodPath, SourceWrite

    value = Arithmetic('add', (Arithmetic('reference', ('body.v',)), Arithmetic('number', (2.,))))
    method = FlashMethodReading('move', (MethodPath((), (SourceWrite('body.v', value, 'source', 1),), (),
                                                   Arithmetic('undefined')),))
    absent = resolve_source_method(method, SourceValues('source', ('body', 1), 3., {}))
    known = resolve_source_method(method, SourceValues('source', ('body', 1), 3., {'body.v': 4.}))
    assert not absent.resolved and absent.needs[0].target == 'body.v' and not absent.writes
    assert known.resolved and known.writes[0].target == 'body.v'
    assert known.writes[0].value == Arithmetic('number', (6.,)) and known.writes[0].offset == 1
    return ()
