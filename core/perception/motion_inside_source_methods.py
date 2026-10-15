"""Bounded arithmetic findings from the methods a program's keys actually call.

Names do not tell this reader what moves. A pair of velocity assignments that
adds sine and cosine of the same angle does; a modular sum written to an
angle does. Decompiled branch offsets without their instruction addresses do
not say which alternative reaches an assignment. Such findings retain their
alternatives and conditions, and are never unconditional controller models.

This is a source reading, not player identification or execution of source.
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.verify import invariant

_REF = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\Z")
_ASSIGN = re.compile(r"^(?:var\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*=\s*(.+?)\s*;?$")
_FUNCTION = re.compile(r"^([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*=\s*function\b")
_ANON = re.compile(r"^\s*// anonymous function\s*\(([^)]*)\)")
_GUARD = re.compile(r"^if\s+(.+?)\s+goto\s+[+-]\d+\s*$")
_MAX_LINES = 40_000
_MAX_METHOD_LINES = 256
_MAX_ALTERNATIVES = 8


@dataclass(frozen=True)
class SourcePredicate:
    expression: str
    line: int
    reason: str = "branch target address was not retained"
    offset: int | None = None
    source: str = ''


@dataclass(frozen=True)
class MethodMotion:
    """An arithmetic effect, with source uncertainty kept beside its numbers.

    ``steps`` are possible signed increments per program frame. For a vector
    increment they are field units/frame; for a turn they are radians/frame.
    ``rates`` convert these to field units/second or radians/second. A vector
    increment is not acceleration until its fields are linked to position.
    A nonempty ``unknown`` prevents this from being an unconditional model.
    """

    key: str
    method: str
    kind: str
    body: str
    heading: str
    changed: tuple[str, ...]
    steps: tuple[float, ...]
    rates: tuple[float, ...]
    predicates: tuple[SourcePredicate, ...]
    unknown: tuple[str, ...]
    lines: tuple[int, ...]
    offsets: tuple[int, ...] = ()

    @property
    def unconditional(self) -> bool:
        return self.kind != 'vector_increment' and not self.predicates and not self.unknown


@dataclass(frozen=True)
class DampedRecovery:
    """A measured-source recurrence, not a claim that recovery was observed."""

    retained_per_frame: float
    stopping_speed: float
    frames_a_second: float
    lines: tuple[int, ...]
    predicates: tuple[SourcePredicate, ...] = ()

    def __post_init__(self) -> None:
        if (any(type(value) not in (float, int) or not math.isfinite(value) for value in
                (self.retained_per_frame, self.stopping_speed, self.frames_a_second))
                or not 0 < self.retained_per_frame < 1 or self.stopping_speed <= 0
                or not 0 < self.frames_a_second <= 1000):
            raise ValueError('a damped recovery requires measured finite decay, stop and cadence')

    def frames_below(self, speed: float) -> int | None:
        """First frame for which speed * retained**n is strictly below stop."""
        if not math.isfinite(speed) or speed < 0:
            return None
        if speed < self.stopping_speed:
            return 0
        # Strict comparison matters when speed is exactly the threshold.
        frames = max(1, math.floor((math.log(self.stopping_speed) - math.log(speed)) /
                                   math.log(self.retained_per_frame)) + 1)
        while speed * self.retained_per_frame ** frames >= self.stopping_speed:
            frames += 1
        return frames

    def seconds_below(self, speed: float) -> float | None:
        frames = self.frames_below(speed)
        return None if frames is None else frames / self.frames_a_second


@dataclass(frozen=True)
class SourceMethodMotion:
    effects: tuple[MethodMotion, ...] = ()
    recoveries: tuple[DampedRecovery, ...] = ()
    constants: tuple[tuple[str, float], ...] = ()
    unknown: tuple[str, ...] = ()
    source: str = ''


@dataclass(frozen=True)
class _E:
    op: str
    args: tuple[Any, ...]


@dataclass
class _Method:
    target: str
    params: tuple[str, ...]
    lines: list[tuple[int, str]]


@dataclass
class _Read:
    method: _Method
    updates: dict[str, list[tuple[_E, int]]] = field(default_factory=dict)
    predicates: list[SourcePredicate] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    constant_reads: set[str] = field(default_factory=set)


def _methods(code: list[str]) -> list[_Method]:
    """Keep a qualified method target instead of dropping its receiver."""
    found = []
    for start, line in enumerate(code):
        marker = _ANON.match(line)
        if marker is None:
            continue
        indent = len(line) - len(line.lstrip())
        stop = start + 1
        while (stop < len(code) and stop - start <= _MAX_METHOD_LINES
               and (not code[stop].strip() or len(code[stop]) - len(code[stop].lstrip()) >= indent)):
            stop += 1
        target = _FUNCTION.match(code[stop].strip()) if stop < len(code) else None
        if target is not None and stop - start <= _MAX_METHOD_LINES:
            found.append(_Method(target[1], tuple(p.strip() for p in marker[1].split(',') if p.strip()),
                                 [(i + 1, code[i].strip()) for i in range(start + 1, stop)]))
    return found


def _path(node: ast.AST, owner: str) -> str | None:
    if isinstance(node, ast.Name):
        return owner if node.id == 'this' else node.id
    if isinstance(node, ast.Attribute):
        base = _path(node.value, owner)
        return f'{base}.{node.attr}' if base else None
    return None


def _parse(text: str) -> ast.AST | None:
    if len(text) > 512 or '$' in text:
        return None
    try:
        node = ast.parse(text.rstrip(';'), mode='eval').body
        return node if sum(1 for _ in ast.walk(node)) <= 80 else None
    except (SyntaxError, ValueError, RecursionError):
        return None


def _numeric(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (float, int) and math.isfinite(node.value):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        number = _numeric(node.operand)
        return (-number if isinstance(node.op, ast.USub) else number) if number is not None else None
    return None


def _constants(methods: list[_Method]) -> dict[str, float]:
    values: dict[str, set[float]] = {}
    changed = set()
    for method in methods:
        owner = method.target.rsplit('.', 1)[0]
        for _line, text in method.lines:
            match = _ASSIGN.match(text)
            if match is None or '.' not in match[1] or not match[1].startswith(('this.', owner + '.')):
                continue
            path = owner + match[1][4:] if match[1].startswith('this.') else match[1]
            number = _numeric(_parse(match[2]))
            if number is None:
                changed.add(path)
            else:
                values.setdefault(path, set()).add(number)
    return {path: next(iter(v)) for path, v in values.items() if len(v) == 1 and path not in changed}


def _unique(values: Iterable[_E]) -> tuple[_E, ...]:
    return tuple(dict.fromkeys(values))[:_MAX_ALTERNATIVES]


def _fold(op: str, args: tuple[_E, ...]) -> _E:
    if all(a.op == 'number' for a in args):
        a = args[0].args[0]
        b = args[1].args[0] if len(args) > 1 else None
        try:
            value = {'add': lambda: a + b, 'sub': lambda: a - b, 'mul': lambda: a * b,
                     'div': lambda: a / b, 'mod': lambda: a % b, 'neg': lambda: -a}[op]()
            if math.isfinite(value):
                return _E('number', (float(value),))
        except (KeyError, ZeroDivisionError, OverflowError):
            pass
    return _E(op, args)


def _expressions(node: ast.AST | None, owner: str, env: dict[str, tuple[_E, ...]],
                 constants: Mapping[str, float], depth: int = 0) -> tuple[_E, ...]:
    if node is None or depth > 12:
        return ()
    number = _numeric(node)
    if number is not None:
        return (_E('number', (number,)),)
    path = _path(node, owner)
    if path is not None:
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Attribute):
            bases = _expressions(node.value, owner, env, constants, depth + 1)
            if bases and all(e.op == 'ref' for e in bases):
                paths = [e.args[0] + '.' + node.attr for e in bases]
                return tuple(_E('number', (constants[p],)) if p in constants else _E('ref', (p,)) for p in paths)
        return (_E('number', (constants[path],)) if path in constants else _E('ref', (path,)),)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return tuple(_fold('neg', (v,)) for v in _expressions(node.operand, owner, env, constants, depth + 1))
    operators = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div', ast.Mod: 'mod'}
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        left = _expressions(node.left, owner, env, constants, depth + 1)
        right = _expressions(node.right, owner, env, constants, depth + 1)
        if len(left) * len(right) > _MAX_ALTERNATIVES:
            return ()
        return _unique(_fold(operators[type(node.op)], (a, b)) for a in left for b in right)
    if isinstance(node, ast.Call) and len(node.args) == 1:
        callee = _path(node.func, owner)
        if callee in ('Math.cos', 'Math.sin', 'Math.sqrt', 'Math.abs'):
            op = callee.split('.')[-1]
            return tuple(_E(op, (a,)) for a in _expressions(node.args[0], owner, env, constants, depth + 1))
    return ()


def _read(method: _Method, args: tuple[str, ...], constants: Mapping[str, float]) -> _Read:
    owner = method.target.rsplit('.', 1)[0]
    env = {name: values for name, value in zip(method.params, args, strict=False)
           if (values := _expressions(_parse(value), owner, {}, constants))}
    read = _Read(method)
    for argument in args:
        if argument in constants:
            read.constant_reads.add(argument)
    branched = False
    for line, text in method.lines:
        guard = _GUARD.match(text)
        if guard is not None:
            branched = True
            read.predicates.append(SourcePredicate(guard[1].replace('this.', owner + '.'), line))
            continue
        if text.startswith(('goto ', 'return')):
            read.unknown.append(f'unresolved source control transfer at line {line}')
            continue
        match = _ASSIGN.match(text)
        if match is None:
            if '(' in text and not text.startswith('//'):
                read.unknown.append(f'uncalled or unsupported side effect at line {line}: {text[:100]}')
            continue
        node = _parse(match[2])
        if node is not None:
            for part in ast.walk(node):
                if (path := _path(part, owner)) in constants:
                    read.constant_reads.add(path)
        values = _expressions(node, owner, env, constants)
        if not values:
            read.unknown.append(f'unsupported assignment at line {line}')
            continue
        target = _parse(match[1])
        if isinstance(target, ast.Name):
            previous = env.get(target.id, ()) if branched else ()
            combined = _unique((*previous, *values))
            if len(previous) + len(values) > _MAX_ALTERNATIVES:
                read.unknown.append(f'too many branch alternatives at line {line}')
            env[target.id] = combined
        else:
            paths = _expressions(target, owner, env, {})
            for path in paths:
                if path.op == 'ref':
                    read.updates.setdefault(path.args[0], []).extend((value, line) for value in values)
    return read


def _scale(expression: _E) -> tuple[float, _E]:
    if expression.op == 'neg':
        coefficient, term = _scale(expression.args[0])
        return -coefficient, term
    if expression.op == 'mul':
        a, b = expression.args
        if a.op == 'number':
            coefficient, term = _scale(b)
            return a.args[0] * coefficient, term
        if b.op == 'number':
            coefficient, term = _scale(a)
            return b.args[0] * coefficient, term
    if expression.op == 'div' and expression.args[1].op == 'number' and expression.args[1].args[0]:
        coefficient, term = _scale(expression.args[0])
        return coefficient / expression.args[1].args[0], term
    return 1., expression


def _heading(expression: _E) -> tuple[str, float] | None:
    scale, term = _scale(expression)
    return (term.args[0], scale) if term.op == 'ref' and math.isfinite(scale) and scale != 0 else None


def _added(expression: _E, path: str) -> _E | None:
    if expression.op == 'add':
        a, b = expression.args
        if a == _E('ref', (path,)):
            return b
        if b == _E('ref', (path,)):
            return a
    if expression.op == 'sub' and expression.args[0] == _E('ref', (path,)):
        return _fold('neg', (expression.args[1],))
    return None


def _drive_effects(read: _Read) -> list[tuple[str, str, tuple[str, ...], tuple[float, ...], tuple[int, ...]]]:
    candidates = []
    for path, updates in read.updates.items():
        for expression, line in updates:
            added = _added(expression, path)
            if added is None:
                continue
            coefficient, term = _scale(added)
            if term.op not in ('cos', 'sin') or not (heading := _heading(term.args[0])):
                continue
            if not math.isfinite(coefficient) or not coefficient:
                continue
            candidates.append((path, term.op, heading, coefficient, line))
    out = []
    for path, trig, heading, _coefficient, _line in candidates:
        if trig != 'cos':
            continue
        body = path.rsplit('.', 1)[0]
        pairs = [(a, b) for a in candidates for b in candidates
                 if a[0] == path and b[1] == 'sin' and b[0].rsplit('.', 1)[0] == body
                 and a[2] == b[2] == heading and math.isclose(a[3], b[3], abs_tol=1e-10)]
        if pairs:
            paths = tuple(sorted({a[0] for pair in pairs for a in pair}))
            steps = tuple(sorted({a[3] for pair in pairs for a in pair}))
            lines = tuple(sorted({a[4] for pair in pairs for a in pair}))
            finding = body, heading[0], paths, steps, lines
            if finding not in out:
                out.append(finding)
    return out


def _trig_headings(read: _Read) -> dict[str, tuple[float, ...]]:
    found: dict[tuple[str, float], set[str]] = {}
    def visit(expression: _E) -> None:
        if expression.op in ('cos', 'sin') and (heading := _heading(expression.args[0])):
            found.setdefault(heading, set()).add(expression.op)
        for arg in expression.args:
            if isinstance(arg, _E):
                visit(arg)
    for updates in read.updates.values():
        for expression, _line in updates:
            visit(expression)
    return {path: tuple(scale for (name, scale), trigs in found.items()
                        if name == path and trigs == {'cos', 'sin'})
            for path, _scale in found}


def _turn_effects(read: _Read, known_headings: Mapping[str, tuple[float, ...]]) -> list[
        tuple[str, str, tuple[str, ...], tuple[float, ...], tuple[int, ...]]]:
    out = []
    for path, updates in read.updates.items():
        increments: dict[str, list[tuple[float, int]]] = {}
        for expression, line in updates:
            if expression.op != 'mod' or expression.args[1].op != 'number':
                continue
            period = expression.args[1].args[0]
            inner = expression.args[0]
            if period <= 0 or inner.op not in ('add', 'sub'):
                continue
            a, b = inner.args
            if a.op != 'ref' or b.op != 'number' or a.args[0].rsplit('.', 1)[0] != path.rsplit('.', 1)[0]:
                continue
            if not any(math.isclose(abs(scale * period), 2 * math.pi, rel_tol=1e-4)
                       for scale in known_headings.get(a.args[0], ())):
                continue
            step = b.args[0] * (-1 if inner.op == 'sub' else 1)
            step = ((step + period / 2) % period - period / 2) * (2 * math.pi / period)
            if step and math.isfinite(step):
                increments.setdefault(a.args[0], []).append((step, line))
        for heading, values in increments.items():
            out.append((path.rsplit('.', 1)[0], heading, (path,),
                        tuple(sorted({v for v, _ in values})), tuple(sorted({line for _, line in values}))))
    return out


def _caller_predicates(methods: list[_Method], by: str) -> tuple[SourcePredicate, ...]:
    found = []
    for method in methods:
        for line, text in method.lines:
            if text == by:
                # Flat relative jumps cannot establish dominance. Keep preceding
                # non-key predicates as unqualified context, never assert them.
                for number, previous in method.lines:
                    if number >= line:
                        break
                    guard = _GUARD.match(previous)
                    if guard is not None and 'Key.' not in guard[1]:
                        found.append(SourcePredicate(guard[1], number, 'caller dominance is unknown'))
    return tuple(dict.fromkeys(found))


def _recoveries(methods: list[_Method], constants: Mapping[str, float], fps: float) -> tuple[DampedRecovery, ...]:
    """Find a vector's common multiplicative decay and its norm stop test."""
    found = []
    for method in methods:
        read = _read(method, (), constants)
        decays: dict[str, list[tuple[float, int]]] = {}
        for path, updates in read.updates.items():
            for expression, line in updates:
                coefficient, term = _scale(expression)
                if term == _E('ref', (path,)) and 0 < coefficient < 1:
                    decays.setdefault(path, []).append((coefficient, line))
        for line, text in method.lines:
            guard = _GUARD.match(text)
            if guard is None:
                continue
            owner = method.target.rsplit('.', 1)[0]
            condition = _parse(guard[1].lstrip('!'))
            if not isinstance(condition, ast.Compare) or len(condition.ops) != 1 or not isinstance(condition.ops[0], ast.Lt):
                continue
            left = _expressions(condition.left, owner, {}, constants)
            right = _expressions(condition.comparators[0], owner, {}, constants)
            if len(left) != 1 or len(right) != 1 or right[0].op != 'number' or right[0].args[0] <= 0:
                continue
            norm = left[0]
            if norm.op != 'sqrt' or norm.args[0].op != 'add':
                continue
            paths = []
            for term in norm.args[0].args:
                if term.op == 'mul' and term.args[0] == term.args[1] and term.args[0].op == 'ref':
                    paths.append(term.args[0].args[0])
            if len(set(paths)) != 2 or any(path not in decays for path in paths):
                continue
            for a in decays[paths[0]]:
                for b in decays[paths[1]]:
                    if a[0] != b[0]:
                        continue
                    finding = DampedRecovery(a[0], right[0].args[0], fps, tuple(sorted({line, a[1], b[1]})),
                                             tuple(read.predicates))
                    if finding not in found:
                        found.append(finding)
    return tuple(found)


def motion_in_source_methods(code: Iterable[str], keys: Mapping[str, Any], frames_a_second: float | None, *,
                             source: str = '') -> SourceMethodMotion:
    """Read bounded source arithmetic; unsupported control flow stays unknown.

    ``keys`` is the parser's key-to-call reading (values expose ``by``), so
    arbitrary keys and renamed methods work without semantic name matching.
    No finding alone qualifies a controller, identity or a physical action.
    """
    if (type(frames_a_second) not in (float, int) or not math.isfinite(frames_a_second)
            or not 0 < frames_a_second <= 1000):
        return SourceMethodMotion(unknown=('unmeasured program frame rate',))
    lines, total_bytes = [], 0
    for index, line in enumerate(code):
        if index >= _MAX_LINES:
            return SourceMethodMotion(unknown=('source exceeds bounded method reader',))
        if not isinstance(line, str):
            return SourceMethodMotion(unknown=('source contains a non-text statement',))
        total_bytes += len(line)
        if len(line) > 4096 or total_bytes > 4 * 1024 * 1024:
            return SourceMethodMotion(unknown=('source text exceeds bounded method reader',))
        lines.append(line)
    methods = _methods(lines)
    constants = _constants(methods)
    counts = Counter(method.target for method in methods)
    by_target = {method.target: method for method in methods if counts[method.target] == 1}
    readings = []
    for key, what in list(keys.items())[:64]:
        by = getattr(what, 'by', '')
        node = _parse(by) if isinstance(by, str) else None
        if not isinstance(key, str) or not isinstance(node, ast.Call):
            continue
        target = _path(node.func, '')
        method = by_target.get(target)
        if method is None or len(node.args) != len(method.params):
            continue
        args = tuple(ast.unparse(arg) for arg in node.args)
        read = _read(method, args, constants)
        predicates = tuple(dict.fromkeys((*_caller_predicates(methods, by), *read.predicates)))
        readings.append((key, by, read, predicates))
    headings: dict[str, tuple[float, ...]] = {}
    for _key, _by, read, _predicates in readings:
        for path, scales in _trig_headings(read).items():
            headings[path] = tuple(set((*headings.get(path, ()), *scales)))
    effects = []
    for key, _by, read, predicates in readings:
        for kind, findings in (('vector_increment', _drive_effects(read)), ('turn', _turn_effects(read, headings))):
            for body, heading, changed, steps, source_lines in findings:
                unknown = tuple(dict.fromkeys((*read.unknown,
                                                *(('incremented fields have not been linked to rendered position',)
                                                  if kind == 'vector_increment' else ()),
                                                *(('source constant initialization is unproved',) if read.constant_reads else ()),
                                                *(('unresolved source control flow',) if predicates else ()))))
                power = frames_a_second
                effects.append(MethodMotion(key, read.method.target, kind, body, heading, changed, steps,
                                            tuple(value * power for value in steps), predicates, unknown, source_lines))
    return SourceMethodMotion(tuple(effects), _recoveries(methods, constants, float(frames_a_second)),
                              tuple(sorted(constants.items())), source=source)


def kept(reading: SourceMethodMotion | None) -> dict[str, Any]:
    """An explicit bounded JSON projection; unknown branch evidence is retained."""
    if reading is None:
        return {}
    from dataclasses import asdict

    projected = {'version': 1, **asdict(reading)}
    if unkept(projected) is None:
        raise ValueError('source motion reading exceeds its serializable contract')
    return projected


def unkept(value: Any) -> SourceMethodMotion | None:
    """Refuse invalid saved arithmetic rather than manufacturing default numbers."""
    if not isinstance(value, dict) or value.get('version') != 1:
        return None
    def text(item: Any, longest: int = 512) -> str:
        if not isinstance(item, str) or len(item) > longest:
            raise ValueError
        return item
    def items(item: Any, longest: int = 256) -> tuple:
        if not isinstance(item, (list, tuple)) or len(item) > longest:
            raise ValueError
        return tuple(item)
    def numbers(item: Any) -> tuple[float, ...]:
        values = items(item, 8)
        if any(type(number) not in (int, float) or not math.isfinite(number) for number in values):
            raise ValueError
        return tuple(float(number) for number in values)
    def addresses(item: Any, *, offset: bool = False) -> tuple[int, ...]:
        values = items(item, 256)
        if any(type(number) is not int or number < (0 if offset else 1) or number > 100_000_000 for number in values):
            raise ValueError
        return values
    def predicates(item: Any) -> tuple[SourcePredicate, ...]:
        found = []
        for part in items(item, 128):
            if not isinstance(part, dict):
                raise ValueError
            line = part['line']
            offset = part.get('offset')
            if type(line) is not int or line < 0 or line > _MAX_LINES:
                raise ValueError
            if offset is not None and (type(offset) is not int or not 0 <= offset <= 100_000_000):
                raise ValueError
            if line == 0 and offset is None:
                raise ValueError
            found.append(SourcePredicate(text(part['expression']), line, text(part['reason']), offset,
                                         text(part.get('source', ''), 1024)))
        return tuple(found)
    try:
        source = text(value.get('source', ''), 1024)
        if source and re.fullmatch(r'swf:[0-9a-f]{64}', source) is None:
            return None
        effects = []
        for part in items(value['effects']):
            if not isinstance(part, dict) or part['kind'] not in ('drive', 'turn', 'vector_increment'):
                return None
            steps, rates = numbers(part['steps']), numbers(part['rates'])
            if not steps or len(steps) != len(rates):
                return None
            effects.append(MethodMotion(text(part['key'], 64), text(part['method']), part['kind'],
                                        text(part['body']), text(part['heading']),
                                        tuple(text(path) for path in items(part['changed'], 16)), steps, rates,
                                        predicates(part['predicates']), tuple(text(reason) for reason in items(part['unknown'])),
                                        addresses(part['lines']), addresses(part.get('offsets', ()), offset=True)))
        recoveries = []
        for part in items(value['recoveries'], 64):
            if not isinstance(part, dict):
                return None
            recoveries.append(DampedRecovery(part['retained_per_frame'], part['stopping_speed'], part['frames_a_second'],
                                             addresses(part['lines']), predicates(part['predicates'])))
        constants = []
        for pair in items(value['constants'], 1024):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                return None
            constants.append((text(pair[0]), numbers([pair[1]])[0]))
        return SourceMethodMotion(tuple(effects), tuple(recoveries), tuple(constants),
                                  tuple(text(reason) for reason in items(value['unknown'])), source)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


@invariant('perception.source_motion_keeps_unresolved_branch_conditions', scope='perception',
           owner='core/perception/motion_inside_source_methods.py', observational=False)
def _source_motion_conditions_invariant() -> tuple:
    effect = MethodMotion('z', 'Opaque.method', 'turn', 'Opaque.body', 'Opaque.body.q', ('Opaque.body.r',),
                          (.1,), (1.,), (SourcePredicate('!Opaque.body.available', 4),), (), (5,))
    assert not effect.unconditional
    return ()
