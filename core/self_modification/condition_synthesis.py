"""The smallest condition that agrees with every run: the synthesis half of semantic program repair (Master Todo F16).

Semantic repair (semantic_repair.py) records, at a branch, the values in scope each time it was reached and what the
branch had to decide there: what it did decide in every test that passes, and what a failing test needed once it was
forced (angelic fix localisation, Nopol, Xuan et al., TSE 2017). This finds conditions over those values that decide
the same as every one of those runs, the way SemFix (Nguyen et al., ICSE 2013) and Angelix (Mechtaev et al., ICSE
2016) synthesise a repair from the components in scope, and orders them by how little they change the condition that
was there:

1. the condition itself with one part changed: a comparison turned (``<`` to ``<=``), a bound moved by one or to a
   value the runs show is the boundary, ``and`` and ``or`` swapped, one clause of a conjunction or disjunction dropped,
   or the whole negated;
2. the condition joined with one more test by ``and`` or ``or``: a guard it was missing;
3. a test on its own;
4. two tests joined.

Nothing here is a guess: every candidate decides as every recorded run needed, and the tests then judge it on the
code itself (repair_by_selection.py). The terms come from what the probe recorded (semantic_probe.summarise): a
name's value, ``len(x)``, ``x is None``, ``obj.attr``, and the parts of the original condition read on their own.
"""
from __future__ import annotations

import ast
import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Example", "conditions", "ranked_conditions"]

#: How many distinct values of one term are offered as bounds.
BOUNDS_PER_TERM = 10
#: How many single tests are kept for joining two at a time, after those that cannot help are left out.
JOINED_AT_MOST = 120
#: Bounds that name no particular run: comparing with one of these does not pin the condition to an example.
NEUTRAL = (0, 1, -1, "", None)

_MISSING = object()
_COMPARISONS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
_TURNED = {ast.Lt: (ast.LtE, ast.Gt, ast.GtE), ast.LtE: (ast.Lt, ast.GtE, ast.Gt), ast.Gt: (ast.GtE, ast.Lt, ast.LtE),
           ast.GtE: (ast.Gt, ast.LtE, ast.Lt), ast.Eq: (ast.NotEq,), ast.NotEq: (ast.Eq,), ast.Is: (ast.IsNot,),
           ast.IsNot: (ast.Is,), ast.In: (ast.NotIn,), ast.NotIn: (ast.In,)}


@dataclass(frozen=True)
class Example:
    """One time the branch was reached: the terms in scope, the condition's own parts, what it decided, and what it
    had to decide."""

    terms: Mapping[str, Any]
    want: bool
    decided: Any = None
    parts: Mapping[str, Any] | None = None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _evaluate(node: ast.AST, env: Mapping[str, Any]) -> Any:
    """``node`` read over recorded values; _MISSING when a value it needs was not recorded."""
    text = ast.unparse(node)
    if text in env:
        return env[text]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        inner = _evaluate(node.operand, env)
        if inner is _MISSING:
            return _MISSING
        if isinstance(node.op, ast.Not):
            return not inner
        if isinstance(node.op, ast.USub) and _number(inner):
            return -inner
        return _MISSING
    if isinstance(node, ast.BoolOp):
        result: Any = None
        for value in node.values:
            result = _evaluate(value, env)
            if result is _MISSING:
                return _MISSING
            if isinstance(node.op, ast.And) and not result:
                return result
            if isinstance(node.op, ast.Or) and result:
                return result
        return result
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left, env), _evaluate(node.right, env)
        if not (_number(left) and _number(right)):
            return _MISSING
        try:
            return {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right,
                    ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right}[type(node.op)]()
        except (KeyError, ZeroDivisionError):
            return _MISSING
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, env)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(comparator, ast.Constant) \
                    and comparator.value is None:
                said = env.get(f"{ast.unparse(node.left)} is None", _MISSING)
                outcome = (left is None) if left is not _MISSING else said
                if outcome is _MISSING:
                    return _MISSING
                if isinstance(op, ast.IsNot):
                    outcome = not outcome
                if not outcome:
                    return False
                left = None
                continue
            right = _evaluate(comparator, env)
            if left is _MISSING or right is _MISSING:
                return _MISSING
            name = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}.get(type(op))
            if name is None:
                return _MISSING
            try:
                if not _COMPARISONS[name](left, right):
                    return False
            except TypeError:
                return _MISSING
            left = right
        return True
    if isinstance(node, ast.IfExp):
        test = _evaluate(node.test, env)
        if test is _MISSING:
            return _MISSING
        return _evaluate(node.body if test else node.orelse, env)
    return _MISSING


@dataclass(frozen=True)
class _Atom:
    text: str
    values: tuple[bool, ...]
    #: Whether it is a part of the original condition (a smaller change when it is).
    original: bool = False


def _value_of(term: str, example: Example) -> Any:
    if term in example.terms:
        return example.terms[term]
    if example.parts and term in example.parts:
        return example.parts[term]
    return _MISSING


def _is_none(base: str, example: Example) -> Any:
    if base in example.terms:
        return example.terms[base] is None
    said = example.terms.get(f"{base} is None", _MISSING)
    return said if said is _MISSING else bool(said)


def _negated(term: str) -> str:
    """``term`` negated as Python writes it: ``k not in d`` for a membership, ``not t`` otherwise."""
    try:
        node = ast.parse(term, mode="eval").body
    except SyntaxError:
        return f"not {term}"
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
        return ast.unparse(ast.Compare(left=node.left, ops=[ast.NotIn()], comparators=node.comparators))
    return f"not {term}"


def _atoms(examples: Sequence[Example], constants: Sequence[Any], original: str,
           reads: Sequence[str] = ()) -> list[_Atom]:
    """Every single test over the recorded terms that is defined in every run, one per distinct outcome: of tests that
    decide every run alike, the one that reads more of ``reads`` (what the code at the place reads) is kept."""
    names = set.intersection(*({*e.terms, *(e.parts or {})} for e in examples))
    bases = {n[: -len(" is None")] for n in names if n.endswith(" is None")} | {
        n for n in names if any(_value_of(n, e) is None for e in examples)}
    found: list[_Atom] = []

    def add(text: str, values: Sequence[Any], from_original: bool = False) -> None:
        if any(v is _MISSING for v in values):
            return
        found.append(_Atom(text, tuple(bool(v) for v in values), from_original))

    for base in sorted(bases):
        if all(_is_none(base, e) is not _MISSING for e in examples):
            outcome = [_is_none(base, e) for e in examples]
            add(f"{base} is None", outcome)
            add(f"{base} is not None", [not v for v in outcome])
    numeric: list[str] = []
    for term in sorted(n for n in names if not n.endswith(" is None")):
        values = [_value_of(term, e) for e in examples]
        from_original = bool(examples[0].parts) and term in (examples[0].parts or {})
        if all(isinstance(v, bool) for v in values):
            add(term, values, from_original)
            add(_negated(term), [not v for v in values], from_original)
        elif all(_number(v) for v in values):
            numeric.append(term)
        elif all(isinstance(v, str) for v in values):
            add(term, values, from_original)
            add(f"not {term}", [not v for v in values], from_original)
            for literal in sorted({v for v in values})[:BOUNDS_PER_TERM]:
                add(f"{term} == {literal!r}", [v == literal for v in values])
                add(f"{term} != {literal!r}", [v != literal for v in values])
    for term in numeric:
        values = [_value_of(term, e) for e in examples]
        if term.startswith("len(") and term.endswith(")"):
            # A size is read as the thing being empty or not, the way Python code says it.
            inner = term[4:-1]
            add(inner, [v > 0 for v in values])
            add(f"not {inner}", [v == 0 for v in values])
        seen = sorted({v for v in values})
        bounds = list(dict.fromkeys([0, 1, *[c for c in constants if _number(c)], *seen[:BOUNDS_PER_TERM]]))
        for bound in bounds:
            for op, compare in _COMPARISONS.items():
                add(f"{term} {op} {bound!r}", [compare(v, bound) for v in values])
    for left, right in itertools.permutations(numeric, 2):
        lefts, rights = [_value_of(left, e) for e in examples], [_value_of(right, e) for e in examples]
        for shift, written in ((0, right), (1, f"{right} + 1"), (-1, f"{right} - 1")):
            for op, compare in _COMPARISONS.items():
                if op in {"<", "<="} or shift == 0 or op in {"==", "!="}:
                    add(f"{left} {op} {written}", [compare(a, b + shift) for a, b in zip(lefts, rights, strict=True)])
    if original:
        add(original, [e.decided for e in examples], True)
    # One test for each outcome: from the original condition if one is, then one that pins no run, then the shortest.
    distinct: dict[tuple[bool, ...], _Atom] = {}
    for atom in found:
        held = distinct.get(atom.values)
        if held is None or _preferred(atom, reads) > _preferred(held, reads):
            distinct[atom.values] = atom
    return list(distinct.values())


def _preferred(atom: _Atom, reads: Sequence[str] = ()) -> tuple[bool, int, int, int]:
    # Reading more of what the faulting code reads before the shorter text: LIVE 17 November 2026 ``len(prices) <= 1``
    # was kept over ``item not in prices`` for ``prices[item]``, being two characters shorter, and was wrong.
    return atom.original, len(_names(atom.text) & set(reads)), -_pins(atom.text), -len(atom.text)


def _variants(original: str, constants: Sequence[Any], examples: Sequence[Example]) -> Iterator[tuple[str, int]]:
    """The original condition with one part changed, each once, with what the change costs.

    A comparison turned, a bound moved by one, ``and`` and ``or`` swapped, a clause dropped or the whole negated cost
    one: the shapes one-line fixes most often take (PAR, Kim et al., ICSE 2013; TBar, Liu et al., ISSTA 2019). A bound
    moved to a value the runs happened to show costs three: it fits the runs seen and may fit no others.
    """
    try:
        tree = ast.parse(original, mode="eval")
    # not a failure: a condition that does not parse has no parts to change, so none is offered.
    except SyntaxError:
        return
    seen: set[str] = {original}
    nodes = list(ast.walk(tree.body))
    bounds = sorted({v for e in examples for v in list(e.terms.values()) + list((e.parts or {}).values())
                     if _number(v)} | {c for c in constants if _number(c)})[: BOUNDS_PER_TERM * 3]

    def changed(edit, cost: int = 1) -> Iterator[tuple[str, int]]:
        copy = ast.parse(original, mode="eval")
        twins = list(ast.walk(copy.body))
        if edit(twins):
            text = ast.unparse(_Simplified().visit(copy).body)
            if text not in seen:
                seen.add(text)
                yield text, cost

    for index, node in enumerate(nodes):
        if isinstance(node, ast.Compare):
            for at, op in enumerate(node.ops):
                for other in _TURNED.get(type(op), ()):
                    def turn(twins, index=index, at=at, other=other):
                        twins[index].ops[at] = other()
                        return True
                    yield from changed(turn)
        elif isinstance(node, ast.BoolOp):
            def swap(twins, index=index):
                twins[index].op = ast.Or() if isinstance(twins[index].op, ast.And) else ast.And()
                return True
            yield from changed(swap)
            for drop in range(len(node.values)):
                def without(twins, index=index, drop=drop):
                    kept = [v for i, v in enumerate(twins[index].values) if i != drop]
                    if len(kept) == 1:
                        return _replace_node(twins, index, kept[0])
                    twins[index].values = kept
                    return True
                yield from changed(without)
        elif isinstance(node, ast.Constant) and _number(node.value):
            for other in [node.value + 1, node.value - 1, *bounds]:
                if other != node.value:
                    def move(twins, index=index, other=other):
                        twins[index].value = other
                        return True
                    yield from changed(move, 1 if abs(other - node.value) == 1 else 3)
    negated = f"not ({original})" if " " in original else f"not {original}"
    if negated not in seen:
        yield negated, 1


class _Simplified(ast.NodeTransformer):
    """``x + 0`` and ``x - 0`` written as ``x``, and ``x * 1`` as ``x``: what a bound moved to zero leaves behind."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        right = node.right
        if isinstance(right, ast.Constant) and _number(right.value):
            if isinstance(node.op, (ast.Add, ast.Sub)) and right.value == 0:
                return node.left
            if isinstance(node.op, ast.Mult) and right.value == 1:
                return node.left
        return node


def _replace_node(twins: list[ast.AST], index: int, replacement: ast.AST) -> bool:
    """Put ``replacement`` where ``twins[index]`` sits in its parent."""
    target = twins[index]
    for parent in twins:
        for field, value in ast.iter_fields(parent):
            if value is target:
                setattr(parent, field, replacement)
                return True
            if isinstance(value, list) and target in value:
                value[value.index(target)] = replacement
                return True
    return False


def _join(word: str, left: str, right: str) -> str:
    """``left word right``, with a disjunction inside a conjunction held in parentheses so it still reads as one."""
    def wrapped(text: str) -> str:
        return f"({text})" if word == "and" and " or " in text else text
    return f"{wrapped(left)} {word} {wrapped(right)}"


def _pins(text: str, written: Sequence[Any] = ()) -> int:
    """How tightly a test fits the runs seen and maybe no others: three for an equality with a value that is neither
    neutral nor written in the code, two for an ordering against such a value (a bound the runs supplied), else 0."""
    try:
        tree = ast.parse(text, mode="eval")
    # not a failure: a test that does not parse pins no value, which 0 says.
    except SyntaxError:
        return 0
    worst = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators, strict=True):
                if not isinstance(right, ast.Constant) or right.value in NEUTRAL or right.value in written:
                    continue
                worst = max(worst, 3 if isinstance(op, (ast.Eq, ast.NotEq)) or isinstance(right.value, str) else 2)
    return worst


#: Values a size can take when a condition is tried for whether it can ever decide both ways.
_SIZES = (0, 1, 2, 3, 100)
#: Combinations of term values tried for that, at most.
_TRIED_AT_MOST = 4096


def _domain(term: str, examples: Sequence[Example]) -> list[Any]:
    """Values ``term`` can take: a size from empty up, a number around what was seen, both truth values, None or not."""
    seen = [_value_of(term, e) for e in examples]
    seen = [v for v in seen if v is not _MISSING]
    if term.startswith("len(") or term.endswith(" is None"):
        return list(_SIZES) if term.startswith("len(") else [True, False]
    if seen and all(isinstance(v, bool) for v in seen):
        return [True, False]
    if seen and all(_number(v) for v in seen):
        near = {w for v in seen for w in (v - 1, v, v + 1)}
        return sorted(near | {-1, 0, 1, 10**6, -(10**6)})
    if seen and all(isinstance(v, str) for v in seen):
        return list(dict.fromkeys([*seen, "", "x"]))
    return list(dict.fromkeys([*seen, None]))


def _decides_both_ways(text: str, examples: Sequence[Example]) -> bool:
    """Whether the condition can come out true and false over the values its terms can take.

    One that cannot deletes a branch while seeming to mend it: ``len(rows) < 0`` before a ``raise`` passes every
    test that only asks the raise to stop, and the refusal is dead code. LIVE 16 November 2026 it was the first
    semantic candidate for her drill's planted fault, whose guarding test reached nothing of the refusal.
    """
    try:
        body = ast.parse(text, mode="eval").body
    # not a failure: a condition that does not parse cannot come out either way, which False says.
    except SyntaxError:
        return False
    terms = sorted({t for e in examples for t in [*e.terms, *(e.parts or {})] if t in text}, key=len, reverse=True)
    if not terms:
        return True
    domains = [_domain(t, examples) for t in terms]
    outcomes: set[bool] = set()
    for tried, values in enumerate(itertools.product(*domains)):
        if tried >= _TRIED_AT_MOST:
            break
        env = dict(zip(terms, values, strict=True))
        for term, value in list(env.items()):
            if term.startswith("len(") and term.endswith(")"):
                env.setdefault(term[4:-1], value > 0)
            if term.endswith(" is None"):
                env.setdefault(term[: -len(" is None")], None if value else 0)
        outcome = _evaluate(body, env)
        if outcome is _MISSING:
            continue
        outcomes.add(bool(outcome))
        if len(outcomes) == 2:
            return True
    return not outcomes


def _names(text: str) -> set[str]:
    try:
        return {n.id for n in ast.walk(ast.parse(text, mode="eval")) if isinstance(n, ast.Name)}
    except SyntaxError:
        return set()


def ranked_conditions(examples: Sequence[Example], *, original: str = "", constants: Sequence[Any] = (),
                      names: Sequence[str] = (), most: int = 12) -> list[tuple[int, str]]:
    """Conditions that decide as every example needed, each with its cost, the cheapest first.

    The cost is how much the condition changes what was there (see the module docstring), plus what _pins says
    when it compares with a value the runs supplied (it may hold for the runs seen and no others), less one when it
    reads a name
    the code at the place already reads: a repair over the names the fault's own statement uses is the likelier one
    (ACS, Xiong et al., ICSE 2017, rank predicates the same way).
    """
    if not examples:
        return []
    want = tuple(e.want for e in examples)
    found: dict[str, int] = {}
    used = set(names)

    def keep(cost: int, text: str) -> None:
        if text == original or not _decides_both_ways(text, examples):
            return
        cost += _pins(text, constants)
        cost -= 1 if used and _names(text) & used else 0
        if text not in found or cost < found[text]:
            found[text] = cost

    if original:
        for variant, cost in _variants(original, constants, examples):
            try:
                body = ast.parse(variant, mode="eval").body
            except SyntaxError:
                continue
            outcome = [_evaluate(body, {**e.terms, **(e.parts or {})}) for e in examples]
            if all(v is not _MISSING for v in outcome) and tuple(bool(v) for v in outcome) == want:
                keep(cost, variant)
    atoms = _atoms(examples, constants, original, names)
    decided = tuple(bool(e.decided) for e in examples) if original else None
    for atom in atoms:
        if atom.values == want:
            keep(2 if atom.original else 3, atom.text)
    if decided is not None:
        for atom in atoms:
            if atom.text == original:
                continue
            if tuple(d and a for d, a in zip(decided, atom.values, strict=True)) == want:
                keep(2, _join("and", original, atom.text))
            if tuple(d or a for d, a in zip(decided, atom.values, strict=True)) == want:
                keep(2, _join("or", original, atom.text))
    # Two tests joined. For "a and b" each must hold wherever the branch had to be taken; for "a or b" each must
    # fail wherever it had to be passed by. Only those are joined.
    covers = [a for a in atoms if all(v for v, w in zip(a.values, want, strict=True) if w)][:JOINED_AT_MOST]
    for left, right in itertools.combinations(covers, 2):
        if tuple(x and y for x, y in zip(left.values, right.values, strict=True)) == want:
            keep(4 - left.original - right.original, _join("and", left.text, right.text))
    misses = [a for a in atoms if not any(v for v, w in zip(a.values, want, strict=True) if not w)][:JOINED_AT_MOST]
    for left, right in itertools.combinations(misses, 2):
        if tuple(x or y for x, y in zip(left.values, right.values, strict=True)) == want:
            keep(4 - left.original - right.original, _join("or", left.text, right.text))
    return sorted(((cost, text) for text, cost in found.items()), key=lambda pair: (pair[0], len(pair[1])))[:most]


def conditions(examples: Sequence[Example], *, original: str = "", constants: Sequence[Any] = (),
               names: Sequence[str] = (), most: int = 12) -> list[str]:
    """The conditions of ranked_conditions, without their costs."""
    return [text for _, text in ranked_conditions(examples, original=original, constants=constants, names=names,
                                                   most=most)]
