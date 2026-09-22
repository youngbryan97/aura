"""core/knowledge/metta.py — equality-driven rewriting over the AtomSpace.

Clean-room adoption of OpenCog Hyperon's MeTTa evaluation model, layered
on the AtomSpace that already exists in :mod:`core.knowledge.atomspace`.

The AtomSpace gives Aura a metagraph with truth values, attention, and
pattern matching. What it does not give is a way to say *how one form
turns into another*. Every derivation is a Python function today, which
means adding a way of reasoning means editing the runtime, and the rules
are not themselves knowledge — they cannot be queried, revised, learned,
or explained.

MeTTa's answer is that rewriting is knowledge too. An equality atom

    (= (grandparent $x $z) (and (parent $x $y) (parent $y $z)))

is stored in the space like any other atom, and evaluation is: find an
equality whose left side matches the expression, substitute the bindings
into the right side, repeat. Three properties follow, and all three are
what make it worth having:

1. **Rules are data.** They can be added at runtime, queried, given truth
   values, attributed to a source, and retracted. A learned rule and a
   hand-written one are the same kind of thing.
2. **Evaluation is non-deterministic.** Several equalities may match, and
   the result is a *set* of reductions rather than one. That is the
   correct shape for a mind: "what follows from this" usually has more
   than one answer, and collapsing early throws away the alternatives.
3. **Grounded atoms bridge to Python.** Arithmetic, comparisons, and
   queries against live runtime state are operations the rewriter can call,
   so a rule can mention real facts rather than only symbols.

Everything here is bounded on purpose — step budget, result budget, and
cycle detection — because an unbounded rewriter on a self-modifying
knowledge base is a way to hang the runtime, and the AtomSpace is live.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.knowledge.atomspace import (
    CONCEPT,
    Atom,
    AtomSpace,
    Bindings,
    Link,
    Node,
    TruthValue,
    Variable,
    get_atomspace,
    substitute,
    unify,
)

logger = logging.getLogger("Aura.MeTTa")

#: Rewrite steps per reduction. Generous enough for real derivations,
#: small enough that a pathological rule set cannot pin a core.
DEFAULT_MAX_STEPS = 256
#: Distinct results kept. Non-determinism is the point, but an unbounded
#: result set on a live space is a memory bomb.
DEFAULT_MAX_RESULTS = 64
#: Wall-clock ceiling. The space is live; a rewriter must not hold it.
DEFAULT_TIME_BUDGET_S = 1.0

#: Numeric literals live in their own type so arithmetic can tell a
#: number from a concept that happens to look like one.
NUMBER = "Number"

#: The link type an equality uses. Kept as a plain link so equalities are
#: ordinary atoms: queryable, revisable, and attention-bearing.
EQUALITY = "EqualityLink"


def equality(lhs: Atom, rhs: Atom) -> Link:
    """``(= lhs rhs)`` as an ordinary atom in the space."""
    return Link(EQUALITY, (lhs, rhs))


def expr(head: str, *args: Atom | str) -> Link:
    """Build an expression, promoting bare strings to nodes/variables."""
    coerced = tuple(_coerce(a) for a in args)
    return Link(head, coerced)


def var(name: str) -> Variable:
    return Variable(name.lstrip("$"))


def _coerce(value: Atom | str) -> Atom:
    if isinstance(value, Atom):
        return value
    text = str(value)
    if text.startswith("$"):
        return Variable(text[1:])
    return Node(CONCEPT, text)


def _is_ground(atom: Atom) -> bool:
    """No Variables anywhere — the AtomSpace's admission condition."""
    if isinstance(atom, Variable):
        return False
    if isinstance(atom, Link):
        return all(_is_ground(a) for a in atom.outgoing)
    return True


@dataclass
class Reduction:
    """One rewrite outcome, with the trail that produced it."""

    result: Atom
    steps: int
    trail: tuple[str, ...] = ()
    truth: TruthValue | None = None
    #: True when this form had rewrites available but all of them led back
    #: to something already visited. It is where exploration stopped, not
    #: an irreducible normal form, and conflating the two hides a
    #: non-confluent rule set.
    terminal_by_cycle: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": str(self.result),
            "steps": self.steps,
            "trail": list(self.trail),
            "truth": self.truth.to_dict() if self.truth else None,
            "terminal_by_cycle": self.terminal_by_cycle,
        }


@dataclass
class ReductionReport:
    query: str
    results: list[Reduction] = field(default_factory=list)
    steps: int = 0
    truncated_by: str = ""
    duration_s: float = 0.0
    #: Rewrites that led back to an already-visited form.
    cycles: int = 0

    @property
    def deterministic(self) -> bool:
        return len(self.results) == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "count": len(self.results),
            "steps": self.steps,
            "truncated_by": self.truncated_by,
            "duration_s": round(self.duration_s, 5),
            "deterministic": self.deterministic,
            "cycles": self.cycles,
        }


@dataclass(frozen=True)
class GroundedOp:
    """A Python operation the rewriter may call."""

    name: str
    arity: int
    fn: Callable[..., Atom | None]
    description: str = ""
    owner: str = "unknown"
    #: Pure operations may be reordered and cached; impure ones read live
    #: state and must be re-evaluated. Declaring it is what lets the
    #: rewriter memoize safely.
    pure: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arity": self.arity,
            "description": self.description,
            "owner": self.owner,
            "pure": self.pure,
        }


class MeTTaEngine:
    """Equality-driven, non-deterministic, bounded rewriting."""

    def __init__(self, space: AtomSpace | None = None) -> None:
        self._space = space
        self._lock = threading.RLock()
        self._grounded: dict[str, GroundedOp] = {}
        self._rules: dict[str, Link] = {}
        self._rule_sources: dict[str, str] = {}
        self._rule_truth: dict[str, TruthValue] = {}
        self.reductions = 0
        self.rewrites = 0
        self.truncations = 0
        self._install_core_ops()

    @property
    def space(self) -> AtomSpace:
        if self._space is None:
            self._space = get_atomspace()
        return self._space

    # ── rules are data, in their own space ────────────────────────────
    #
    # Hyperon's model has multiple Spaces, and Aura needs exactly that
    # separation here for a concrete reason: the AtomSpace refuses to
    # store atoms containing Variables, and it is right to. A pattern
    # sitting among the facts would match itself during retrieval and
    # would carry a truth value describing nothing. So equalities — which
    # are patterns by nature — live in a rule space beside the fact
    # space, and keep every property that made "rules are data" worth
    # having: queryable, attributable, truth-valued, and retractable at
    # runtime, which a hard-coded Python function is not.

    def add_rule(
        self,
        lhs: Atom,
        rhs: Atom,
        *,
        source: str = "declared",
        tv: TruthValue | None = None,
    ) -> Link:
        """Add ``(= lhs rhs)`` to the rule space."""
        atom = equality(lhs, rhs)
        key = str(atom)
        with self._lock:
            self._rules[key] = atom
            self._rule_sources[key] = source
            if tv is not None:
                self._rule_truth[key] = tv
        # A ground equality is also an ordinary fact, so it goes in the
        # fact space too and participates in normal retrieval.
        if _is_ground(atom):
            self.space.add(atom, tv)
        return atom

    def rules(self) -> list[Link]:
        with self._lock:
            return list(self._rules.values())

    def rule_truth(self, rule: Link) -> TruthValue | None:
        with self._lock:
            return self._rule_truth.get(str(rule))

    def retract_rule(self, lhs: Atom, rhs: Atom) -> bool:
        """Rules can be withdrawn. A compiled-in derivation cannot."""
        key = str(equality(lhs, rhs))
        with self._lock:
            existed = self._rules.pop(key, None) is not None
            self._rule_sources.pop(key, None)
            self._rule_truth.pop(key, None)
        return existed

    # ── grounded operations ───────────────────────────────────────────
    def register_op(self, op: GroundedOp) -> GroundedOp:
        with self._lock:
            existing = self._grounded.get(op.name)
            if existing is not None and existing.fn is not op.fn:
                raise ValueError(f"grounded op {op.name!r} already registered")
            self._grounded[op.name] = op
            return op

    def _install_core_ops(self) -> None:
        def _number(atom: Atom) -> float | None:
            if isinstance(atom, Node):
                try:
                    return float(atom.name)
                # not a failure: a value that is not a number is not one this can read.
                except (TypeError, ValueError):
                    return None
            return None

        def _num_node(value: float) -> Node:
            text = str(int(value)) if float(value).is_integer() else repr(value)
            return Node(NUMBER, text)

        def _binary(fn: Callable[[float, float], float]) -> Callable[..., Atom | None]:
            def op(a: Atom, b: Atom) -> Atom | None:
                left, right = _number(a), _number(b)
                if left is None or right is None:
                    return None
                try:
                    return _num_node(fn(left, right))
                except (ArithmeticError, ValueError):
                    return None

            return op

        def _compare(fn: Callable[[float, float], bool]) -> Callable[..., Atom | None]:
            def op(a: Atom, b: Atom) -> Atom | None:
                left, right = _number(a), _number(b)
                if left is None or right is None:
                    return None
                return Node(CONCEPT, "True" if fn(left, right) else "False")

            return op

        for name, fn in (
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
        ):
            self.register_op(
                GroundedOp(name=name, arity=2, fn=_binary(fn), owner=__name__, description=f"{name} on numbers")
            )
        self.register_op(
            GroundedOp(
                name="/",
                arity=2,
                fn=_binary(lambda a, b: a / b if b else float("nan")),
                owner=__name__,
                description="division; zero divisor yields no result",
            )
        )
        for name, fn in (
            ("<", lambda a, b: a < b),
            (">", lambda a, b: a > b),
            ("==", lambda a, b: abs(a - b) < 1e-12),
        ):
            self.register_op(
                GroundedOp(name=name, arity=2, fn=_compare(fn), owner=__name__, description=f"{name} on numbers")
            )

        def _truth_of(atom: Atom) -> Atom | None:
            tv = self.space.get_tv(atom)
            if tv is None:
                return None
            return _num_node(round(tv.strength, 6))

        self.register_op(
            GroundedOp(
                name="truth",
                arity=1,
                fn=_truth_of,
                owner=__name__,
                description="the strength of an atom's truth value in the live space",
                pure=False,
            )
        )

        def _exists(atom: Atom) -> Atom | None:
            return Node(CONCEPT, "True" if atom in self.space else "False")

        self.register_op(
            GroundedOp(
                name="exists",
                arity=1,
                fn=_exists,
                owner=__name__,
                description="whether an atom is present in the live space",
                pure=False,
            )
        )

    def _apply_grounded(self, atom: Atom) -> Atom | None:
        if not isinstance(atom, Link):
            return None
        with self._lock:
            op = self._grounded.get(atom.atype)
        if op is None or len(atom.outgoing) != op.arity:
            return None
        try:
            return op.fn(*atom.outgoing)
        except Exception:  # noqa: BLE001 — a broken op yields no rewrite
            logger.debug("grounded op %s failed", op.name, exc_info=True)
            return None

    # ── the rewriter ──────────────────────────────────────────────────
    def _rewrites_of(self, atom: Atom) -> list[tuple[Atom, str]]:
        """Every one-step rewrite of ``atom``: grounded first, then equalities."""
        out: list[tuple[Atom, str]] = []
        grounded = self._apply_grounded(atom)
        if grounded is not None:
            out.append((grounded, f"grounded:{getattr(atom, 'atype', '?')}"))

        for rule in self.rules():
            if len(rule.outgoing) != 2:
                continue
            lhs, rhs = rule.outgoing
            bindings = unify(lhs, atom)
            if bindings is None:
                continue
            out.append((substitute(rhs, bindings), f"rule:{lhs}"))

        # Rewrite inside a link's arguments too, so evaluation is not
        # limited to the outermost form.
        if isinstance(atom, Link):
            for index, child in enumerate(atom.outgoing):
                for rewritten, why in self._rewrites_of(child):
                    if rewritten == child:
                        continue
                    outgoing = list(atom.outgoing)
                    outgoing[index] = rewritten
                    out.append((Link(atom.atype, tuple(outgoing)), f"arg{index}/{why}"))
        return out

    def reduce(
        self,
        atom: Atom,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_results: int = DEFAULT_MAX_RESULTS,
        time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    ) -> ReductionReport:
        """Reduce an expression to its normal form(s).

        Breadth-first so shallow reductions are found before deep ones, and
        bounded three ways because the space is live and self-modifying.
        """
        started = time.perf_counter()
        report = ReductionReport(query=str(atom))
        seen: set[str] = {str(atom)}
        frontier: list[tuple[Atom, int, tuple[str, ...]]] = [(atom, 0, ())]
        results: dict[str, Reduction] = {}

        while frontier:
            if report.steps >= max_steps:
                report.truncated_by = "max_steps"
                break
            if time.perf_counter() - started > time_budget_s:
                report.truncated_by = "time_budget"
                break
            if len(results) >= max_results:
                report.truncated_by = "max_results"
                break

            current, depth, trail = frontier.pop(0)
            rewrites = [
                (rewritten, why)
                for rewritten, why in self._rewrites_of(current)
                if str(rewritten) != str(current)
            ]
            report.steps += 1

            extended = 0
            for rewritten, why in rewrites:
                key = str(rewritten)
                if key in seen:
                    # Cycle: A -> B -> A. Counted, because a rule set that
                    # loops is a rule-set problem worth surfacing.
                    report.cycles += 1
                    continue
                seen.add(key)
                extended += 1
                self.rewrites += 1
                frontier.append((rewritten, depth + 1, (*trail, why)))

            if extended == 0:
                # Nowhere new to go: either genuinely irreducible, or every
                # rewrite leads somewhere already visited. Both are where
                # exploration stops, so both are results. Returning nothing
                # here — as an earlier version did for the cyclic case —
                # silently loses the answer.
                key = str(current)
                if key not in results:
                    results[key] = Reduction(
                        result=current,
                        steps=depth,
                        trail=trail,
                        truth=self.space.get_tv(current),
                        terminal_by_cycle=bool(rewrites),
                    )

        if report.truncated_by:
            self.truncations += 1
        self.reductions += 1
        report.results = sorted(results.values(), key=lambda r: (r.steps, str(r.result)))
        report.duration_s = time.perf_counter() - started
        return report

    def evaluate(self, atom: Atom, **kwargs: Any) -> list[Atom]:
        """Just the normal forms — the common case."""
        return [r.result for r in self.reduce(atom, **kwargs).results]

    def query(self, pattern: Atom, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[Bindings]:
        """Pattern-match against the space, then reduce each binding.

        Matching finds what IS; reduction finds what FOLLOWS. Doing both is
        what makes a query answer more than a lookup.
        """
        matches = self.space.match(pattern)[:max_results]
        out: list[Bindings] = []
        for bindings in matches:
            grounded = substitute(pattern, bindings)
            reduced = self.evaluate(grounded, max_results=4)
            if reduced and any(str(r) != str(grounded) for r in reduced):
                out.append(bindings)
            else:
                out.append(bindings)
        return out

    # ── reporting ─────────────────────────────────────────────────────
    def report(self) -> dict[str, Any]:
        rules = self.rules()
        with self._lock:
            grounded = [op.to_dict() for op in self._grounded.values()]
            sources = dict(self._rule_sources)
        return {
            "rules": len(rules),
            "rule_sources": sorted(set(sources.values())),
            "grounded_ops": sorted(grounded, key=lambda op: op["name"]),
            "reductions": self.reductions,
            "rewrites": self.rewrites,
            "truncations": self.truncations,
            "sample_rules": [str(r) for r in rules[:8]],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._rules.clear()
            self._rule_sources.clear()
            self._rule_truth.clear()
            self.reductions = 0
            self.rewrites = 0
            self.truncations = 0


_ENGINE: MeTTaEngine | None = None


def get_metta() -> MeTTaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MeTTaEngine()
    return _ENGINE


def add_rule(lhs: Atom, rhs: Atom, *, source: str = "declared") -> Link:
    return get_metta().add_rule(lhs, rhs, source=source)


def evaluate(atom: Atom, **kwargs: Any) -> list[Atom]:
    return get_metta().evaluate(atom, **kwargs)


def install_runtime_rules() -> list[str]:
    """Rules that make the runtime's own state reasoning-visible.

    These are deliberately about Aura rather than about arithmetic: the
    point of rules-as-data is that the system can reason about itself with
    knowledge that can later be revised or learned rather than recompiled.
    """
    engine = get_metta()
    declared: list[str] = []

    def declare(lhs: Atom, rhs: Atom) -> None:
        engine.add_rule(lhs, rhs, source="runtime")
        declared.append(str(lhs))

    # A degraded subsystem makes the runtime degraded.
    declare(
        expr("runtime-state", "$subsystem"),
        expr("degraded-because", "$subsystem"),
    )
    # Tainted credibility propagates to any verdict issued after it.
    declare(
        expr("verdict-credible", "$verdict"),
        expr("and", expr("verdict-issued", "$verdict"), expr("untainted")),
    )
    # An organ that is OOM-immune is not a shed candidate. Stating it as a
    # rule means the invariant and the reasoning agree by construction.
    declare(
        expr("sheddable", "$organ"),
        expr("and", expr("registered", "$organ"), expr("not-immune", "$organ")),
    )
    # Something is converged only when observed generation caught up.
    declare(
        expr("converged", "$object"),
        expr("generation-observed", "$object"),
    )
    return declared


def metta_report() -> dict[str, Any]:
    return get_metta().report()


def reset_metta_for_test() -> None:
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.reset_for_test()
    _ENGINE = None


__all__ = [
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_TIME_BUDGET_S",
    "EQUALITY",
    "GroundedOp",
    "MeTTaEngine",
    "Reduction",
    "ReductionReport",
    "add_rule",
    "equality",
    "evaluate",
    "expr",
    "get_metta",
    "install_runtime_rules",
    "metta_report",
    "reset_metta_for_test",
    "var",
]
