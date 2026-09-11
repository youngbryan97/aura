"""Which process is the subject, when several overlapping ones could be.

Three things decide it, and two of them are gates rather than scores.

A candidate is admissible when nothing outside it secretly determines its
future once its own state and its interface are known, and when it is
recurrent as a whole. Closure and recurrence are not weighted ingredients
added to a total; a candidate that fails either is not a candidate, and its
rate is zero by definition rather than by arithmetic.

Among the admissible ones, overlapping candidates are ordered by dominance
over the whole intrinsic causal spectrum:

    A >= B  iff  Phi_A(T) >= Phi_B(T) for every horizon T

with strict inequality somewhere. That is deliberately a partial order. Two
candidates whose spectra cross are genuinely incomparable, and a rule that
picked one anyway would be smuggling in a preferred timescale — there is no
normalised scale-invariant measure over all positive times to average them
with, because the Haar measure of the multiplicative reals is dt/t and its
integral diverges. So the output is the Pareto frontier, which is a single
candidate when one dominates and a symmetry class when none does.

That is not an unfinished calculation. Two supports related by an exact
symmetry of the causal structure get equal values from every permutation-
invariant functional, so no intrinsic symmetry-respecting law can separate
them. Returning the class is the correct answer; returning one member would
require an exclusion postulate this module does not have and will not invent.

At the ten-domain grain the search collapses usefully. If the whole declared
core is one strongly connected component and it is closed against the measured
periphery, every proper subset has an incoming channel from its complement and
is therefore not closed — so the core is the unique domain-level carrier, and
that is a measured conclusion rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "Candidate",
    "ExclusionReport",
    "dominates",
    "pareto_frontier",
    "strongly_connected_components",
    "select_carriers",
]


def strongly_connected_components(
    nodes: Sequence[str], edges: Sequence[tuple[str, str]]
) -> list[tuple[str, ...]]:
    """Tarjan's components, largest first, each sorted.

    A recurrent subject support has to lie inside one of these: a node outside
    every cycle cannot be part of a process that returns to itself.
    """
    order = list(dict.fromkeys(nodes))
    outgoing: dict[str, list[str]] = {node: [] for node in order}
    for source, target in edges:
        if source in outgoing and target in outgoing and source != target:
            outgoing[source].append(target)

    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    found: list[tuple[str, ...]] = []

    for root in order:
        if root in index_of:
            continue
        # Explicit frames rather than recursion: ten domains is small, but a
        # finer grain is not, and a recursion limit is a silent wrong answer.
        work: list[tuple[str, int]] = [(root, 0)]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, position = work[-1]
            if position < len(outgoing[node]):
                work[-1] = (node, position + 1)
                nxt = outgoing[node][position]
                if nxt not in index_of:
                    index_of[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index_of[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                block: list[str] = []
                while True:
                    top = stack.pop()
                    on_stack.discard(top)
                    block.append(top)
                    if top == node:
                        break
                found.append(tuple(sorted(block)))
    found.sort(key=lambda block: (-len(block), block))
    return found


@dataclass
class Candidate:
    """One support, its admissibility, and its whole causal spectrum."""

    support: frozenset[str]
    closed: bool
    recurrent: bool
    #: Horizon in seconds -> irreducible rate at that horizon. The spectrum is
    #: the fundamental object; a single tau-star is a summary of it.
    spectrum: dict[float, float] = field(default_factory=dict)
    leak: float = 0.0
    note: str = ""

    @property
    def admissible(self) -> bool:
        return self.closed and self.recurrent

    @property
    def nonzero(self) -> bool:
        """Some horizon at which cutting it changes what it does."""
        return any(value > 0.0 for value in self.spectrum.values())

    @property
    def tau_star(self) -> float | None:
        """The horizon of the largest rate. A summary, never a law."""
        if not self.spectrum:
            return None
        return max(self.spectrum, key=lambda tau: self.spectrum[tau])

    @property
    def peak(self) -> float:
        return max(self.spectrum.values(), default=0.0)

    @property
    def carrier(self) -> bool:
        """F_intrinsic > 0: admissible, and irreducible somewhere."""
        return self.admissible and self.nonzero

    def as_dict(self) -> dict[str, Any]:
        return {
            "support": "".join(sorted(self.support)),
            "closed": self.closed,
            "recurrent": self.recurrent,
            "admissible": self.admissible,
            "leak": round(self.leak, 6),
            "spectrum": {str(round(tau, 6)): round(value, 6) for tau, value in sorted(self.spectrum.items())},
            "tau_star_seconds": self.tau_star,
            "peak_rate": round(self.peak, 6),
            "is_carrier": self.carrier,
            "note": self.note,
        }


def dominates(left: Mapping[float, float], right: Mapping[float, float]) -> bool:
    """`left >= right` at every shared horizon, and strictly greater somewhere.

    Horizons neither measured at are not evidence either way, so the comparison
    runs over the horizons both were measured at. With none in common the
    spectra are incomparable and this is False, which keeps both on the
    frontier rather than letting an unmeasured gap decide.
    """
    shared = set(left) & set(right)
    if not shared:
        return False
    if not all(left[tau] >= right[tau] for tau in shared):
        return False
    return any(left[tau] > right[tau] for tau in shared)


def pareto_frontier(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """The non-dominated carriers among those that overlap.

    Disjoint candidates do not compete: two processes that share no part are
    two subjects, not one subject and a loser. Only overlap triggers exclusion.
    """
    live = [c for c in candidates if c.carrier]
    out: list[Candidate] = []
    for candidate in live:
        beaten = False
        for other in live:
            if other is candidate or not (candidate.support & other.support):
                continue
            if dominates(other.spectrum, candidate.spectrum):
                beaten = True
                break
        if not beaten:
            out.append(candidate)
    return tuple(sorted(out, key=lambda c: (-len(c.support), "".join(sorted(c.support)))))


@dataclass
class ExclusionReport:
    """What the carrier search found, and whether it found one thing."""

    components: list[tuple[str, ...]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    frontier: tuple[Candidate, ...] = ()
    whole_core_is_one_component: bool = False
    whole_core_closed: bool = False

    @property
    def unique_carrier(self) -> Candidate | None:
        return self.frontier[0] if len(self.frontier) == 1 else None

    @property
    def symmetry_class(self) -> tuple[str, ...]:
        """Named supports that tie. Non-empty means the physics did not decide."""
        if len(self.frontier) <= 1:
            return ()
        return tuple("".join(sorted(c.support)) for c in self.frontier)

    @property
    def status(self) -> str:
        if not self.frontier:
            return "NOT_FOUND"
        if len(self.frontier) == 1:
            return "FOUND"
        overlapping = any(
            a.support & b.support
            for index, a in enumerate(self.frontier)
            for b in self.frontier[index + 1 :]
        )
        return "SYMMETRY_CLASS" if overlapping else "FOUND"

    def as_dict(self) -> dict[str, Any]:
        carrier = self.unique_carrier
        return {
            "status": self.status,
            "components": ["".join(block) for block in self.components],
            "whole_core_is_one_component": self.whole_core_is_one_component,
            "whole_core_closed": self.whole_core_closed,
            "the_core_is_the_unique_domain_level_carrier": bool(
                self.whole_core_is_one_component and self.whole_core_closed
            ),
            "candidates": [c.as_dict() for c in self.candidates],
            "frontier": ["".join(sorted(c.support)) for c in self.frontier],
            "symmetry_class": list(self.symmetry_class),
            "selected_carrier": None if carrier is None else "".join(sorted(carrier.support)),
            "selected_tau_star_seconds": None if carrier is None else carrier.tau_star,
            "selected_peak_rate": None if carrier is None else round(carrier.peak, 6),
        }


def select_carriers(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    spectra: Mapping[frozenset[str], Mapping[float, float]],
    closure: Mapping[frozenset[str], tuple[bool, float]],
) -> ExclusionReport:
    """Carriers from the graph, the closure result and the measured spectra.

    `spectra` and `closure` are keyed by support. A support with no spectrum is
    recorded as a candidate that was never measured rather than as one that
    scored zero — the difference between those two is the whole reason the
    authority gate exists.
    """
    components = strongly_connected_components(nodes, edges)
    whole = frozenset(nodes)
    one_component = bool(components) and frozenset(components[0]) == whole and len(components) == 1

    candidates: list[Candidate] = []
    for block in components:
        support = frozenset(block)
        # A single node with no self-loop is not a recurrent process; Tarjan
        # returns it as a component of one all the same.
        recurrent = len(block) > 1 or any(a == b == block[0] for a, b in edges)
        closed, leak = closure.get(support, (False, float("nan")))
        spectrum = dict(spectra.get(support, {}))
        candidates.append(
            Candidate(
                support=support,
                closed=bool(closed),
                recurrent=bool(recurrent),
                spectrum=spectrum,
                leak=float(leak) if leak == leak else 0.0,
                note="" if spectrum else "no spectrum was measured for this support",
            )
        )

    whole_closed = bool(closure.get(whole, (False, 0.0))[0])
    return ExclusionReport(
        components=components,
        candidates=candidates,
        frontier=pareto_frontier(candidates),
        whole_core_is_one_component=one_component,
        whole_core_closed=whole_closed,
    )
