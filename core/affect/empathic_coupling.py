"""core/affect/empathic_coupling.py — being moved by someone without becoming them.

Feeling what someone near you is feeling is a coupling. Their state pulls
yours, more strongly the closer you are, and the pull is what makes an
otherwise abstract fact about them into something you act on. Modelled as
diffusion on a graph of relationships, this is one of the standard consensus
dynamics and it has a well-known property:

**Pure diffusion converges to agreement.** Left alone, every state in a
connected graph runs to the same value. That is the correct model of a crowd
and the wrong model of a person. Someone whose state is entirely a function of
the states around them has no vantage point from which the others are other
people, and the caring stops being about them. It is also, in the ordinary
sense, how carers burn out: not from caring too much, but from the return path
having gone.

So the model needs a second term, and the second term is the whole content:

    dx_i/dt = -sum_j K_ij (x_i - x_j) - lambda_i (x_i - s_i)

A pull toward the people you are coupled to, and a pull back toward your own
resting state. The ratio between them decides whether there is anyone left to
do the feeling, and it is not a metaphor — it comes out of the algebra. At
rest,

    x* = (Lambda + L)^-1 Lambda s

where ``L`` is the graph Laplacian and ``Lambda`` the diagonal of anchor
strengths. The row of that matrix belonging to a person says exactly how much
of where they end up is their own set point and how much is everyone else's.
``autonomy`` returns the diagonal, ``attribution`` returns the whole row, and
the difference between a system that has this and one that does not is
visible in a single number.

Two properties fall out that are worth having:

**Coupling is asymmetric.** You are moved more by some people than they are by
you, ``K_ij`` need not equal ``K_ji``, and nothing in the solution requires
symmetry. A model that symmetrises has removed the most common shape a
relationship actually has.

**Zero anchor is a singular case, not a large one.** With every anchor at
zero the matrix is singular and there is no rest to return: the states drift
together and stop being anybody's. The solver reports that rather than
returning a plausible vector, because a plausible vector is exactly what a
system in that state would go on producing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Affect.Empathy")

#: Anchor below which a person is treated as having no resting state of their
#: own. Not zero, because the matrix is singular there and a solver that
#: returns something anyway is the failure this module is about.
MIN_ANCHOR = 1e-9

#: Share of a rest state that has to come from one's own set point before the
#: state counts as still one's own. Half: below it, most of where you are is
#: somebody else's, which is the line between being moved and being merged.
AUTONOMY_FLOOR = 0.5


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. Nothing when singular.

    Returning nothing rather than raising, because singular here is a
    meaningful state of the system — everyone anchored to nobody — and the
    caller has something to say about it.
    """
    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col] / a[col][col]
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    return [a[i][n] / a[i][i] for i in range(n)]


@dataclass
class EmpathicField:
    """People, how much each moves the others, and how anchored each is.

    ``coupling[(a, b)]`` is how much b's state pulls a's. The pair is ordered
    and the two directions are separate entries.
    """

    setpoints: dict[str, float] = field(default_factory=dict)
    anchors: dict[str, float] = field(default_factory=dict)
    coupling: dict[tuple[str, str], float] = field(default_factory=dict)
    observed: dict[str, float] = field(default_factory=dict)

    def add_person(self, key: str, *, setpoint: float = 0.0,
                   anchor: float = 1.0) -> None:
        self.setpoints[key] = float(setpoint)
        self.anchors[key] = max(0.0, float(anchor))

    def couple(self, subject: str, other: str, strength: float) -> None:
        """How much ``other``'s state pulls ``subject``'s."""
        self.coupling[(subject, other)] = max(0.0, float(strength))

    def observe(self, key: str, state: float) -> None:
        """Someone's state as read from outside. Used in place of a set point."""
        self.observed[key] = float(state)

    def people(self) -> list[str]:
        keys = set(self.setpoints) | set(self.anchors) | set(self.observed)
        for a, b in self.coupling:
            keys.add(a)
            keys.add(b)
        return sorted(keys)

    # ---------------------------------------------------------------- solve

    def _system(self, people: Sequence[str]) -> tuple[list[list[float]], list[float]]:
        n = len(people)
        index = {k: i for i, k in enumerate(people)}
        matrix = [[0.0] * n for _ in range(n)]
        rhs = [0.0] * n
        for key in people:
            i = index[key]
            anchor = self.anchors.get(key, 1.0)
            matrix[i][i] += anchor
            rhs[i] += anchor * self.setpoints.get(key, self.observed.get(key, 0.0))
            for other in people:
                if other == key:
                    continue
                k = self.coupling.get((key, other), 0.0)
                if k <= 0:
                    continue
                matrix[i][i] += k
                matrix[i][index[other]] -= k
        return matrix, rhs

    def rest(self) -> dict[str, float] | None:
        """Where everyone ends up. Nothing when nobody is anchored."""
        people = self.people()
        if not people:
            return {}
        if all(self.anchors.get(k, 1.0) <= MIN_ANCHOR for k in people):
            return None
        matrix, rhs = self._system(people)
        solution = solve(matrix, rhs)
        if solution is None:
            return None
        return dict(zip(people, solution, strict=True))

    def attribution(self, person: str) -> dict[str, float] | None:
        """Whose set point each part of this person's rest state came from.

        The row of the resolvent. It sums to one — the rest state is a convex
        combination of everyone's set points — so it reads directly as a
        share, and the entry for the person themselves is what is left of them.
        """
        people = self.people()
        if person not in people:
            return None
        matrix, _ = self._system(people)
        shares: dict[str, float] = {}
        for source in people:
            rhs = [
                (self.anchors.get(k, 1.0) if k == source else 0.0)
                for k in people
            ]
            solution = solve(matrix, rhs)
            if solution is None:
                return None
            shares[source] = solution[people.index(person)]
        return shares

    def autonomy(self, person: str) -> float | None:
        """How much of where this person ends up is their own set point."""
        shares = self.attribution(person)
        if shares is None:
            return None
        return shares.get(person)

    def merged(self) -> list[str]:
        """People whose state is now mostly somebody else's.

        The measurement the model exists for. It is not a warning about
        feeling too much; a strongly coupled person with a firm anchor does
        not appear here. It is about the return path being gone.
        """
        out: list[str] = []
        for key in self.people():
            own = self.autonomy(key)
            if own is not None and own < AUTONOMY_FLOOR:
                out.append(key)
        return out

    def step(self, dt: float = 0.05) -> dict[str, float]:
        """One step of the flow, for watching it move rather than solving it."""
        people = self.people()
        current = {k: self.observed.get(k, self.setpoints.get(k, 0.0)) for k in people}
        nxt: dict[str, float] = {}
        for key in people:
            pull = sum(
                self.coupling.get((key, other), 0.0) * (current[other] - current[key])
                for other in people if other != key
            )
            anchor = self.anchors.get(key, 1.0)
            drift = pull - anchor * (current[key] - self.setpoints.get(key, 0.0))
            nxt[key] = current[key] + dt * drift
        self.observed.update(nxt)
        return nxt

    def status(self) -> dict[str, Any]:
        rest = self.rest()
        return {
            "people": len(self.people()),
            "rest": None if rest is None else {k: round(v, 4) for k, v in rest.items()},
            "autonomy": {
                k: (None if self.autonomy(k) is None else round(self.autonomy(k), 4))
                for k in self.people()
            },
            "merged": self.merged(),
            # Nobody anchored anywhere. The states run together and stop being
            # anyone's, and a solver that answered would be describing a
            # consensus rather than a set of people.
            "no_rest": rest is None,
        }


_FIELD: EmpathicField | None = None


def get_empathic_field() -> EmpathicField:
    global _FIELD
    if _FIELD is None:
        _FIELD = EmpathicField()
    return _FIELD


def reset_empathic_field_for_test() -> None:
    global _FIELD
    _FIELD = None
