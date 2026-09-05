"""core/environment/prospect_refuge.py — where to be, and why that place.

Somewhere to sit with your back to a wall and a view of the door. A window
seat. The gap behind a hedge. A booth rather than the middle of a room. Jay
Appleton's name for what these have in common is prospect and refuge: seeing
without being seen. Whether the evolutionary story behind it is right is
argued about, and the argument does not matter for what follows, because the
quantity itself is well defined and computable on anything with a visibility
relation.

The quantity is an asymmetry. A position gives you a set of things you can
observe from it, and it belongs to a set of positions from which you can be
observed. Those two sets are different, and the difference is the whole
subject. A stage has enormous prospect and no refuge. A cupboard has refuge
and no prospect. The positions people go looking for have both, and there are
usually few of them.

## Refuge is not in the sightlines

Worth saying plainly, because building it the obvious way produces a module
that cannot represent the thing it is named after. Physical sight is
symmetric: if a sightline runs from here to there, it runs back. Derive both
prospect and exposure from one visibility relation and they come out equal at
every position, the asymmetry is identically zero everywhere, and what is left
is a degree count wearing Appleton's vocabulary.

Seeing without being seen needs a second property that the sightline graph
does not contain — cover. The hedge, the booth, the dim corner, the low
profile. It is held per position as ``concealment``, it is measured or
supplied rather than inferred, and it is the only reason any position in this
module can score differently on the two terms.

## The two terms do not get added up

Stamps and Dosen's meta-analysis of the environmental preference literature
found the two halves supported unequally: prospect holds up across studies of
urban, interior and natural settings, refuge holds up in natural settings and
comes out inconsistent elsewhere. So a composite score is the wrong output.
This module reports prospect and exposure separately and takes the weighting
from the caller, because collapsing an unevenly-supported pair into one number
buries the half that might be wrong, and no reader downstream can recover it.

## Crevices are a structural property, not a metaphor

The small enclosed place that is nonetheless well placed has a definition:
few ways in, much visible out. Low degree, high prospect. On a graph that is
one line of arithmetic, and it means the same thing for a physical alcove, a
position in a conversation, and a module that reads a great deal of the system
while almost nothing calls it.

## Where you should be is a separate question again

Prospect and refuge are about the position. Fit is about the match between
what a position demands and what you have. The best vantage in a place you
cannot function is not where you should be, and a system that scores only
position will keep sending you there. ``fit`` is the third term and it is also
kept separate.

The module has no notion of physical space. A graph, a visibility relation and
optionally a demand vector per node is all it takes, and grids and rooms are
one adapter away.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Environment.ProspectRefuge")


@dataclass(frozen=True)
class Position:
    """One place, scored on terms that are never summed for you."""

    key: str
    prospect: float
    """Share of the space observable from here, in [0, 1]."""

    exposure: float
    """Share of the space that can actually observe you here, in [0, 1].

    Sightlines onto this position, discounted by what covers it. Equal to
    ``sightlines`` when nothing does.
    """

    sightlines_onto: float
    """Share of the space with a line of sight here, before any cover."""

    concealment: float
    """How much of you the position hides, in [0, 1]."""

    approaches: int
    """How many ways in. The count that makes an alcove an alcove."""

    fit: float | None = None
    """Match between what this position demands and what the agent holds."""

    @property
    def refuge(self) -> float:
        return 1.0 - self.exposure

    @property
    def asymmetry(self) -> float:
        """How much more you see than is seen of you. The quantity itself."""
        return self.prospect - self.exposure

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prospect": round(self.prospect, 4),
            "exposure": round(self.exposure, 4),
            "sightlines_onto": round(self.sightlines_onto, 4),
            "concealment": round(self.concealment, 4),
            "refuge": round(self.refuge, 4),
            "asymmetry": round(self.asymmetry, 4),
            "approaches": self.approaches,
            "fit": None if self.fit is None else round(self.fit, 4),
        }


@dataclass
class VisibilityField:
    """A space, what can be seen from where, and what each place asks of you.

    ``visible`` maps a position to everything observable from it. The relation
    is deliberately allowed to be asymmetric — that is the point, and a model
    that quietly symmetrises it has removed the phenomenon before measuring
    it. Being seen from the doorway while not being able to see the doorway is
    an ordinary situation and the commonest reason a seat is wrong.
    """

    visible: dict[str, set[str]] = field(default_factory=dict)
    adjacency: dict[str, set[str]] = field(default_factory=dict)
    demands: dict[str, dict[str, float]] = field(default_factory=dict)
    concealment: dict[str, float] = field(default_factory=dict)

    def add(self, key: str, *, sees: Iterable[str] = (),
            reached_from: Iterable[str] = (),
            demands: Mapping[str, float] | None = None,
            concealment: float = 0.0) -> None:
        self.visible.setdefault(key, set()).update(sees)
        self.adjacency.setdefault(key, set()).update(reached_from)
        for other in reached_from:
            self.adjacency.setdefault(other, set()).add(key)
        if demands:
            self.demands[key] = dict(demands)
        if concealment:
            self.concealment[key] = min(max(float(concealment), 0.0), 1.0)

    def keys(self) -> list[str]:
        return sorted(
            set(self.visible) | set(self.adjacency)
            | set(self.demands) | set(self.concealment)
        )

    def conceal(self, key: str, amount: float) -> None:
        """Say how much cover a position gives. The refuge half comes from here."""
        self.concealment[key] = min(max(float(amount), 0.0), 1.0)

    def _seen_by(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {k: set() for k in self.keys()}
        for observer, targets in self.visible.items():
            for target in targets:
                out.setdefault(target, set()).add(observer)
        return out

    def score(self, capabilities: Mapping[str, float] | None = None) -> list[Position]:
        """Score every position. Nothing is combined and nothing is ranked."""
        keys = self.keys()
        n = max(len(keys), 1)
        seen_by = self._seen_by()
        out: list[Position] = []
        for key in keys:
            sees = self.visible.get(key, set()) - {key}
            watchers = seen_by.get(key, set()) - {key}
            cover = self.concealment.get(key, 0.0)
            sightlines = len(watchers) / max(n - 1, 1)
            out.append(
                Position(
                    key=key,
                    prospect=len(sees) / max(n - 1, 1),
                    exposure=sightlines * (1.0 - cover),
                    sightlines_onto=sightlines,
                    concealment=cover,
                    approaches=len(self.adjacency.get(key, set())),
                    fit=self._fit(key, capabilities),
                )
            )
        return out

    def _fit(self, key: str, capabilities: Mapping[str, float] | None) -> float | None:
        """How well what is held matches what this place asks for.

        Excess capability is not counted as fit and not counted against it.
        Being overqualified for a spot is neither a match nor a mismatch, and
        scoring it either way makes the measure answer a different question
        from the one it is being asked.
        """
        demand = self.demands.get(key)
        if not demand or capabilities is None:
            return None
        total = sum(max(0.0, v) for v in demand.values())
        if total <= 0:
            return 1.0
        met = sum(
            min(max(0.0, v), max(0.0, float(capabilities.get(k, 0.0))))
            for k, v in demand.items()
        )
        return met / total

    # ------------------------------------------------------------- queries

    def crevices(self, *, capabilities: Mapping[str, float] | None = None,
                 max_approaches: int | None = None) -> list[Position]:
        """Enclosed positions with a view, best first.

        A ranking rather than a set. In an open room most of the edge has few
        ways in and a fair view, and returning a short list would mean a
        cutoff chosen to make the list short. The medians keep "few ways in"
        and "a fair view" meaning the same thing in a corridor and in a
        lattice, and the ordering — asymmetry, then prospect — is what
        actually answers the question, so a caller takes as many as it wants
        from the front.
        """
        scored = self.score(capabilities)
        if not scored:
            return []
        prospects = sorted(p.prospect for p in scored)
        degrees = sorted(p.approaches for p in scored)
        mid = len(scored) // 2
        prospect_floor = prospects[mid]
        degree_ceiling = (
            max_approaches if max_approaches is not None else degrees[mid]
        )
        found = [
            p for p in scored
            if p.approaches <= degree_ceiling and p.prospect >= prospect_floor
        ]
        # Ordered by the asymmetry first: among places with a view and few ways
        # in, the ones that also hide you are the ones being asked about.
        found.sort(key=lambda p: (p.asymmetry, p.prospect), reverse=True)
        return found

    def rank(
        self,
        *,
        prospect_weight: float,
        refuge_weight: float,
        fit_weight: float = 0.0,
        capabilities: Mapping[str, float] | None = None,
    ) -> list[tuple[Position, float]]:
        """Order positions under weights the caller supplies.

        There is no default weighting and there will not be one. The evidence
        for the two halves is not equally strong, so a default would be this
        module asserting a resolution to a question the literature has not
        resolved, in a place where nobody downstream would see it.
        """
        scored = self.score(capabilities)
        ranked: list[tuple[Position, float]] = []
        for position in scored:
            value = (
                prospect_weight * position.prospect
                + refuge_weight * position.refuge
            )
            if fit_weight and position.fit is not None:
                value += fit_weight * position.fit
            ranked.append((position, value))
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def status(self, capabilities: Mapping[str, float] | None = None) -> dict[str, Any]:
        scored = self.score(capabilities)
        return {
            "positions": len(scored),
            "mean_prospect": round(
                sum(p.prospect for p in scored) / len(scored), 4) if scored else None,
            "mean_exposure": round(
                sum(p.exposure for p in scored) / len(scored), 4) if scored else None,
            # Zero everywhere means no cover was ever supplied, so the refuge
            # half of this field is not being modelled at all. Silent in a
            # composite score; named here.
            "refuge_modelled": any(p.concealment > 0 for p in scored),
            "crevices": [p.key for p in self.crevices(capabilities=capabilities)],
            "most_exposed": max(scored, key=lambda p: p.exposure).key if scored else None,
        }


def grid_field(
    width: int,
    height: int,
    *,
    walls: Iterable[tuple[int, int]] = (),
    sight: int | None = None,
    cover: Mapping[tuple[int, int], float] | None = None,
) -> VisibilityField:
    """Build a field from a rectangular grid with walls.

    Line of sight is taken along rows and columns and stops at a wall, which
    is the cheap approximation and the honest one to name: a diagonal sightline
    is not counted, so an open room scores slightly lower than it should and a
    corner scores correctly. The adapter exists so the graph code above can be
    exercised against something with an answer a person can check by looking.
    """
    blocked = {tuple(w) for w in walls}
    field_obj = VisibilityField()
    cells = [
        (x, y) for y in range(height) for x in range(width) if (x, y) not in blocked
    ]

    def key(cell: tuple[int, int]) -> str:
        return f"{cell[0]},{cell[1]}"

    for cell in cells:
        x, y = cell
        sees: set[str] = set()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            step = 1
            while sight is None or step <= sight:
                nxt = (x + dx * step, y + dy * step)
                if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                    break
                if nxt in blocked:
                    break
                sees.add(key(nxt))
                step += 1
        neighbours = {
            key((x + dx, y + dy))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (x + dx, y + dy) in set(cells)
        }
        field_obj.add(
            key(cell), sees=sees, reached_from=neighbours,
            concealment=float(cover.get(cell, 0.0)) if cover else 0.0,
        )
    return field_obj


def graph_field(
    edges: Sequence[tuple[str, str]],
    *,
    visibility: Mapping[str, Iterable[str]] | None = None,
    demands: Mapping[str, Mapping[str, float]] | None = None,
    cover: Mapping[str, float] | None = None,
) -> VisibilityField:
    """Build a field from a graph, taking neighbours as sightlines by default."""
    field_obj = VisibilityField()
    neighbours: dict[str, set[str]] = {}
    for a, b in edges:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    for node, near in neighbours.items():
        sees = set(visibility[node]) if visibility and node in visibility else set(near)
        field_obj.add(
            node, sees=sees, reached_from=near,
            demands=dict(demands[node]) if demands and node in demands else None,
            concealment=float(cover.get(node, 0.0)) if cover else 0.0,
        )
    return field_obj
