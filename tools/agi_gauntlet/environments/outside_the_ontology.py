"""Tasks somebody else designed, before this solver existed.

Gate 1 seals its rules after the freeze, so the instances are genuinely fresh.
The criticism that lands is about the ONTOLOGY rather than the instances: the
generator composes offsets, mirrors, end exchanges, groupings and value maps,
and the native induction machinery represents offsets, mirrors, exchanges,
ends, groupings and affine maps. A new instance is not a new hypothesis
family, so a high score there is evidence about composition and search inside
a representational universe the evaluator shares with the solver.

The control that fixes this cannot be written here, because anything written
here is written by the same hand. It has to come from outside: a task family
designed by somebody who had never seen this code, whose primitives were
chosen for other reasons.

ARC-AGI is that family. Published in 2019 by François Chollet, 400 evaluation
tasks, each a handful of worked grid examples and one held out. Its ontology
is objectness, counting, containment, symmetry and goal-directedness — chosen
as a claim about core knowledge, not as a claim about any program.

It is not a perfect control. The tasks are public, so a language model could
have memorised them; that is why this runs the SYMBOLIC induction and no model
at all. Nothing here can recall an answer, so a solved task was searched for.

The dataset is not vendored. Point ``AGI_GAUNTLET_OUTSIDE_TASKS`` at a
directory of task JSON and the gate runs; leave it unset and the gate says
what it needs, the same as the ones that need an evaluator.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ATaskFromOutside", "read_the_outside_tasks", "where_they_live"]


@dataclass(frozen=True)
class ATaskFromOutside:
    """One task: worked examples, and the one that is held out."""

    name: str
    shown: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    asked: tuple[int, ...]
    answer: tuple[int, ...]
    #: Rows and columns of every grid, so a solver may know the shape without
    #: being told anything about the task.
    shape_in: tuple[int, int]
    shape_out: tuple[int, int]

    @property
    def same_shape(self) -> bool:
        return self.shape_in == self.shape_out


def where_they_live() -> Path | None:
    raw = os.environ.get("AGI_GAUNTLET_OUTSIDE_TASKS", "").strip()
    if not raw:
        return None
    place = Path(raw).expanduser()
    return place if place.is_dir() else None


def _flat(grid: list[list[int]]) -> tuple[int, ...]:
    return tuple(int(cell) for row in grid for cell in row)


def read_the_outside_tasks(place: Path, *, most: int = 0) -> tuple[list[ATaskFromOutside], str]:
    """Every task in a directory, and a digest of exactly what was read.

    The digest is the receipt. A number against a dataset nobody can identify
    is not a measurement, and "ARC-AGI" names three different releases.
    """

    found: list[ATaskFromOutside] = []
    digest = hashlib.sha256()
    for path in sorted(place.glob("*.json")):
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(raw)
        try:
            body = json.loads(raw)
            shown = tuple(
                (_flat(pair["input"]), _flat(pair["output"])) for pair in body["train"]
            )
            asked = body["test"][0]
            first = body["train"][0]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not shown:
            continue
        found.append(
            ATaskFromOutside(
                name=path.stem,
                shown=shown,
                asked=_flat(asked["input"]),
                answer=_flat(asked["output"]),
                shape_in=(len(asked["input"]), len(asked["input"][0])),
                shape_out=(len(asked["output"]), len(asked["output"][0])),
            )
        )
        if most and len(found) >= most:
            break
    return found, digest.hexdigest()[:16]
