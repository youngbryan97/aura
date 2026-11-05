"""Her irreducibility under other groupings of the same variables: the regrouping screen.

Preregistered in docs/ISC_V6_PREREGISTRATION.md (addendum of 5 November 2026, the regrouping screen).

The carrier sweep cuts her at the grain of her ten declared domains. An objection the paper names is that ten
domains her builder chose make irreducibility easy, and that it might hold only there. The screen asks whether it
holds when the same variables are grouped otherwise.

Holding a domain still is done by `core/subject/clamp.Clamp`, which captures the state fields the domain's readers
read, the organs they read, what K* holds of the domain beyond its columns, and, for development, the reservoir, and
writes them back after every phase. A domain is a set of those pieces. Here each piece is a unit:

    one field of `CLAMPED_FIELDS`,  one organ a domain's readers read,  a domain's K* holdings,  N's reservoir,

and every one of her columns belongs to the unit whose source it reads. A grouping is a partition of the units into
named groups. A group is held by holding its units, through a `Clamp` of each domain it touches with everything but
its own units dropped, so the capture and the write-back are the clamp's own; and a cut's rows are composed column by
column from the side each column's unit is on. With these two in place of `clamped` and `compose`, the carrier sweep
runs unchanged on a grouping's cuts: the same anchors, horizons, looks and decision rule.

Three groupings are screened beside the declared one, which is rebuilt from units as the control: two random ones of
ten groups that mix every domain, and a finer one that splits each domain in two. Each group is read against the rest
of her (the singleton cuts), and before any cut is read, every group is held for a turn and its columns are checked
for anything that moved (`unheld`): a column that moves while its group is held is a channel the clamp did not sever,
and a grouping with one does not count.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.clamp import CLAMPED_FIELDS, HELD_TOLERANCE, Clamp, organs_read_by
from core.subject.state import _SCHEMAS, DOMAINS, domain_slices, feature_names

__all__ = ["GROUP_NAMES", "Unit", "grouping", "patched", "unheld", "units"]

#: Group names are single letters, lower case so none is a domain's, because the sweep names a cut by joining its
#: sides' names and reads a name back one character at a time.
GROUP_NAMES: str = "abcdefghijklmnopqrst"


@dataclass(frozen=True)
class Unit:
    """One piece of what holding a domain holds, and the columns of that domain that read it."""

    name: str
    domain: str
    fields: tuple[str, ...] = ()
    organs: tuple[str, ...] = ()
    held: bool = False
    reservoir: bool = False
    columns: tuple[int, ...] = ()


def _base(source: str) -> str:
    return re.sub(r"\[[^\]]*\].*$", "", source)


def units() -> tuple[Unit, ...]:
    """Every unit of every domain, each column of her state in exactly one of them.

    Raises when a column reads nothing a unit holds: a column no clamp can hold would move inside its own lesion.
    """
    from core.subject.carrier_reads import held_by_domain

    out: list[Unit] = []
    for domain in DOMAINS:
        schema = _SCHEMAS[domain]
        fields = CLAMPED_FIELDS.get(domain, ())
        owner: list[str] = []
        for feature, source in zip(schema.features, schema.sources, strict=True):
            if source.startswith("organ:"):
                owner.append("organ:" + source[len("organ:"):].split(".", 1)[0])
            elif domain == "N":
                owner.append("reservoir")
            else:
                base = _base(source)
                match = max(
                    (f for f in fields if base == f or base.startswith(f + ".") or f.startswith(base + ".")),
                    key=len, default=None,
                )
                if match is None:
                    raise ValueError(f"{domain}.{feature} reads {source}, which no unit of {domain} holds")
                owner.append(match)
        def cols(key: str, column_owner: list[str] = owner) -> tuple[int, ...]:
            return tuple(i for i, o in enumerate(column_owner) if o == key)

        for field in fields:
            out.append(Unit(f"{domain}:{field}", domain, fields=(field,), columns=cols(field)))
        for organ in organs_read_by([domain]):
            out.append(Unit(f"{domain}:organ:{organ}", domain, organs=(organ,), columns=cols("organ:" + organ)))
        if domain == "N":
            out.append(Unit("N:reservoir", "N", reservoir=True, columns=cols("reservoir")))
        if held_by_domain(domain):
            out.append(Unit(f"{domain}:held", domain, held=True))
    covered = sorted((u.domain, i) for u in out for i in u.columns)
    expected = sorted((d, i) for d in DOMAINS for i in range(len(_SCHEMAS[d].features)))
    if covered != expected:
        raise ValueError("the units do not cover every column exactly once")
    return tuple(out)


def grouping(kind: str, every: Sequence[Unit] | None = None) -> dict[str, tuple[Unit, ...]]:
    """A partition of the units into named groups.

    `declared`: one group per domain, every unit of it (the control). `random:K:SEED`: the units shuffled and dealt
    into K groups in turn, so every group mixes domains. `fine`: each domain's units in two groups, its first half and
    its second, and a domain of one unit kept whole.
    """
    every = tuple(every if every is not None else units())
    if kind == "declared":
        groups = [tuple(u for u in every if u.domain == d) for d in DOMAINS]
    elif kind.startswith("random:"):
        _, count, seed = kind.split(":")
        order = list(every)
        np.random.default_rng(int(seed)).shuffle(order)
        groups = [tuple(order[i::int(count)]) for i in range(int(count))]
    elif kind == "fine":
        groups = []
        for d in DOMAINS:
            own = [u for u in every if u.domain == d]
            half = (len(own) + 1) // 2
            groups.extend([tuple(own[:half]), tuple(own[half:])] if len(own) > 1 else [tuple(own)])
    else:
        raise ValueError(f"no grouping named {kind!r}")
    groups = [g for g in groups if g]
    if len(groups) > len(GROUP_NAMES):
        raise ValueError(f"{len(groups)} groups is more than there are names for")
    return {GROUP_NAMES[i]: g for i, g in enumerate(groups)}


class UnitClamp(Clamp):
    """A `Clamp` over units: each domain the units touch captured by the clamp's own code, and everything but the
    units' own pieces dropped. `apply` is the clamp's."""

    def __init__(self, holder: Any, held_units: Sequence[Unit]) -> None:
        self.holder = holder
        self.units = tuple(held_units)
        self.domains = ()
        self.values: dict[str, Any] = {}
        self.hidden: Any = None
        self.reservoir: dict[str, Any] = {}
        self.organs: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.held: list[tuple[Any, str, Any, bool]] = []
        self.capture()

    def capture(self) -> None:
        fields = {f for u in self.units for f in u.fields}
        organs = {o for u in self.units for o in u.organs}
        held = {u.domain for u in self.units if u.held}
        reservoir = any(u.reservoir for u in self.units)
        self.values, self.organs, self.held, self.reservoir, self.hidden = {}, {}, [], {}, None
        for domain in (d for d in DOMAINS if any(u.domain == d for u in self.units)):
            part = Clamp(self.holder, [domain])
            self.values.update({path: v for path, v in part.values.items() if path in fields})
            self.organs.update({name: v for name, v in part.organs.items() if name in organs})
            if domain in held:
                self.held.extend(part.held)
            if domain == "N" and reservoir:
                self.hidden, self.reservoir = part.hidden, dict(part.reservoir)


def _masks(of: Sequence[Unit]) -> dict[str, np.ndarray]:
    masks = {d: np.zeros(len(_SCHEMAS[d].features), dtype=bool) for d in DOMAINS}
    for u in of:
        masks[u.domain][list(u.columns)] = True
    return masks


def compose_units(left_rows: Sequence[Any], right_rows: Sequence[Any], left_units: Sequence[Unit]) -> list[Any]:
    """`clamp.compose` column by column: each column from the run in which its unit's side was free."""
    import dataclasses

    from core.subject.state import UnreadFrame

    masks = _masks(left_units)
    out: list[Any] = []
    for a, b in zip(left_rows, right_rows, strict=False):
        if isinstance(a, UnreadFrame) or isinstance(b, UnreadFrame):
            out.append(a if isinstance(a, UnreadFrame) else b)
            continue
        values = {d: np.where(masks[d], a.values[d], b.values[d]) for d in DOMAINS}
        misses = dict(getattr(a, "misses", {}) or {})
        misses.update(getattr(b, "misses", {}) or {})
        out.append(dataclasses.replace(a, values=values, misses=misses))
    return out


def unheld(rows: Sequence[Any], held_units: Sequence[Unit], *, tolerance: float = HELD_TOLERANCE) -> dict[str, float]:
    """`clamp.still` for units: every column of these units that moved across these readings, and how far."""
    if not rows:
        return {}
    matrix = np.vstack([row.vector() for row in rows])
    slices = domain_slices()
    moved: dict[str, float] = {}
    for u in held_units:
        names = feature_names(u.domain)
        for index in u.columns:
            column = matrix[:, slices[u.domain].start + index]
            amount = float(np.max(np.abs(column - column[0])))
            if amount > tolerance:
                moved[f"{u.domain}.{names[index]}"] = round(amount, 6)
    return moved


@contextlib.contextmanager
def patched(groups: Mapping[str, Sequence[Unit]]) -> Iterator[None]:
    """The carrier sweep with these groups as its "domains": `clamped` and `compose` in `core.subject.v25_runtime`
    swapped for the unit versions while inside, and put back after."""
    from core.subject import v25_runtime

    def members(names: Sequence[str]) -> list[Unit]:
        return [u for name in names for u in groups[name]]

    @contextlib.contextmanager
    def clamped(runtime: Any, names: Sequence[str]) -> Iterator[UnitClamp]:
        clamp = UnitClamp(runtime, members(names))
        previous = getattr(runtime, "after_phase", None)
        runtime.after_phase = clamp.apply
        try:
            yield clamp
        finally:
            runtime.after_phase = previous

    def compose(left_rows: Sequence[Any], right_rows: Sequence[Any], left: Sequence[str]) -> list[Any]:
        return compose_units(left_rows, right_rows, members(left))

    before = (v25_runtime.clamped, v25_runtime.compose)
    v25_runtime.clamped, v25_runtime.compose = clamped, compose
    try:
        yield
    finally:
        v25_runtime.clamped, v25_runtime.compose = before
