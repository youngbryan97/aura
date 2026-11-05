"""The regrouping screen holds and composes her variables as the clamp does (core/subject/regroup.py).

These hold the construction, never the outcome. Every column is in one unit; every grouping is a partition; with her
declared domains as the groups, the unit clamp captures what the domain clamp captures and the unit compose builds
the rows the domain compose builds; and the sweep's two functions are swapped in and put back.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.subject import regroup
from core.subject.clamp import Clamp, compose
from core.subject.state import _SCHEMAS, DOMAINS, CoreState


def test_every_column_is_in_exactly_one_unit():
    units = regroup.units()
    for domain in DOMAINS:
        columns = sorted(c for u in units if u.domain == domain for c in u.columns)
        assert columns == list(range(len(_SCHEMAS[domain].features)))


@pytest.mark.parametrize("kind", ["declared", "random:10:1", "random:10:2", "fine"])
def test_every_grouping_is_a_partition_with_single_letter_names(kind):
    units = regroup.units()
    groups = regroup.grouping(kind, units)
    placed = [u.name for group in groups.values() for u in group]
    assert sorted(placed) == sorted(u.name for u in units)
    assert all(len(name) == 1 and name.islower() for name in groups)


def test_the_random_groupings_mix_domains_and_the_declared_one_does_not():
    units = regroup.units()
    assert all(len({u.domain for u in g}) == 1 for g in regroup.grouping("declared", units).values())
    for kind in ("random:10:1", "random:10:2"):
        assert all(len({u.domain for u in g}) >= 3 for g in regroup.grouping(kind, units).values())
    assert regroup.grouping("random:10:1", units) != regroup.grouping("random:10:2", units)


def _row(seed: int) -> CoreState:
    rng = np.random.default_rng(seed)
    return CoreState(values={d: rng.normal(size=len(_SCHEMAS[d].features)) for d in DOMAINS}, t=float(seed))


def test_composing_by_units_is_composing_by_domain_when_the_groups_are_her_domains():
    units = regroup.units()
    declared = regroup.grouping("declared", units)
    by_domain = dict(zip(DOMAINS, declared, strict=True))
    left_domains = ("P", "A", "S", "N")
    left_units = [u for d in left_domains for u in declared[by_domain[d]]]
    a, b = [_row(1), _row(2)], [_row(3), _row(4)]
    mine = regroup.compose_units(a, b, left_units)
    theirs = compose(a, b, left_domains)
    for m, t in zip(mine, theirs, strict=True):
        assert all(np.array_equal(m.values[d], t.values[d]) for d in DOMAINS)


def _holder():
    from core.state.aura_state import AuraState

    return SimpleNamespace(
        state=AuraState.default(), organs=None, kernel=None,
        ontogeny=SimpleNamespace(h=np.zeros(3), steps=4, era=1, last_novelty=0.1, last_displacement=0.2,
                                 last_relative_displacement=0.3),
    )


@pytest.mark.parametrize("domain", ["A", "G", "S", "N"])
def test_the_unit_clamp_holds_what_the_domain_clamp_holds(domain):
    holder = _holder()
    units = [u for u in regroup.units() if u.domain == domain]
    mine, theirs = regroup.UnitClamp(holder, units), Clamp(holder, [domain])
    assert mine.values.keys() == theirs.values.keys()
    assert mine.organs.keys() == theirs.organs.keys()
    assert len(mine.held) == len(theirs.held)
    assert mine.reservoir.keys() == theirs.reservoir.keys()


def test_a_unit_clamp_holds_only_its_own_units():
    holder = _holder()
    a_units = [u for u in regroup.units() if u.domain == "A"]
    first = a_units[0]
    clamp = regroup.UnitClamp(holder, [first])
    assert set(clamp.values) <= set(first.fields)


def test_the_sweep_s_clamp_and_compose_are_swapped_in_and_put_back():
    from core.subject import v25_runtime

    before = (v25_runtime.clamped, v25_runtime.compose)
    groups = regroup.grouping("random:10:1")
    with regroup.patched(groups):
        assert (v25_runtime.clamped, v25_runtime.compose) != before
    assert (v25_runtime.clamped, v25_runtime.compose) == before


def test_a_held_column_that_moved_is_named():
    units = regroup.units()
    unit = next(u for u in units if u.domain == "A" and u.columns)
    rows = [_row(1), _row(1)]
    assert regroup.unheld(rows, [unit]) == {}
    moved = CoreState(values={d: v.copy() for d, v in rows[0].values.items()}, t=2.0)
    moved.values["A"][unit.columns[0]] += 1.0
    names = regroup.unheld([rows[0], moved], [unit])
    assert len(names) == 1 and next(iter(names)).startswith("A.")
