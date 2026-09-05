"""Which kind of memory a thing should be.

Aura can hold something symbolically, in an adapter, or in the weights, and
they are not the same kind of holding. Nothing chose between them: every
durable thing became a symbolic record because that is the cheap path, so the
model learned nothing from a year of conversation while the store filled with
facts nobody queried.
"""

from __future__ import annotations

import pytest

from core.memory.substrate_allocation import (
    CONFIDENCE_FOR_ADAPTER,
    CONFIDENCE_FOR_WEIGHTS,
    Item,
    Shelf,
    Substrate,
    allocate,
    allocate_all,
    capacity_report,
)


def _item(**over):
    base = dict(
        key="thing",
        use_rate=1.0,
        lifetime=100.0,
        value_per_use=0.5,
        confidence=0.9,
        breadth=0.3,
        needs_unbidden=False,
    )
    base.update(over)
    return Item(**base)


# ── the substrates are different kinds of holding ────────────────────────


def test_only_the_weights_are_irreversible():
    assert Substrate.WEIGHTS.reversible is False
    for substrate in (Substrate.SYMBOLIC, Substrate.ADAPTER, Substrate.DISCARD):
        assert substrate.reversible is True


def test_a_symbolic_record_does_not_apply_unless_something_looks_for_it():
    assert Substrate.SYMBOLIC.applies_unbidden is False
    assert Substrate.ADAPTER.applies_unbidden is True
    assert Substrate.WEIGHTS.applies_unbidden is True


# ── the decision ─────────────────────────────────────────────────────────


def test_something_looked_up_rarely_goes_in_the_store():
    result = allocate(_item(use_rate=0.05, lifetime=200, confidence=1.0, breadth=0.05))
    assert result.substrate is Substrate.SYMBOLIC


def test_something_needed_constantly_and_unbidden_earns_an_adapter():
    result = allocate(
        _item(use_rate=5.0, lifetime=400, confidence=0.99, breadth=0.2, needs_unbidden=True)
    )
    assert result.substrate is Substrate.ADAPTER


def test_a_one_off_is_not_held_anywhere():
    result = allocate(_item(use_rate=0.01, lifetime=2, value_per_use=0.2))
    assert result.substrate is Substrate.DISCARD


def test_a_record_nothing_will_look_for_is_worth_nothing():
    """The cheap path is not merely worse here, it is nothing."""
    quiet = allocate(_item(use_rate=0.05, lifetime=50, confidence=0.5, needs_unbidden=True))
    assert quiet.considered["symbolic"] < 0.0
    assert quiet.substrate is Substrate.DISCARD


# ── the confidence gates ─────────────────────────────────────────────────


def test_an_uncertain_thing_may_not_go_anywhere_irreversible():
    """An error in the weights is an error forever."""
    huge = dict(use_rate=50.0, lifetime=1000, confidence=CONFIDENCE_FOR_WEIGHTS - 0.01,
                breadth=0.15, needs_unbidden=True)
    result = allocate(_item(**huge), Shelf(adapters_held=16))
    assert result.substrate is not Substrate.WEIGHTS


def test_the_same_thing_one_point_more_confident_may():
    huge = dict(use_rate=50.0, lifetime=1000, confidence=CONFIDENCE_FOR_WEIGHTS,
                breadth=0.15, needs_unbidden=True)
    result = allocate(_item(**huge), Shelf(adapters_held=16))
    assert result.substrate is Substrate.WEIGHTS


def test_an_uncertain_thing_that_must_apply_unbidden_is_discarded_not_stored():
    """Storing it would be pretending it will help, and it will not."""
    result = allocate(
        _item(use_rate=1.0, lifetime=300, confidence=CONFIDENCE_FOR_ADAPTER - 0.1,
              needs_unbidden=True)
    )
    assert result.substrate is Substrate.DISCARD
    assert "without being asked" in result.because


def test_no_amount_of_benefit_buys_past_the_confidence_gate():
    result = allocate(
        _item(use_rate=10000.0, lifetime=10000, value_per_use=100.0, confidence=0.5),
        Shelf(adapters_held=16),
    )
    assert result.substrate is not Substrate.WEIGHTS


# ── adapter capacity is what makes the weights ever right ────────────────


def test_the_weights_branch_is_reachable():
    """Without contention it is unreachable, which makes it a dead branch."""
    core = _item(use_rate=50.0, lifetime=1000, confidence=0.99, breadth=0.15,
                 needs_unbidden=True)
    assert allocate(core, Shelf(adapters_held=0)).substrate is Substrate.ADAPTER
    assert allocate(core, Shelf(adapters_held=16)).substrate is Substrate.WEIGHTS


def test_a_crowded_shelf_makes_the_next_adapter_cost_more():
    empty = Shelf(adapters_held=0)
    full = Shelf(adapters_held=16)
    assert full.contention > empty.contention
    core = _item(use_rate=50.0, lifetime=1000, confidence=0.99, breadth=0.15)
    assert (
        allocate(core, full).considered["adapter"]
        < allocate(core, empty).considered["adapter"]
    )


def test_the_first_few_adapters_are_nearly_free():
    """Contention squared: a fixed budget does not bite until it is nearly gone."""
    assert Shelf(adapters_held=2).contention < 1.5
    assert Shelf(adapters_held=16).contention > 10.0


def test_a_batch_fills_the_shelf_as_it_goes():
    """Allocating independently is how a system gets more adapters than it can load."""
    items = [
        _item(key=f"k{i}", use_rate=5.0, lifetime=400, confidence=0.99, breadth=0.2,
              needs_unbidden=True)
        for i in range(20)
    ]
    allocations = allocate_all(items)
    adapters = sum(1 for a in allocations if a.substrate is Substrate.ADAPTER)
    assert adapters < len(items), (
        "every item became an adapter; the shelf never filled, so the batch "
        "was allocated as though capacity were infinite"
    )


# ── the record ───────────────────────────────────────────────────────────


def test_the_allocation_shows_what_was_close():
    result = allocate(_item(use_rate=5.0, lifetime=400, confidence=0.99))
    assert set(result.considered) == {"symbolic", "adapter", "weights"}
    assert result.because


def test_the_capacity_report_counts_what_cannot_be_undone():
    core = _item(use_rate=50.0, lifetime=1000, confidence=0.99, breadth=0.15,
                 needs_unbidden=True)
    report = capacity_report((allocate(core, Shelf(adapters_held=16)),))
    assert report["irreversible"] == 1


def test_expected_uses_is_rate_times_lifetime():
    assert _item(use_rate=2.0, lifetime=50).expected_uses == pytest.approx(100.0)
