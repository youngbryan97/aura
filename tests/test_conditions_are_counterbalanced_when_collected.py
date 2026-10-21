"""Each condition follows every other one equally often when the order is counterbalanced."""
from __future__ import annotations

from collections import Counter

import pytest

from core.subject.counterbalance import round_orders, williams_square


@pytest.mark.parametrize("n", [4, 6, 8])
def test_a_williams_square_puts_each_condition_after_every_other_once(n: int):
    square = williams_square(n)
    assert all(sorted(row) == list(range(n)) for row in square)
    assert all(sorted(column) == list(range(n)) for column in zip(*square, strict=True))
    pairs = Counter((row[i], row[i + 1]) for row in square for i in range(n - 1))
    assert set(pairs.values()) == {1}
    assert len(pairs) == n * (n - 1)


def test_rounds_cycle_through_the_rows_in_a_seeded_random_order():
    names = list("abcdefgh")
    orders = round_orders(names, 24, seed=7)
    assert len(orders) == 24
    assert all(sorted(order) == names for order in orders)
    assert round_orders(names, 24, seed=7) == orders
    assert round_orders(names, 24, seed=8) != orders
    first_block = {tuple(order) for order in orders[:8]}
    assert len(first_block) == 8
