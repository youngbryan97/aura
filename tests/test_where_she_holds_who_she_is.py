"""The rule that picks where to change who she is, fixed before any run: it keeps what she knows first."""
from __future__ import annotations

import pytest

from tools.where_she_holds_who_she_is import chosen

pytestmark = pytest.mark.unit

UNCHANGED = {"window": None, "push": None, "development": [3.0, 2.0], "knowledge": [4.0, 4.0]}


def test_the_lowest_development_leaning_that_keeps_what_she_knows():
    results = [UNCHANGED,
               {"window": [0, 64], "push": 2.0, "development": [-9.0, -9.0], "knowledge": [-1.0, 4.0]},  # forgets
               {"window": [16, 48], "push": None, "development": [-2.0, -1.0], "knowledge": [3.5, 3.5]},
               {"window": [16, 24], "push": 1.0, "development": [-2.0, -1.0], "knowledge": [3.2, 3.2]},  # fewer layers
               {"window": [24, 32], "push": 0.5, "development": [-4.0, -1.0], "knowledge": [2.0, 2.0]}]  # below 3/4
    assert chosen(results)["window"] == [16, 24]


def test_none_when_no_change_keeps_what_she_knows():
    assert chosen([UNCHANGED, {"window": [0, 8], "push": None, "development": [-5.0], "knowledge": [0.0, 4.0]}]) is None
