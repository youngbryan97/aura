"""Perturbational complexity is read beside whether its response unfolds in time and differs by source.

Complexity over the whole response matrix can be bought two cheap ways: switch
every reached domain on and hold it for the whole horizon, or have every source
reach the same set. P34.3 asks that the response stay temporally structured and
P34.4 that it stay heterogeneous. These pin both readings on matrices shaped as
the report stores them.
"""

from __future__ import annotations

import pytest

from core.subject.pci import response_structure

pytestmark = pytest.mark.unit


def _row(reached: list[str], rows: list[str]) -> dict:
    return {"reached": reached, "rows": rows}


def test_a_response_that_rises_and_falls_and_differs_by_source_is_both() -> None:
    reading = response_structure({
        "P": _row(["A", "G"], ["0110", "0011", "0000"]),
        "A": _row(["G"], ["0000", "1100", "0000"]),
    })
    assert reading["temporally_structured"] is True
    assert reading["heterogeneous"] is True
    assert reading["distinct_reach_patterns"] == 2


def test_a_response_held_on_for_the_whole_horizon_is_not_structured_in_time() -> None:
    reading = response_structure({
        "P": _row(["A"], ["1111", "0000"]),
        "A": _row(["G"], ["0110", "0000"]),
    })
    assert reading["held_for_the_whole_horizon"] == ["P"]
    assert reading["temporally_structured"] is False


def test_every_source_reaching_the_same_set_is_not_heterogeneous() -> None:
    reading = response_structure({
        "P": _row(["A", "G"], ["0110", "0011"]),
        "S": _row(["G", "A"], ["0100", "0010"]),
    })
    assert reading["heterogeneous"] is False


def test_a_response_that_reached_nothing_is_neither() -> None:
    reading = response_structure({"P": _row([], ["0000"])})
    assert reading["sources_reaching"] == 0
    assert reading["temporally_structured"] is False
    assert reading["heterogeneous"] is False
