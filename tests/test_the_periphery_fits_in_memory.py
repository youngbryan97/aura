"""A campaign that took the machine into swap.

`read_periphery` returns a fresh dict of up to four hundred entries whose keys
it formats fresh on every call, so keeping one per frame keeps four hundred new
strings per frame with it. Measured on the offline organism: 284 KB a frame
against 2.5 KB for everything else the recording holds. At sixty rounds that is
4.3 GB; at three hundred it is 21.5 GB, and the campaign reached thirty-five
gigabytes resident, pushed 6.8 GB of swap and ran at three hundred seconds a
turn against 0.8 before.

The readings are numbers in a fixed set of slots, so they belong in an array.
These pin that the array is the one `periphery_matrix` would have built from
the same dicts, because the closure result has to be the same measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.closure import PeripheryAccumulator, periphery_matrix


def test_it_builds_what_the_dicts_would_have_built() -> None:
    rows = [{"a": 1.0}, {"a": 2.0, "b": 3.0}, {"b": 4.0, "c": 5.0}]
    acc = PeripheryAccumulator(chunk=2)
    for row in rows:
        acc.note(row)
    matrix, names = acc.matrix()
    expected, expected_names = periphery_matrix(rows)
    assert names == expected_names
    assert np.array_equal(matrix, expected)


def test_a_key_that_arrives_late_reads_zero_before_it_arrived() -> None:
    acc = PeripheryAccumulator()
    acc.note({"early": 1.0})
    acc.note({"early": 2.0, "late": 9.0})
    matrix, names = acc.matrix()
    assert names == ("early", "late")
    assert matrix[0, names.index("late")] == 0.0
    assert matrix[1, names.index("late")] == 9.0


def test_columns_come_out_sorted_so_the_array_is_comparable() -> None:
    acc = PeripheryAccumulator()
    acc.note({"z": 1.0, "a": 2.0, "m": 3.0})
    _, names = acc.matrix()
    assert list(names) == sorted(names)


def test_it_grows_past_its_first_chunk() -> None:
    acc = PeripheryAccumulator(chunk=64)
    for index in range(500):
        acc.note({"x": float(index)})
    matrix, names = acc.matrix()
    assert acc.rows() == 500
    assert matrix.shape == (500, 1)
    assert matrix[499, 0] == 499.0
    assert matrix[0, 0] == 0.0


def test_nothing_read_is_an_empty_matrix_rather_than_a_guess() -> None:
    acc = PeripheryAccumulator()
    matrix, names = acc.matrix()
    assert names == ()
    assert matrix.shape == (0, 0)


def test_a_value_that_is_not_a_number_is_skipped_not_guessed() -> None:
    acc = PeripheryAccumulator()
    acc.note({"good": 1.5, "bad": "text"})
    matrix, names = acc.matrix()
    assert matrix[0, names.index("good")] == pytest.approx(1.5)
    assert matrix[0, names.index("bad")] == 0.0


def test_a_row_does_not_inherit_the_previous_row() -> None:
    """The buffer is reused, so a slot a frame did not report must read zero."""
    acc = PeripheryAccumulator()
    acc.note({"a": 5.0, "b": 6.0})
    acc.note({"a": 7.0})
    matrix, names = acc.matrix()
    assert matrix[1, names.index("b")] == 0.0


def test_the_array_is_a_copy_the_caller_owns() -> None:
    acc = PeripheryAccumulator()
    acc.note({"a": 1.0})
    matrix, _ = acc.matrix()
    acc.note({"a": 2.0})
    assert matrix.shape[0] == 1


def test_both_runners_accumulate_rather_than_keep_the_dicts() -> None:
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    for name in ("run_subject_core.py", "run_subject_core_v25.py"):
        source = (repo / "tools" / name).read_text()
        assert "PeripheryAccumulator" in source, name
        assert "periphery.note(read_periphery" in source, name
        assert "periphery_rows" not in source, f"{name} still keeps the dicts"
