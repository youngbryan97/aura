"""Her domains cut apart are read with the power to tell her whole from her cut apart.

The 31 October reading is the case these pin: cut apart read as predicted, and
the intact life read nothing above its standard error either, which the three
rules of 2 November must read as not holding rather than as a pass.
"""

from __future__ import annotations

import json

import pytest

from tools.read_disconnected_powered import READING_FILE, REPORT_FILE, main, read

pytestmark = pytest.mark.unit


def _report(intact, cut, drift=(0.03, 0.01), as_predicted=True):
    life = lambda phi, se: {"phi_do": phi, "standard_error": se}  # noqa: E731
    return {
        "rounds": 120,
        "intact": life(*intact),
        "drift": life(*drift),
        "disconnected": life(*cut),
        "verdict": {"as_predicted": as_predicted, "why": "cut apart, her domains read no integration"},
    }


def test_the_reading_of_31_october_does_not_hold():
    reading = read(_report(intact=(0.010424, 0.018350), cut=(-0.005716, 0.014496), drift=(0.034651, 0.023289)))
    assert reading["as_before"]
    assert not reading["intact"]["integrated"]
    assert not reading["intact"]["differs_from_cut_apart"]
    assert not reading["holds"]


def test_a_whole_that_reads_integrated_and_apart_from_the_cut_holds():
    reading = read(_report(intact=(0.02, 0.004), cut=(0.0, 0.003)))
    assert reading["intact"]["integrated"]
    assert reading["intact"]["minus_cut_apart"] == pytest.approx(0.02)
    assert reading["intact"]["difference_standard_error"] == pytest.approx(0.005)
    assert reading["holds"]


def test_integrated_but_not_apart_from_the_cut_does_not_hold():
    reading = read(_report(intact=(0.012, 0.005), cut=(0.004, 0.005)))
    assert reading["intact"]["integrated"]
    assert not reading["intact"]["differs_from_cut_apart"]
    assert not reading["holds"]


def test_a_cut_that_reads_integrated_does_not_hold_whatever_the_whole_reads():
    reading = read(_report(intact=(0.05, 0.004), cut=(0.0, 0.003), as_predicted=False))
    assert not reading["holds"]
    assert "did not read as predicted" in reading["why"]


def test_the_drift_life_is_reported_and_decides_nothing():
    reading = read(_report(intact=(0.02, 0.004), cut=(0.0, 0.003), drift=(0.0, 0.01)))
    assert not reading["drift_beside"]["integrated"]
    assert reading["holds"]


def test_it_reads_the_tool_report_and_writes_beside_it(tmp_path):
    (tmp_path / REPORT_FILE).write_text(json.dumps(_report(intact=(0.02, 0.004), cut=(0.0, 0.003))))
    assert main([str(tmp_path)]) == 0
    assert json.loads((tmp_path / READING_FILE).read_text())["holds"]
