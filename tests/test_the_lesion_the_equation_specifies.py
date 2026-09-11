"""Cutting the channels across a partition is not freezing one side of it.

The equation says to remove `E(A*, B*)` and `E(B*, A*)` and leave each side's
internal dynamics alone. Holding one side still does not do that: it severs
every edge out of those domains and destroys their own dynamics as well, so it
reads as a much larger intervention than the one written down — and a deficit
under it shows that something depended on those domains existing, not that
anything crossed the cut.

Each side is now run once with the other held at the cut instant, so it evolves
with nothing crossing and its own pipeline intact, and the two recordings are
composed column-wise. Both halves start from one snapshot and the experiment
clock makes them the same length of life.

The node clamp stays, reported as what it is.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.subject.clamp import compose
from core.subject.state import DOMAINS, CoreState


def _row(fill: float, t: float) -> CoreState:
    return CoreState(
        values={key: np.full(3, fill) for key in DOMAINS},
        t=t,
        condition="probe",
        tag="after",
    )


def test_each_side_comes_from_the_run_in_which_it_was_free():
    left = ["A", "G"]
    composed = compose([_row(1.0, 0.0)], [_row(2.0, 0.0)], left)
    assert len(composed) == 1
    row = composed[0]
    for key in DOMAINS:
        expected = 1.0 if key in left else 2.0
        assert row.domain(key) == pytest.approx(np.full(3, expected)), key


def test_the_composition_keeps_what_a_frame_is():
    composed = compose([_row(1.0, 5.0)], [_row(2.0, 5.0)], ["A"])
    assert composed[0].t == 5.0
    assert composed[0].condition == "probe"
    assert composed[0].tag == "after"


def test_a_miss_on_either_side_survives_the_composition():
    a = _row(1.0, 0.0)
    b = _row(2.0, 0.0)
    a.misses["one"] = "left could not read it"
    b.misses["two"] = "right could not read it"
    row = compose([a], [b], ["A"])[0]
    assert row.misses == {"one": "left could not read it", "two": "right could not read it"}


def test_an_uneven_pair_composes_to_the_shorter_one():
    """Two runs of the same length is the contract; a short one is not silent."""
    composed = compose([_row(1.0, 0.0), _row(1.0, 1.0)], [_row(2.0, 0.0)], ["A"])
    assert len(composed) == 1


def test_the_run_reports_the_node_clamp_separately():
    """It is a useful ablation and it is not the partition lesion."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "tools" / "run_subject_core.py").read_text()
    assert '"node_clamp"' in source
    assert '"severed"' in source
