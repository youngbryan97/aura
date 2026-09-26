"""A sweep can be pointed at the lopsided cuts, and a look at some is never a sweep of all.

On seed 7 the cheapest bipartition by regression is recurrent cognition alone
against the other nine, and the cheapest cuts in practice are lopsided. The v5
partition line has never been read on her by intervention, and a full sweep is
511 cuts at up to 128 anchors each. The ten singleton cuts cost about a
fiftieth of that. `--screen` takes an even stride through the enumeration and
reaches few of them, so `only` names them by one side. Either way the sweep has
not scored every cut and says so.
"""

from __future__ import annotations

import pytest

from core.subject.state import DOMAINS
from core.subject.v25_cut import chosen_cuts
from core.subject.v25_runtime import bipartitions

pytestmark = pytest.mark.unit


def test_one_side_names_one_cut():
    cuts, screened = chosen_cuts(bipartitions(), only=("C",))
    assert screened
    assert len(cuts) == 1
    left, right = cuts[0]
    assert "C" in (("".join(left)), ("".join(right)))


def test_every_domain_alone_is_ten_cuts():
    cuts, screened = chosen_cuts(bipartitions(), only=tuple(DOMAINS))
    assert screened and len(cuts) == len(DOMAINS)
    assert all(min(len(left), len(right)) == 1 for left, right in cuts)


def test_a_side_is_found_whichever_way_the_cut_is_written():
    # The enumeration fixes the first domain on the left, so P alone is written
    # P|rest and C alone is written rest|C; both are found, in any letter order.
    cuts, _ = chosen_cuts(bipartitions(), only=("P", "CS", "SC"))
    sides = {"".join(sorted(left)) for left, _ in cuts} | {"".join(sorted(right)) for _, right in cuts}
    assert "P" in sides and "CS" in sides
    assert len(cuts) == 2


def test_a_side_no_cut_has_is_refused():
    with pytest.raises(ValueError):
        chosen_cuts(bipartitions(), only=("X",))


def test_without_a_choice_every_cut_is_scored_and_not_screened():
    cuts, screened = chosen_cuts(bipartitions())
    assert len(cuts) == 511 and not screened


def test_the_stride_screen_is_unchanged():
    everything = list(bipartitions())
    cuts, screened = chosen_cuts(everything, screen=24)
    assert screened and cuts == everything[:: len(everything) // 24][:24]
