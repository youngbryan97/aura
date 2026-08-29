"""Binding "the largest" to what it names right now costs half her play.

The argument for binding it is good, which is why it was built. "The largest"
names a specific thing once she has looked at a state, and a multi-step search
that re-reads the word against every state it imagines will score a merge
elsewhere as losing the line.

Measured 2026-08-29, five games each run to a dead board, same seeds:

    her line + the world model      without binding      with binding
    median best tile                       1024                  512
    best seen                              2048                 1024
    total                                  2142                 1658
    moves                                  1069                  827

In a world that combines, the superlative is MEANT to float. A line about
keeping the largest in a corner is about whatever is largest as the game goes
on; bound to the 128 that was there when the line was formed, it names
something that no longer exists the moment she merges past it.

The function is kept because the argument survives for worlds where the
extreme thing does not change. This pins that it is not wired into the search,
so nobody reinstates it from the reasoning alone.
"""

from __future__ import annotations

import inspect

import pytest

from core.agency.how_good_is_this import bound_to
from core.perception.what_is_there import arranged


def board(*values):
    return arranged([(0.2 + n * 0.15, 0.2, v) for n, v in enumerate(values)])


# ── it still does what it says ───────────────────────────────────────────

def test_a_superlative_can_be_bound_to_what_it_names():
    assert bound_to("keep the largest in the corner", board("128", "64", "4")) == (
        "keep the 128 in the corner"
    )


def test_and_the_other_end_of_it_too():
    assert bound_to("keep the smallest out of the middle", board("128", "64", "4")) == (
        "keep the 4 out of the middle"
    )


@pytest.mark.parametrize(
    "line", ["press left often", "", "keep to the bottom row"]
)
def test_a_line_naming_no_superlative_is_untouched(line):
    assert bound_to(line, board("128", "64")) == line


def test_a_thing_with_no_numbers_in_it_binds_nothing():
    words = arranged([(0.2, 0.2, "Mon"), (0.35, 0.2, "Tue")])
    assert bound_to("keep the largest first", words) == "keep the largest first"


# ── and it is deliberately not in the search ─────────────────────────────

def test_the_search_does_not_bind_the_line():
    """Measured: binding halves the median best tile. See the module docstring."""
    import core.agency.looking_ahead as looking_ahead

    doing = [
        line
        for line in inspect.getsource(looking_ahead).splitlines()
        if not line.strip().startswith("#")
    ]
    assert not any("bound_to(" in line for line in doing)


def test_and_the_reason_is_written_down_where_it_would_be_reinstated():
    assert "float" in inspect.getdoc(bound_to)
    assert "1024" in inspect.getdoc(bound_to)
