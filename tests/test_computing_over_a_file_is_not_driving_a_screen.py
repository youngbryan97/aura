"""An average over a spreadsheet shows nothing on a screen.

LIVE, 2026-08-27: "Since West came out top on average approved deal size in
<path>/deals.csv, what's West doing that the other regions should copy?" was
routed to the desktop actuation lane and came back "OS automation refused to
act because the objective has no complete observable acceptance contract.
Completed 0/1 steps."

It never could have one. The actuation lane is for what a screen shows, and
this module already draws that line twice — a filesystem READ goes to the read
lane, and building software goes to the builder. Working something out from a
named file is the third case.

A turn that also asks for a change still belongs to the lane that can write:
"count the .py files in <dir> and write the number into <file>" needs both.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import (
    _asks_to_change_a_file,
    looks_like_desktop_objective,
)

_CSV = "/private/tmp/claude-501/scratchpad/deals.csv"


@pytest.mark.parametrize(
    "asked",
    [
        f"Since West came out top on average approved deal size in {_CSV}, "
        "what's West doing that the other regions should copy?",
        f"how many of those are approved in {_CSV}?",
        f"which region has the highest average approved amount_gbp in {_CSV}?",
    ],
)
def test_working_something_out_from_a_file_is_not_actuation(asked: str) -> None:
    assert looks_like_desktop_objective(asked) is False


@pytest.mark.parametrize(
    "asked",
    [
        "Count how many .py files are in /tmp and write the number into ~/Documents/n.txt",
        "write hello into ~/Documents/x.txt",
        "open Notes and write something",
        "delete everything in ~/Downloads",
        "copy ~/a.txt to ~/b.txt",
    ],
)
def test_a_turn_that_changes_something_still_goes_to_the_lane_that_can(asked: str) -> None:
    assert looks_like_desktop_objective(asked) is True


def test_a_mutation_verb_needs_something_to_mutate() -> None:
    """"...that the other regions should COPY" is not a file operation.

    Same shape as `\\boff\\b` matching inside "off-the-shelf": a word read out
    of prose that was never about a file. A change needs somewhere to land.
    """
    assert _asks_to_change_a_file("the other regions should copy that") is False
    assert _asks_to_change_a_file("write up what you found") is False
    assert _asks_to_change_a_file("copy ~/a.txt to ~/b.txt") is True
    assert _asks_to_change_a_file("write the number into ~/Documents/n.txt") is True
    assert _asks_to_change_a_file("delete that file") is True
