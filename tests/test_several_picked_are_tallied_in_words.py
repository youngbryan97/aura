"""Several things picked and done in turn end with a tally in words, not the machinery's step count.

LIVE 2026-10-08 three games' account closed with "That much is verified. The
rest did not complete: step 3 not ok (2/3 steps)."
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_objective import _how_many_as_asked


@pytest.mark.unit
def test_none_one_and_all_are_said_as_words():
    assert _how_many_as_asked([{"had": True, "completed": False}] * 3) == "Of the three, none ended the way you asked."
    assert _how_many_as_asked([{"had": True, "completed": True}] * 3) == "All three ended the way you asked."
    assert _how_many_as_asked([{"had": True, "completed": True}, {"had": True}, {"had": False}]) == (
        "Of the three, one ended the way you asked, and one could not be had anywhere.")


@pytest.mark.unit
def test_one_thing_is_not_tallied():
    assert _how_many_as_asked([{"had": True, "completed": False}]) == ""
    assert _how_many_as_asked(None) == ""
