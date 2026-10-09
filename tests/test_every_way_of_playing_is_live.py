"""Every way of playing she is said to have is done by a function her live play calls, and is shown by a test."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from core.agency.ways_of_playing import WAYS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("way", WAYS, ids=[way.name for way in WAYS])
def test_the_way_is_done_by_a_function_her_live_play_calls(way):
    module_name, function_name = way.done_by.split(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, function_name, None)), way.done_by
    caller = inspect.getsource(importlib.import_module(way.called_from))
    calls = [line for line in caller.splitlines() if function_name in line and "def " + function_name not in line]
    assert calls, f"{way.called_from} never calls {function_name}"


@pytest.mark.parametrize("way", WAYS, ids=[way.name for way in WAYS])
def test_the_way_is_shown_by_a_test(way):
    assert (ROOT / way.shown_by).is_file(), way.shown_by


def test_every_way_the_map_says_she_has_is_listed():
    table = (ROOT / "docs/design-docs/THE_56_GAMES_AND_WHAT_THEY_ASK.md").read_text()
    rows = [line.split("|") for line in table.splitlines() if line.startswith("| ") and "`" in line]
    said_to_have = {row[1].strip() for row in rows if "not yet" not in row[3]}
    listed = {way.name for way in WAYS}
    assert said_to_have == listed, (said_to_have - listed, listed - said_to_have)
