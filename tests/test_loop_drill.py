"""The fire drill of her repair loop raises its planted fault only when armed, and its function keeps its answer.

The first test is the guard her loop's mend of the drill must keep: a person's name still comes back as a title.
"""
from __future__ import annotations

import pytest

import core.self_modification.loop_drill as loop_drill

pytestmark = pytest.mark.unit


class _Person:
    name = "ada lovelace"


def test_a_name_comes_back_as_a_title():
    assert loop_drill.label(_Person()) == "Ada Lovelace"


def test_a_row_the_rows_reach_comes_back():
    assert loop_drill.nth([1, 2, 3], 1) == 2


def test_the_drill_runs_only_when_armed(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setattr(loop_drill, "armed_at", lambda: tmp_path / "loop_drill.armed")
    assert loop_drill.run_the_drill_if_armed() is False
    (tmp_path / "loop_drill.armed").write_text("")
    with caplog.at_level(logging.INFO, logger="Aura.SelfModification.Drill"):
        ran = loop_drill.run_the_drill_if_armed()
    assert ran is True or "no longer raises" in caplog.text
