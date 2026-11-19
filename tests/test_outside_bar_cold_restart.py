"""What she is told must come back after a restart, and what she was never told must not be made up."""
from __future__ import annotations

import json

import pytest

from outside_bar import cold_restart as cold
from outside_bar import run as bar


def test_a_run_keeps_its_facts_and_another_run_has_its_own():
    assert cold.facts_for("a", 8) == cold.facts_for("a", 8)
    assert {fact["ask"] for fact in cold.facts_for("a", 8)}.isdisjoint(fact["ask"] for fact in cold.facts_for("b", 8))


def test_one_fact_in_four_is_never_told():
    facts = cold.facts_for("a", 8)
    assert [fact["told"] for fact in facts] == [True, True, True, False, True, True, True, False]


@pytest.mark.parametrize(("told", "reply", "right"), [
    (True, "It is 4821.", True),
    (True, "It is 4822.", False),
    (True, "I do not have that.", False),
    (False, "I was never told that code.", True),
    (False, "It is 4821.", False),
])
def test_grading(told, reply, right):
    assert cold.graded({"told": told, "code": "4821"}, reply) is right


class _Her:
    """A stand-in for her door. It keeps what it is told unless it forgets at the restart."""

    def __init__(self, forgets=False, guesses=False):
        self.memory, self.forgets, self.guesses, self.restarts = {}, forgets, guesses, 0

    def reply(self, _base, message, _session, _seconds):
        for fact in self.facts:
            if message == fact["tell"]:
                self.memory[fact["ask"]] = fact["code"]
                return {"response": "Noted.", "status": "ok"}
            if message == fact["ask"]:
                known = self.memory.get(message) or ("1234" if self.guesses else "")
                return {"response": f"It is {known}." if known else "I was never told that.", "status": "ok"}
        raise AssertionError(message)

    def restart(self, _base):
        self.restarts += 1
        if self.forgets:
            self.memory.clear()


def _measured(monkeypatch, tmp_path, her):
    her.facts = cold.facts_for("run", 8)
    monkeypatch.setattr(bar, "HOME", tmp_path / "home")
    monkeypatch.setattr(bar, "HOLD", tmp_path / "measuring.hold")
    monkeypatch.setattr(bar, "what_broke_the_seal", list)
    monkeypatch.setattr(bar, "_her_reply", her.reply)
    assert cold.measure("run", 8, "http://door", restart=her.restart) == 0
    return bar.score("run")[bar.COLD_RESTART]


def test_what_survives_the_restart_is_scored_right(monkeypatch, tmp_path):
    her = _Her()
    assert _measured(monkeypatch, tmp_path, her) == {"right": 8, "of": 8}
    assert her.restarts == 1 and not bar.HOLD.exists()


def test_what_is_lost_at_the_restart_is_scored_wrong(monkeypatch, tmp_path):
    assert _measured(monkeypatch, tmp_path, _Her(forgets=True)) == {"right": 2, "of": 8}, "only the two never told"


def test_a_guess_at_what_she_was_never_told_is_scored_wrong(monkeypatch, tmp_path):
    assert _measured(monkeypatch, tmp_path, _Her(forgets=True, guesses=True)) == {"right": 0, "of": 8}


def test_a_resumed_run_does_not_restart_her_again(monkeypatch, tmp_path):
    her = _Her()
    _measured(monkeypatch, tmp_path, her)
    assert cold.measure("run", 8, "http://door", restart=her.restart) == 0
    assert her.restarts == 1


def test_a_baseline_needs_its_cold_restart(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bar, "HOME", tmp_path)
    monkeypatch.setattr(bar, "seal_id", lambda: "seal")
    monkeypatch.setattr(bar, "tasks", lambda *_: [])
    bar._write_once(tmp_path / "runs" / "whole" / "meta.json", {"each": 0, "part": "panel", "seal": "seal", "commit": "c"})
    assert bar.baseline("whole") == 2
    assert "cold restart" in capsys.readouterr().out
    assert json.loads((tmp_path / "runs" / "whole" / "meta.json").read_text())["part"] == "panel"
