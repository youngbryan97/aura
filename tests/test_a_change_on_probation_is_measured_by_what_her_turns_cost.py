"""A change on probation that answers the same and costs more is taken back (Master Todo F16, to build).

Her turns are kept as she finishes them, with a reading of her memory, and the turns after a change landed are
compared with those before it in the same running: their length by the Mann-Whitney test, her memory by how fast it
grows, each one-sided at 0.05.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency import her_goal_at_work as work
from core.agency import her_turns_on_probation as turns

LANDED = 1_000_000.0 + 20 * 300


def _rows(path: Path, *, after_seconds: float, after_growth: float, running: str = "R") -> Path:
    shuffle = random.Random(7)
    rows = []
    for side, start, seconds, growth, base in (("before", 1_000_000.0, 60.0, 10.0, 9000.0),
                                               ("after", LANDED, after_seconds, after_growth, 9100.0)):
        for i in range(20):
            at = start + i * 300
            rows.append({"turn": f"{side}{i}", "at": at, "outside_model": shuffle.gauss(seconds, 8), "model": True,
                         "running": running})
            rows.append({"memory_mb": base + (i * 300 / 3600) * growth + shuffle.gauss(0, 3), "at": at,
                         "running": running})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_turns_costlier_since_it_landed_are_named_and_kept_short_of_an_order_of_magnitude(tmp_path):
    found = turns.costlier_since(LANDED, path=_rows(tmp_path / "t.jsonl", after_seconds=90, after_growth=10), running="R")
    assert found["said"].startswith("her turns spend 9") and "spent 5" in found["said"] and "memory" not in found["said"]
    assert not found["beyond_doubt"], "half as much again may be what she was asked"


def test_an_order_of_magnitude_more_is_beyond_doubt(tmp_path):
    found = turns.costlier_since(LANDED, path=_rows(tmp_path / "t.jsonl", after_seconds=700, after_growth=10), running="R")
    assert found["beyond_doubt"]


def test_one_long_turn_after_many_short_ones_is_said_and_not_acted_on(tmp_path):
    path = _rows(tmp_path / "t.jsonl", after_seconds=90, after_growth=10)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row.get("at", 0) < LANDED or row.get("turn") == "after0"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    found = turns.costlier_since(LANDED, path=path, running="R")
    assert not found["beyond_doubt"], found


def test_memory_growing_faster_since_it_landed_is_named_and_its_level_alone_is_not(tmp_path):
    faster = turns.costlier_since(LANDED, path=_rows(tmp_path / "a.jsonl", after_seconds=60, after_growth=200),
                                  running="R")["said"]
    assert faster.startswith("her memory grows 19") or faster.startswith("her memory grows 20"), faster
    # Higher after the landing, as memory is over any running, and growing as fast as before: nothing to say.
    assert turns.costlier_since(LANDED, path=_rows(tmp_path / "b.jsonl", after_seconds=60, after_growth=10),
                                running="R")["said"] == ""


def test_nothing_is_said_of_another_running(tmp_path):
    path = _rows(tmp_path / "t.jsonl", after_seconds=90, after_growth=200)
    assert turns.costlier_since(LANDED, path=path, running="another") == {"said": "", "beyond_doubt": False}


def test_a_turn_that_did_not_call_her_model_says_nothing_of_her(tmp_path):
    path = _rows(tmp_path / "t.jsonl", after_seconds=90, after_growth=10)
    rows = [{**json.loads(line), "model": False} if "outside_model" in line else json.loads(line)
            for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert turns.costlier_since(LANDED, path=path, running="R")["said"] == ""


def test_her_finished_turns_are_kept_once_with_a_memory_reading(tmp_path):
    path = tmp_path / "her_turns" / "turns.jsonl"
    receipts = [{"turn_id": "t1", "finished_at": 10.0, "duration_s": 42.0, "model_generation": True,
                 "latency_s": {"prefill": 10.0, "decode": 30.0, "tool": 1.0}},
                {"turn_id": "t2", "finished_at": None, "duration_s": None, "model_generation": True}]
    assert turns.keep_her_recent_turns(path=path, receipts=receipts) == 1
    assert turns.keep_her_recent_turns(path=path, receipts=receipts) == 0, "a turn is kept once"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["turn"], row["outside_model"]) for row in rows if "turn" in row] == [("t1", 2.0)]
    assert sum("memory_mb" in row for row in rows) == 2


def test_a_receipt_says_when_its_turn_finished():
    from core.verify.turn_receipt import TurnReceipt

    receipt = TurnReceipt(turn_id="t", started_at=1.0)
    receipt.finished_at = 3.5
    assert receipt.as_dict()["finished_at"] == 3.5 and receipt.as_dict()["duration_s"] == 2.5


def test_probation_keeps_a_cost_short_of_an_order_of_magnitude_on_her_record(monkeypatch):
    standing = SimpleNamespace(slug="mend-x", goal="Mend x", offer={"merged_at": LANDED, "on_probation": True})
    monkeypatch.setattr(work, "change_on_probation", lambda: standing)
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: [])
    monkeypatch.setattr(work, "_tried_on_her_real_calls", lambda record: {"disposition": "not_observed"})
    monkeypatch.setattr(turns, "costlier_since", lambda landed: {"said": "her turns spend 9.0 s", "beyond_doubt": False})
    written: list = []
    monkeypatch.setattr(work, "_write_now", lambda record: written.append(dict(record.offer)))
    monkeypatch.setattr(work, "take_it_back", lambda slug, why: pytest.fail("not taken back short of beyond doubt"))
    assert work.settle_probation() == []
    assert written[-1]["cost_since_landing"] == "her turns spend 9.0 s"


def test_probation_takes_back_a_change_that_made_her_costlier(monkeypatch):
    standing = SimpleNamespace(slug="mend-x", goal="Mend x", offer={"merged_at": LANDED, "on_probation": True})
    monkeypatch.setattr(work, "change_on_probation", lambda: standing)
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: [])
    monkeypatch.setattr(work, "_tried_on_her_real_calls", lambda record: {"disposition": "not_observed"})
    monkeypatch.setattr(turns, "costlier_since",
                        lambda landed: {"said": "her turns spend 91 s outside her model", "beyond_doubt": True})
    taken: list = []
    monkeypatch.setattr(work, "take_it_back", lambda slug, why: (taken.append(why), {"ok": True})[1])
    assert work.settle_probation() == [{"slug": "mend-x", "status": "taken back", "why": ""}]
    assert taken == ["since it landed her turns spend 91 s outside her model"]


@pytest.mark.parametrize("module", ["core/agency/her_turns_on_probation.py",
                                    "core/self_modification/what_her_change_must_keep.py"])
def test_what_judges_her_changes_is_hers_to_leave_alone(module):
    assert work.governs_her(module)
