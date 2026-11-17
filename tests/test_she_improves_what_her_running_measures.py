"""A change of hers for a cost her own running measures, kept only when the same measurement says it helped (Master
Todo F16, part 3).

LIVE 15 to 17 November 2026: the line in her resource governor that asks for a full collection asked for 482 of them,
a median of 232 ms each and 127 s in all, freeing a median of 586 objects.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from core.agency import her_goal_at_work as goals
from core.agency import improving_by_measurement as improving
from core.runtime import collector_pauses

_KEEPER = "import gc\n\n\ndef tidy():\n    return gc.collect()\n"
_SITE = "planted_gc/keeper.py:5"


def _git(*args, cwd):
    return subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


def _measured(collections, median_ms, collected=586, generations=(2,)):
    return {"collections": collections, "total_s": collections * median_ms / 1000.0, "median_s": median_ms / 1000.0,
            "median_collected": collected, "generations": list(generations)}


def test_the_full_collection_on_the_line_is_asked_as_a_young_one():
    source = "import gc\n\ndef a():\n    gc.collect()\n\ndef b():\n    gc.collect(2)\n"
    assert improving.young_collection_in_place_of(source, 4) == source.replace("    gc.collect()\n",
                                                                              "    gc.collect(1)\n")
    assert improving.young_collection_in_place_of(source, 7) is None, "a generation asked for by name is left alone"
    assert improving.young_collection_in_place_of(source, 3) is None


def test_the_costliest_line_is_the_one_whose_full_collections_held_her_longest():
    sites = {"a.py:1": _measured(482, 232), "b.py:2": _measured(210, 184),
             "c.py:3": _measured(900, 20),  # under her loop's budget at the median
             "d.py:4": _measured(999, 300, generations=(1,))}  # young already
    cost = improving.the_costliest_asked_for_collection(0.0, 3600.0, budget_s=0.05, sites=sites)
    assert cost.site == "a.py:1" and cost.path == "a.py" and cost.line == 1
    assert "482 full collections asked for at a.py:1, a median of 232 ms" in cost.in_words()
    assert improving.the_costliest_asked_for_collection(0.0, 3600.0, budget_s=0.05, sites={"c.py:3": sites["c.py:3"]}) \
        is None


@pytest.mark.parametrize("after,kept", [
    (_measured(10, 4, collected=500, generations=(1,)), True),  # held less, asked as often
    (_measured(40, 4, collected=500, generations=(1,)), False),  # held less, asked four times as often
    (_measured(10, 300), False),  # held more
])
def test_it_is_kept_only_when_each_collection_holds_her_less_and_the_line_asks_no_more_often(after, kept):
    verdict, said = improving.judged(_measured(10, 232), after, before_s=3600.0, after_s=3600.0)
    assert verdict is kept
    assert "at the median against 232 ms before" in said


@pytest.fixture
def her_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "planted_gc").mkdir(parents=True)
    (source / "planted_gc" / "__init__.py").write_text("")
    (source / "planted_gc" / "keeper.py").write_text(_KEEPER)
    _git("init", "-q", "-b", "main", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "planted", cwd=source)
    monkeypatch.setattr(goals, "source_root", lambda: source)
    monkeypatch.setattr(goals, "workspaces_dir", lambda: tmp_path / "workspaces")
    monkeypatch.setattr(goals, "records_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(goals, "MEASURING_HOLD", tmp_path / "no-hold")
    monkeypatch.setattr(goals, "the_gates_her_yes_asks", lambda record: {"stand-in": {"passed": True, "tail": ""}})
    monkeypatch.syspath_prepend(str(source))
    for name in ("planted_gc", "planted_gc.keeper"):
        sys.modules.pop(name, None)
    yield source
    from core.self_modification.into_the_running_process import undo_every_load_for_test

    undo_every_load_for_test()


@pytest.mark.parametrize("after,kept", [(_measured(10, 4, collected=500, generations=(1,)), True),
                                        (_measured(30, 4, collected=500, generations=(1,)), False)])
def test_her_change_is_made_loaded_and_then_judged_by_the_same_measurement(her_source, monkeypatch, after, kept):
    import planted_gc.keeper as keeper

    before = _measured(10, 232)
    # Before the change, from her start to 1000 s; after it, the line asks for its collections across a window as
    # long as that one, and none in its first seconds.
    monkeypatch.setattr(collector_pauses, "asked_for_collections",
                        lambda since, until: {_SITE: before} if until <= 1000.0 else
                        {_SITE: after} if until - since >= 1000.0 else {})
    monkeypatch.setattr("core.runtime.lockdep.LOOP_BLOCKING_HOLD_S", 0.05)

    made = asyncio.run(improving.improve_by_measurement(since=0.0, now=1000.0))
    assert made and made[0]["status"] == "merged", made
    assert made[0]["loaded"]["functions"] == ["tidy"]
    assert "gc.collect(1)" in (her_source / "planted_gc" / "keeper.py").read_text()
    assert keeper.tidy.__code__.co_consts != compile(_KEEPER, "k", "exec").co_consts, "the running process runs it"
    slug = made[0]["slug"]
    loaded_at = goals.load(slug).offer["loaded"]["at"]

    assert asyncio.run(improving.improve_by_measurement(since=0.0, now=loaded_at + 10.0)) == [], \
        "not judged before as much has run after it as before"
    judged = asyncio.run(improving.improve_by_measurement(since=0.0, now=loaded_at + 1000.0))
    record = goals.load(slug)
    assert record.offer["judged"]["kept"] is kept, judged
    if kept:
        assert judged[0]["status"] == "kept" and "gc.collect(1)" in (her_source / "planted_gc" / "keeper.py").read_text()
    else:
        assert judged[0]["status"] == "taken back"
        assert "gc.collect()" in (her_source / "planted_gc" / "keeper.py").read_text()
        assert record.status == goals.STALLED and record.offer.get("reverted")
        again = asyncio.run(improving.improve_by_measurement(since=0.0, now=loaded_at + 2000.0))
        assert not again or again[0]["why"] == "measured no better when it was made before", again
