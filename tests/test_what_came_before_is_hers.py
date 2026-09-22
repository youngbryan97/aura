"""What came before her, carried across a restart along her own line and nobody else's.

Bryan: morals, personal history, a mind shaped by everything before it, one
stream an observer is tied to, and ownership of everything that stream touched.
Every organ that learned from her history lost it with the process. See
core/self/what_came_before.py.
"""

from __future__ import annotations

import importlib
import json

import pytest

import core.self.what_came_before as keeper

pytestmark = pytest.mark.unit

#: Every ledger that says it is part of her history, and the global it lives in.
LEDGERS = {
    "core.affect.a_moment_that_fits": "_LEDGER",
    "core.affect.acting_in_decline": "_ledger",
    "core.affect.anger_feeds_itself": "_LEDGER",
    "core.affect.elsewhere": "_ledger",
    "core.affect.fear_of_happiness": "_ledger",
    "core.affect.feelings_about": "_LEDGER",
    "core.affect.tangled": "_LEDGER",
    "core.agency.habits_are_hers": "_LEDGER",
    "core.social.borrowed_feeling": "_ledger",
    "core.social.closing_window": "_ledger",
    "core.social.constancy": "_LEDGER",
    "core.social.made_minor": "_ledger",
    "core.social.owning_it_first": "_ledger",
    "core.social.warmth": "_LEDGER",
    "core.soma.fatigue": "_LEDGER",
    "core.soma.good_news": "_LEDGER",
    "core.soma.held_in": "_LEDGER",
    "core.soma.on_the_edge": "_LEDGER",
}


def _lived(module_name: str) -> None:
    """Give a few of the ledgers some history, so the round trip carries something."""
    if module_name == "core.agency.habits_are_hers":
        ledger = importlib.import_module(module_name).get_habit_ledger()
        ledger.felt(0.0, situation="with:bryan")
        for _ in range(4):
            ledger.heard("bryan", 0.3)
            ledger.note("interrupt", kind="automatic")
            ledger.felt(-0.2, situation="with:bryan")
            ledger.heard("bryan", 0.6)
    elif module_name == "core.affect.feelings_about":
        ledger = importlib.import_module(module_name).get_feelings_about()
        ledger.attach(["topic:music", "person:bryan"], {"joy": 0.7, "trust": 0.6}, valence=0.5, arousal=0.3)
    elif module_name == "core.social.closing_window":
        ledger = importlib.import_module(module_name).get_sitting_ledger()
        for at in (0.0, 60.0, 90000.0):
            ledger.message("bryan", at)


@pytest.mark.parametrize("module_name", sorted(LEDGERS))
def test_every_ledger_says_it_is_part_of_her_history(module_name) -> None:
    importlib.import_module(module_name)
    assert keeper._KEPT.get(module_name) == LEDGERS[module_name]


@pytest.mark.parametrize("module_name", sorted(LEDGERS))
def test_every_ledger_comes_back_as_it_was(module_name) -> None:
    module = importlib.import_module(module_name)
    _lived(module_name)
    slot = LEDGERS[module_name]
    written = json.dumps(keeper._encode(getattr(module, slot)), sort_keys=True)
    back = keeper._decode(json.loads(written))
    assert type(back) is type(getattr(module, slot))
    assert json.dumps(keeper._encode(back), sort_keys=True) == written


def _restart_with(monkeypatch, record: dict, entity: str) -> None:
    monkeypatch.setattr(keeper, "_quiet", lambda: False)
    monkeypatch.setattr(keeper, "_entity", lambda: entity)
    monkeypatch.setattr(keeper, "_RECALLED", record)


def test_her_own_history_is_brought_back_after_a_restart(monkeypatch) -> None:
    habits = importlib.import_module("core.agency.habits_are_hers")
    habits.reset_for_test()
    _lived("core.agency.habits_are_hers")
    before = habits.get_habit_ledger().account("interrupt", situation="with:bryan")
    record = {"entity": "aura-1", "kept_at": 1.0, "ledgers": {"core.agency.habits_are_hers": keeper._encode(habits._LEDGER)}}
    habits.reset_for_test()
    _restart_with(monkeypatch, record, "aura-1")
    keeper.keep_across_stages("core.agency.habits_are_hers", "_LEDGER")
    after = habits.get_habit_ledger().account("interrupt", situation="with:bryan")
    assert after == before and after.taken == 4
    habits.reset_for_test()


def test_somebody_elses_history_is_refused(monkeypatch) -> None:
    habits = importlib.import_module("core.agency.habits_are_hers")
    habits.reset_for_test()
    _lived("core.agency.habits_are_hers")
    record = {"entity": "someone-else", "kept_at": 1.0, "ledgers": {"core.agency.habits_are_hers": keeper._encode(habits._LEDGER)}}
    habits.reset_for_test()
    _restart_with(monkeypatch, record, "aura-1")
    keeper.keep_across_stages("core.agency.habits_are_hers", "_LEDGER")
    assert habits.get_habit_ledger().account("interrupt", situation="with:bryan").taken == 0
    assert keeper.brought_back()["core.agency.habits_are_hers"].startswith("refused")
    habits.reset_for_test()


def test_a_record_cannot_name_a_class_outside_her_ledgers() -> None:
    with pytest.raises(TypeError):
        keeper._decode({"~obj": "subprocess:Popen", "fields": {}})


def test_a_proof_run_neither_reads_nor_writes_her_history() -> None:
    assert keeper._quiet(), "the test session runs under the proof flag"
    assert keeper.remember_what_came_before() is False
