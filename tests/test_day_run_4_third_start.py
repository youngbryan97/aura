"""What the third start of day run 4 (9 November 2026) found, each measured where it happens.

The adaptive immune system asked autopoiesis for reduce_load/global through request_repair 188 times in 37 minutes; no
handler exists, and each ask was a recorded failure that her resilience took as a new loss. Her goal tools were held
by an open journal entry under the executive's temporal rule, eight times in an hour. The strange loop's weight save
was refused 21 times for writing outside a governed scope.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np

from core.cognitive import autopoiesis as auto
from core.executive.executive_core import ActionType, ExecutiveCore, Intent, IntentSource


def test_a_repair_with_no_handler_is_not_attempted_and_is_recorded_once(monkeypatch):
    recorded, asked_the_will = [], []
    monkeypatch.setattr(auto, "_record_autopoiesis_degradation", lambda exc, **kw: recorded.append(str(exc)))
    engine = auto.AutopoiesisEngine()
    monkeypatch.setattr(engine, "_request_governance_approval", lambda *a: asked_the_will.append(a) or True)

    async def ask_thrice():
        return [await engine.request_repair("global", auto.RepairStrategy.REDUCE_LOAD) for _ in range(3)]

    results = asyncio.run(ask_thrice())
    assert [r.success for r in results] == [False, False, False]
    assert recorded == ["No repair handler registered for reduce_load/global"]
    assert asked_the_will == []


def test_a_repair_with_a_handler_is_still_put_to_the_will(monkeypatch):
    asked_the_will = []
    engine = auto.AutopoiesisEngine()
    engine.register_repair_handler(auto.RepairStrategy.CLEAR_CACHE, "memory", lambda: True)
    monkeypatch.setattr(engine, "_request_governance_approval", lambda *a: asked_the_will.append(a) or False)
    result = asyncio.run(engine.request_repair("memory", auto.RepairStrategy.CLEAR_CACHE))
    assert asked_the_will and result.governance_approved is False


def test_her_goal_tools_are_not_held_by_an_open_obligation_and_others_still_are():
    def held(tool):
        intent = Intent(source=IntentSource.AUTONOMOUS, action_type=ActionType.TOOL_CALL,
                        payload={"tool_name": tool, "args": {}})
        return ExecutiveCore._temporal_safe_autonomous_tool_constraints(intent)

    for tool in ("read_source", "find_in_source", "edit_draft", "check_draft", "offer_draft"):
        assert held(tool) == {"timeout_s": 600}
    assert held("file_operation") is None


def test_the_strange_loop_saves_its_weights_inside_a_governed_scope(tmp_path, monkeypatch):
    import core.cognitive.strange_loop as strange
    import core.governance_context as governance

    entered = []
    real = governance.local_internal_governed_scope

    def watched(name, **kwargs):
        entered.append(name)
        return real(name, **kwargs)

    monkeypatch.setattr(governance, "local_internal_governed_scope", watched)
    monkeypatch.setattr(strange, "_PERSIST_DIR", tmp_path)
    loop = SimpleNamespace(_tick_count=7, _models={}, _current_state=np.zeros(2), _previous_state=np.zeros(2))
    strange.StrangeLoop._save_weights(loop)
    assert entered == ["strange_loop.persist_weights"]
    assert json.loads((tmp_path / "rls_weights.json").read_text())["tick_count"] == 7
