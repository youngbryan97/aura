"""A pass at a goal of hers can read what has gone wrong in her own running, from her runtime's own record (F16).

LIVE 21 November 2026: her goal "Find something productive to work on" had made 2,022 calls over 290 passes and no
edit. Her tools could read her source and search it, and she searched it for TODO and FIXME, which her source does not
use. In the same process, every save of her causal world model was being refused, some sixty times that afternoon, and
none of her tools could see it. what_went_wrong reads the faults her runtime recorded: each kind once, how often, where
in her source it was raised, and what was done about it.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.a_call_that_failed import FailedCall
from core.runtime.errors import DegradationRecord, DegradationTracker
from core.skills import her_draft

pytestmark = pytest.mark.unit


def _record(subsystem, severity, error_type, message, *, at, call=None, raised_in=None, action="kept going"):
    return DegradationRecord(subsystem, severity, error_type, message, action, at, failed_call=call, raised_in=raised_in)


@pytest.fixture
def her_record(monkeypatch):
    tracker = DegradationTracker()
    monkeypatch.setattr("core.runtime.errors.get_degradation_tracker", lambda: tracker)
    return tracker


def _read(**params):
    return asyncio.run(her_draft.WhatWentWrongSkill().execute(params, {}))


def test_each_kind_of_fault_is_said_once_the_most_frequent_first(her_record):
    save = FailedCall("core.governance_context", "require_governance", "core/governance_context.py", 548,
                      "core.governance_context.GovernanceViolationError", "called outside governed context")
    for at in range(3):
        her_record.record(_record("causal_world_model", "degraded", "GovernanceViolationError",
                                  "file_write_gateway.write_text:causal_world_model.save called outside governed context",
                                  at=float(at), call=save, action="reported a non-durable causal-graph update"))
    her_record.record(_record("memory.cold_store", "warning", "KeyError", "'owner'", at=5.0,
                              raised_in=("core.memory.cold_store", "ColdStore.recall")))
    her_record.record(_record("chat", "info", "TimeoutError", "a wait under load", at=6.0))
    her_record.record(_record("chat", "debug", "ValueError", "not a failure", at=7.0))
    read = _read()
    assert read["ok"] and read["kinds"] == 2, read
    first, second = read["text"].splitlines()
    assert first.startswith("3x degraded causal_world_model: GovernanceViolationError at "
                            "core/governance_context.py:548 in require_governance: ")
    assert first.endswith("(then: reported a non-durable causal-graph update)")
    assert second.startswith("1x warning memory.cold_store: KeyError at core.memory.cold_store.ColdStore.recall: 'owner'")
    assert read["summary"] == "2 kind(s) of fault recorded, 4 in all"


def test_one_subsystem_is_read_by_the_start_of_its_name(her_record):
    her_record.record(_record("memory.cold_store", "warning", "KeyError", "'owner'", at=1.0))
    her_record.record(_record("chat", "warning", "ValueError", "bad", at=2.0))
    read = _read(subsystem="memory")
    assert read["kinds"] == 1 and "memory.cold_store" in read["text"] and "chat" not in read["text"]


def test_a_quiet_running_says_so(her_record):
    assert _read()["summary"] == "0 kind(s) of fault recorded, 0 in all"


def test_it_is_offered_to_her_goal_pass_and_reads_only():
    from core.executive.her_own_passes import OWN_GOAL_TOOLS

    assert "what_went_wrong" in her_draft.tool_definitions()
    assert her_draft.WhatWentWrongSkill.effect_scope == "read_only"
    assert "what_went_wrong" in OWN_GOAL_TOOLS
