"""Her goal work can read how she is running: what her memory governor measured of her own process and her workers, and what
her health contract says of itself (F15).

LIVE 27 November 2026 her goal of steadying herself read her lane control's source for a leak over eleven passes and
stopped short of a change each time: "I have no live RSS or process probe", and she asked Bryan for a sample.
"""
from __future__ import annotations

import asyncio
from collections import deque

import pytest

from core.agency import her_goal_at_work as goals
from core.resilience.memory_governor import MemoryGovernor
from core.skills import her_draft

pytestmark = pytest.mark.unit


def _governor(samples, workers):
    governor = MemoryGovernor.__new__(MemoryGovernor)
    governor._trend_parts = deque(samples)
    governor._model_resource_state = "steady"
    governor.threshold_prune, governor.threshold_unload, governor.threshold_critical = 1.0, 2.0, 3.0
    governor._cleanup_events = []
    governor._workers_now = lambda: workers
    return governor


def test_what_her_governor_measured_is_hers_to_read():
    governor = _governor([(0.0, 1000.0, 8000.0), (600.0, 1100.0, 9500.0)], [(9500.0, 42, "Qwen3-1.7B-4bit")])
    measured = governor.what_it_measured()
    assert measured["samples"] == [(0.0, 1000.0, 8000.0), (600.0, 1100.0, 9500.0)]
    assert measured["workers_now"] == [(9500.0, 42, "Qwen3-1.7B-4bit")] and measured["state"] == "steady"


def test_her_tool_says_what_grew_and_where_from_her_own_runtime(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract

    governor = _governor([(0.0, 1000.0, 8000.0), (600.0, 1100.0, 9500.0)], [(9500.0, 42, "Qwen3-1.7B-4bit")])
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: governor if name == "memory_governor" else default))
    monkeypatch.setattr(health_contract, "runtime_health_report", lambda: {
        "status": "degraded", "required_probes": {"inference": {"ok": False, "reason": "no lane"}, "event_bus": {"ok": True}},
        "integrity": {"lockdep_splats": [{"a": "b"}], "taint": []}})
    got = asyncio.run(her_draft.HowIAmRunningSkill().execute({}))
    assert got["ok"]
    assert "her own process went from 1000 to 1100 MB and her workers from 8000 to 9500 MB" in got["summary"]
    assert "reads itself degraded" in got["summary"]
    lines = got["text"].splitlines()
    assert lines[0].startswith("probe inference failing") and lines[1].startswith("integrity lockdep_splats")
    assert "worker 42 holds 9500 MB: Qwen3-1.7B-4bit" in lines and not any("taint" in line for line in lines)


def test_without_her_governor_she_is_told_so(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: default))
    monkeypatch.setattr(health_contract, "runtime_health_report", lambda: {})
    got = asyncio.run(her_draft.HowIAmRunningSkill().execute({}))
    assert got["ok"] and got["summary"] == "her memory governor is not running"


def test_her_way_of_working_a_goal_reads_how_she_is_running_before_her_source():
    assert goals.HER_WAY.index("how_i_am_running") < goals.HER_WAY.index("find_in_source")
    assert "how_i_am_running" in {skill.name for skill in her_draft.HER_TOOLS}
    assert her_draft.HowIAmRunningSkill.effect_scope == "read_only"
