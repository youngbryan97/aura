"""What her host monitor notices reaches the person only as her own turn on it (F06, canned answers out).

LIVE 15 November 2026, 16:11, in her conversation: "[Proactive/JARVIS] CPU usage is at 100% — something is running
hot. Want me to check what's consuming resources?" A sentence written in code, sent as hers, with a label in front.
"""
from __future__ import annotations

import pytest

from core.fictional.jarvis import ProactiveAnticipationEngine


class _Orch:
    def __init__(self):
        self.said: list[str] = []

    async def emit_spontaneous_message(self, text, **_k):
        self.said.append(text)
        return {"ok": True}


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(lambda *a, **k: None))
    orch = _Orch()
    made = ProactiveAnticipationEngine(orchestrator=orch)
    made._last_user_activity = 0.0
    made.MIN_INITIATION_INTERVAL_S = 0
    made._system_baseline = {"taken": 1.0}
    made.orch = orch
    return made


def _her_turn(monkeypatch, reply: str):
    asked: list[tuple[str, str, dict]] = []

    async def her_turn_on(heading, reading, **kwargs):
        asked.append((heading, reading, kwargs))
        return reply

    monkeypatch.setattr("core.brain.her_unprompted_turn.her_turn_on", her_turn_on)
    return asked


def _host_reads(engine, **state):
    engine._sample_host_blocking = lambda: state


@pytest.mark.asyncio
async def test_what_the_person_gets_is_her_turn_on_the_reading(engine, monkeypatch):
    asked = _her_turn(monkeypatch, "Memory's getting tight; the 27B is most of it.")
    _host_reads(engine, cpu_percent=None, memory_percent=88.0, memory_available_gb=7.6, disk_percent=72.0)

    await engine._check_system_anomalies()

    assert engine.orch.said == ["Memory's getting tight; the 27B is most of it."]
    heading, reading, how = asked[0]
    assert heading == "THE HOST, READ NOW" and how["origin"] == "jarvis"
    assert "memory in use: 88%" in reading and "memory available: 7.6 GB" in reading
    assert "processor" not in reading, "an unread figure is left out of the reading, not shown as a value"


@pytest.mark.asyncio
async def test_when_nothing_of_hers_forms_nothing_is_said_and_the_slot_is_kept(engine, monkeypatch):
    _her_turn(monkeypatch, "")
    _host_reads(engine, cpu_percent=97.0, memory_percent=40.0, memory_available_gb=30.0, disk_percent=72.0)

    await engine._check_system_anomalies()

    assert engine.orch.said == []
    assert engine._daily_initiation_count == 0


@pytest.mark.asyncio
async def test_an_unread_processor_figure_crosses_no_line(engine, monkeypatch):
    asked = _her_turn(monkeypatch, "anything")
    _host_reads(engine, cpu_percent=None, memory_percent=40.0, memory_available_gb=30.0, disk_percent=72.0)

    await engine._check_system_anomalies()

    assert asked == [] and engine.orch.said == []


@pytest.mark.asyncio
async def test_a_condition_that_cleared_is_news_when_it_returns(engine, monkeypatch):
    """The restatement guard promised to re-arm when a condition clears; nothing called it."""
    asked = _her_turn(monkeypatch, "The disk is nearly full again.")
    full = dict(cpu_percent=20.0, memory_percent=40.0, memory_available_gb=30.0, disk_percent=93.0)

    _host_reads(engine, **full)
    await engine._check_system_anomalies()
    await engine._check_system_anomalies()
    assert len(asked) == 1, "a standing condition is said once while it holds"

    _host_reads(engine, **{**full, "disk_percent": 70.0})
    await engine._check_system_anomalies()
    _host_reads(engine, **full)
    await engine._check_system_anomalies()
    assert len(asked) == 2
