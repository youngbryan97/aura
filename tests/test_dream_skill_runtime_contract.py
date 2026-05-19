from pathlib import Path
from types import SimpleNamespace

import pytest


def test_dream_skill_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/skills/dream_skill.py")) == []


def _install_services(monkeypatch, services):
    import core.skills.dream_skill as dream_module

    def _get(name, default=None):
        return services.get(name, default)

    monkeypatch.setattr(dream_module.ServiceContainer, "get", staticmethod(_get))


def _install_task_tracker(monkeypatch, tracker):
    import core.utils.task_tracker as task_tracker

    monkeypatch.setattr(task_tracker, "get_task_tracker", lambda: tracker)


def _capture_world_state(monkeypatch):
    import core.world_state as world_state

    captured = SimpleNamespace(events=[])

    def _record_event(description, *, source, salience, ttl):
        captured.events.append(
            {
                "description": description,
                "source": source,
                "salience": salience,
                "ttl": ttl,
            }
        )

    monkeypatch.setattr(world_state, "get_world_state", lambda: SimpleNamespace(record_event=_record_event))
    return captured


class ClosingTracker:
    def __init__(self):
        self.tasks = []

    def create_task(self, coro, *, name):
        self.tasks.append(name)
        coro.close()


@pytest.mark.asyncio
async def test_dream_skill_returns_structured_success_receipts(monkeypatch):
    from core.skills.dream_skill import DreamSkill

    class DreamJournal:
        async def synthesize_dream(self):
            return {"dream_content": "A coherent synthesis emerged.", "seed_count": 3}

    class SemanticDefrag:
        async def run_defrag_cycle(self):
            return "defrag complete"

    class DreamCycle:
        async def process_dreams(self):
            return "dlq complete"

    class HeuristicSynthesizer:
        async def synthesize_from_telemetry(self):
            return {"insight": "prefer replayable logs"}

    class Drive:
        async def satisfy(self, name, amount):
            return {"name": name, "amount": amount}

    tracker = ClosingTracker()
    _install_task_tracker(monkeypatch, tracker)
    _install_services(
        monkeypatch,
        {
            "dream_journal": DreamJournal(),
            "orchestrator": SimpleNamespace(
                semantic_defrag=SemanticDefrag(),
                dream_cycle=DreamCycle(),
            ),
            "heuristic_synthesizer": HeuristicSynthesizer(),
            "drive_engine": Drive(),
        },
    )
    captured_world = _capture_world_state(monkeypatch)

    result = await DreamSkill().execute({}, {})

    assert result["ok"] is True
    assert result["subsystems"]["dream_journal"]["status"] == "completed"
    assert result["subsystems"]["semantic_defrag"]["status"] == "queued"
    assert result["subsystems"]["dlq_cycle"]["status"] == "queued"
    assert result["subsystems"]["heuristic_synthesis"]["status"] == "completed"
    assert result["subsystems"]["drive_restoration"]["status"] == "completed"
    assert tracker.tasks == ["dream_skill.semantic_defrag", "dream_skill.process_dreams"]
    assert captured_world.events


@pytest.mark.asyncio
async def test_dream_skill_closes_unscheduled_background_work_and_reports_degradation(monkeypatch):
    from core.skills.dream_skill import DreamSkill

    class BrokenTracker:
        def create_task(self, coro, *, name):
            reason = f"{name}:scheduler unavailable"
            raise RuntimeError(reason)

    class BrokenDreamJournal:
        async def synthesize_dream(self):
            reason = "dream journal unavailable"
            raise TimeoutError(reason)

    class SemanticDefrag:
        async def run_defrag_cycle(self):
            return "defrag complete"

    class DreamCycle:
        async def process_dreams(self):
            return "dlq complete"

    class BrokenHeuristicSynthesizer:
        async def synthesize_from_telemetry(self):
            reason = "telemetry synthesis unavailable"
            raise RuntimeError(reason)

    class BrokenDrive:
        async def satisfy(self, name, amount):
            reason = f"{name}:{amount}:drive unavailable"
            raise RuntimeError(reason)

    tracker = BrokenTracker()
    _install_task_tracker(monkeypatch, tracker)
    _install_services(
        monkeypatch,
        {
            "dream_journal": BrokenDreamJournal(),
            "orchestrator": SimpleNamespace(
                semantic_defrag=SemanticDefrag(),
                dream_cycle=DreamCycle(),
            ),
            "heuristic_synthesizer": BrokenHeuristicSynthesizer(),
            "drive_engine": BrokenDrive(),
        },
    )
    _capture_world_state(monkeypatch)

    result = await DreamSkill().execute({}, {})

    subsystems = result["subsystems"]
    assert result["ok"] is True
    assert subsystems["dream_journal"]["status"] == "failed"
    assert subsystems["semantic_defrag"]["status"] == "failed"
    assert subsystems["dlq_cycle"]["status"] == "failed"
    assert subsystems["heuristic_synthesis"]["status"] == "failed"
    assert subsystems["drive_restoration"]["status"] == "failed"
    assert set(result["degraded_subsystems"]) >= {
        "dream_journal",
        "semantic_defrag",
        "dlq_cycle",
        "heuristic_synthesis",
        "drive_restoration",
    }
