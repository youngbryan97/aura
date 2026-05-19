from pathlib import Path
from types import SimpleNamespace

import pytest


def test_sleep_skill_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/skills/sleep.py")) == []


def _install_services(monkeypatch, services):
    import core.skills.sleep as sleep_module

    def _get(name, default=None):
        return services.get(name, default)

    monkeypatch.setattr(sleep_module.ServiceContainer, "get", staticmethod(_get))


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


@pytest.mark.asyncio
async def test_sleep_uses_conversation_and_heuristic_consolidation_when_primary_paths_fail(monkeypatch):
    from core.skills.sleep import SleepSkill

    class BrokenMemory:
        async def recall(self, query, *, limit):
            reason = f"{query}:{limit}:offline"
            raise RuntimeError(reason)

    class Journal:
        def __init__(self):
            self.entries = []

        async def record(self, *, content, dream_type):
            self.entries.append({"content": content, "dream_type": dream_type})

    class Drive:
        def __init__(self):
            self.restored = []

        async def satisfy(self, name, amount):
            self.restored.append((name, amount))

    journal = Journal()
    drive = Drive()
    _install_services(
        monkeypatch,
        {
            "memory_facade": BrokenMemory(),
            "orchestrator": SimpleNamespace(
                conversation_history=[
                    {"role": "user", "content": "We learned the installer needs rollback receipts."},
                    {"role": "assistant", "content": "I will preserve recovery telemetry."},
                ]
            ),
            "dream_journal": journal,
            "drive_engine": drive,
        },
    )
    captured_world = _capture_world_state(monkeypatch)

    result = await SleepSkill().execute()

    assert result["ok"] is True
    assert result["phases"]["memory_recall"]["status"] == "failed"
    assert result["phases"]["conversation_fallback"]["status"] == "completed"
    assert result["phases"]["dream_synthesis"]["status"] == "skipped"
    assert result["phases"]["heuristic_synthesis"]["status"] == "completed"
    assert "installer needs rollback receipts" in result["knowledge"]
    assert journal.entries and journal.entries[0]["dream_type"] == "consolidation"
    assert ("energy", 30.0) in drive.restored
    assert captured_world.events


@pytest.mark.asyncio
async def test_sleep_reports_partial_failures_and_preserves_recovery_output(monkeypatch):
    from core.skills.sleep import SleepSkill

    class Memory:
        async def recall(self, query, *, limit):
            return [
                "User wants hostile evaluation with replayable logs.",
                "System should prevent regressions before release.",
            ]

    class BrokenBrain:
        async def think(self, prompt, *, mode):
            reason = f"{mode}:reflective path down"
            raise TimeoutError(reason)

    class BrokenJournal:
        async def record(self, *, content, dream_type):
            reason = f"{dream_type}:journal unavailable"
            raise RuntimeError(reason)

    class BrokenIdentity:
        async def evolve_from_dream(self, knowledge):
            reason = f"{len(knowledge)}:identity lane unavailable"
            raise RuntimeError(reason)

    class BrokenCompressor:
        async def compact(self):
            reason = "compaction queue locked"
            raise RuntimeError(reason)

    class PartialDrive:
        async def satisfy(self, name, amount):
            if name == "competence":
                reason = f"{name}:{amount}:drive write unavailable"
                raise RuntimeError(reason)
            return True

    _install_services(
        monkeypatch,
        {
            "memory_facade": Memory(),
            "cognitive_engine": BrokenBrain(),
            "dream_journal": BrokenJournal(),
            "canonical_self_engine": BrokenIdentity(),
            "knowledge_compression": BrokenCompressor(),
            "drive_engine": PartialDrive(),
        },
    )
    _capture_world_state(monkeypatch)

    result = await SleepSkill().execute()

    assert result["ok"] is True
    assert result["phases"]["memory_recall"]["status"] == "completed"
    assert result["phases"]["heuristic_synthesis"]["status"] == "completed"
    assert result["phases"]["dream_journal"]["status"] == "failed"
    assert result["phases"]["identity_evolution"]["status"] == "failed"
    assert result["phases"]["memory_compaction"]["status"] == "failed"
    assert result["phases"]["drive_restoration"]["status"] == "degraded"
    assert result["phases"]["drive_restoration"]["restored"] == ["energy"]
    assert result["phases"]["drive_restoration"]["failed"] == ["competence"]
    assert "hostile evaluation" in result["knowledge"]
    assert set(result["degraded_steps"]) >= {
        "dream_synthesis",
        "dream_journal",
        "identity_evolution",
        "memory_compaction",
        "drive_restoration",
    }
