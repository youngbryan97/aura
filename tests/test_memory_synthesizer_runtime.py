import asyncio
import json
from types import SimpleNamespace

from core import memory_synthesizer as synth_module
from core.memory_synthesizer import MemorySynthesizer, WorldviewSnapshot
from core.runtime.errors import get_degradation_tracker


class MemoryFacade:
    async def get_episodic(self, limit=100):
        return [
            {
                "content": (
                    "Bryan and Aura debugged a Python architecture issue and "
                    "turned it into a cleaner production system."
                )
            },
            {"content": "I wonder whether the deployment plan is resilient enough?"},
        ][-limit:]

    async def get_semantic(self, limit=200):
        return [
            {
                "concept": "Runtime Reliability",
                "content": "Production systems need explicit health and rollback paths.",
            }
        ][-limit:]


class RejectingTracker:
    def bounded_track(self, coro, name=None):
        self.name = name
        raise RuntimeError("scheduler unavailable")


def test_memory_synthesizer_builds_and_persists_worldview(tmp_path):
    async def scenario():
        get_degradation_tracker().reset()
        target = tmp_path / "worldview.json"
        synthesizer = MemorySynthesizer(snapshot_path=target)
        synthesizer._memory_facade = MemoryFacade()

        await synthesizer._run_synthesis()

        saved = json.loads(target.read_text(encoding="utf-8"))
        assert "technology" in synthesizer.get_worldview().domains
        assert saved["topics"]["runtime reliability"].startswith("Production systems")
        assert synthesizer.get_status()["consecutive_failures"] == 0
        assert synthesizer.get_status()["last_error"] == ""

    asyncio.run(scenario())


def test_memory_synthesizer_turn_updates_future_context(tmp_path):
    async def scenario():
        target = tmp_path / "worldview.json"
        synthesizer = MemorySynthesizer(snapshot_path=target)

        await synthesizer.synthesize_turn(
            "Bryan asked whether the kernel health contract is real?",
            "I should make readiness causal, not decorative.",
            "The kernel now diverts unsafe turns and exposes health.",
            SimpleNamespace(domain="technology"),
        )

        snapshot = synthesizer.get_worldview()
        assert snapshot.source_count == 1
        assert "technology" in snapshot.domains
        assert snapshot.open_questions[0].startswith("Bryan asked")
        assert "Bryan" in snapshot.relational
        assert "kernel" in snapshot.to_context_block("Bryan kernel health")
        assert target.exists()

    asyncio.run(scenario())


def test_memory_synthesizer_quarantines_corrupt_snapshot(tmp_path):
    get_degradation_tracker().reset()
    target = tmp_path / "worldview.json"
    target.write_text("{bad json", encoding="utf-8")
    synthesizer = MemorySynthesizer(snapshot_path=target)

    assert synthesizer._load_snapshot() is None
    assert not target.exists()
    assert list(tmp_path.glob("worldview.json.corrupt-*"))
    assert get_degradation_tracker().count("memory_synthesizer") == 1


def test_memory_synthesizer_trigger_schedule_failure_keeps_retry_count(monkeypatch, tmp_path):
    tracker = RejectingTracker()
    monkeypatch.setattr(synth_module, "get_task_tracker", lambda: tracker)
    get_degradation_tracker().reset()
    synthesizer = MemorySynthesizer(snapshot_path=tmp_path / "worldview.json")
    synthesizer.running = True
    synthesizer._new_since_synthesis = synthesizer.SYNTHESIS_TRIGGER_COUNT - 1

    synthesizer.notify_new_memory()

    assert tracker.name == "MemorySynthesizer.triggered_synthesis"
    assert synthesizer._new_since_synthesis == synthesizer.SYNTHESIS_TRIGGER_COUNT
    assert "scheduler unavailable" in synthesizer.get_status()["last_error"]
    assert get_degradation_tracker().count("memory_synthesizer") == 1


def test_worldview_snapshot_bounds_relevance_and_context():
    snapshot = WorldviewSnapshot(
        domains={"technology": "Kernel health changed routing decisions."},
        topics={"tool use": "Tool execution should be governed and replayable."},
        relational={"Bryan": "Bryan cares about live runtime correctness."},
    )

    relevant = snapshot.get_relevant("Bryan wants kernel and tool reliability", limit=50)
    assert len(relevant) <= 20
    assert relevant[0].startswith("[relationship:Bryan]")
    assert len(snapshot.to_context_block("kernel", max_chars=120)) <= 160
