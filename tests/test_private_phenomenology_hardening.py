import asyncio
import json
import time

import pytest

from core.agency.private_phenomenology import PrivatePhenomenology
from core.container import ServiceContainer


@pytest.fixture(autouse=True)
def clear_services():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


class QuietOrchestrator:
    def __init__(self):
        now = time.time()
        self.start_time = now - 1000.0
        self._last_user_interaction_time = now - 1000.0
        self._suppress_unsolicited_proactivity_until = 0.0
        self._foreground_user_quiet_until = 0.0
        self.is_busy = False


def _register_quiet_runtime(engine=None):
    ServiceContainer.register_instance("orchestrator", QuietOrchestrator(), required=False)
    if engine is not None:
        ServiceContainer.register_instance("cognitive_engine", engine, required=False)


def _entry(timestamp, reflection, arousal):
    return {
        "timestamp": timestamp,
        "reflection": reflection,
        "pad_state": {"P": 0.0, "A": arousal, "D": 0.0},
    }


@pytest.mark.asyncio
async def test_prune_keeps_recent_and_high_arousal_entries(tmp_path):
    storage = tmp_path / "monologue.jsonl"
    rows = [
        _entry(1.0, "old low", 0.1),
        _entry(2.0, "old high", 0.95),
        _entry(3.0, "recent low", 0.2),
    ]
    storage.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n{bad json}\n",
        encoding="utf-8",
    )
    manager = PrivatePhenomenology(
        storage_path=str(storage),
        max_storage_bytes=20,
        keep_recent=1,
        high_arousal_threshold=0.7,
    )

    pruned = await manager._prune_if_needed()

    assert pruned is True
    kept = [json.loads(line) for line in storage.read_text(encoding="utf-8").splitlines()]
    assert [row["reflection"] for row in kept] == ["old high", "recent low"]


@pytest.mark.asyncio
async def test_llm_timeout_records_local_reflection(monkeypatch, tmp_path):
    class HangingEngine:
        async def think(self, **_kwargs):
            await asyncio.sleep(10)

    _register_quiet_runtime(HangingEngine())
    monkeypatch.setenv("AURA_PHENOMENOLOGY_USE_LLM", "1")
    manager = PrivatePhenomenology(
        storage_path=str(tmp_path / "monologue.jsonl"),
        reflect_timeout_s=0.01,
    )

    reflection = await manager.reflect({"P": -0.5, "A": 0.2, "D": 0.1}, [{"event": "stall"}])

    assert "friction" in reflection
    assert "stall" in reflection
    assert "friction" in await manager.get_subjective_bias()


@pytest.mark.asyncio
async def test_reflect_defers_when_background_policy_fails(monkeypatch, tmp_path):
    _register_quiet_runtime()

    import core.runtime.background_policy as background_policy

    def broken_policy(*_args, **_kwargs):
        message = "policy unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(background_policy, "background_activity_reason", broken_policy)
    manager = PrivatePhenomenology(storage_path=str(tmp_path / "monologue.jsonl"))

    reflection = await manager.reflect({"P": 0.1, "A": 0.1, "D": 0.1}, [{"event": "x"}])

    assert reflection is None
    assert not (tmp_path / "monologue.jsonl").exists()
