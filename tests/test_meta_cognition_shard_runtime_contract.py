from types import SimpleNamespace

import pytest


def test_meta_cognition_shard_degradation_audit_is_clean():
    from pathlib import Path

    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/orchestrator/meta_cognition_shard.py")) == []


@pytest.mark.asyncio
async def test_meta_cognition_fallback_correction_becomes_runtime_state():
    from core.orchestrator.meta_cognition_shard import MetaCognitionShard

    orchestrator = SimpleNamespace(
        conversation_history=[
            {"role": "assistant", "content": "alpha"},
            {"role": "user", "content": "beta"},
            {"role": "assistant", "content": "alpha"},
            {"role": "user", "content": "beta"},
        ],
        status=SimpleNamespace(is_processing=False, healthy=True, health_metrics={}),
    )
    shard = MetaCognitionShard(orchestrator)

    report = await shard.perform_audit()

    assert report["corrections"] == ["repetition_break"]
    assert orchestrator.meta_cognitive_corrections
    assert "REPETITION_BREAK" in orchestrator.meta_cognitive_corrections[-1]
    assert orchestrator.status.health_metrics["meta_cognitive_corrections"] == orchestrator.meta_cognitive_corrections[-10:]


@pytest.mark.asyncio
async def test_meta_cognition_reentrant_audit_skips_without_double_work():
    from core.orchestrator.meta_cognition_shard import MetaCognitionShard

    shard = MetaCognitionShard(SimpleNamespace(status=SimpleNamespace(is_processing=False, healthy=True)))
    await shard._audit_lock.acquire()
    try:
        report = await shard.perform_audit()
    finally:
        shard._audit_lock.release()

    assert report["status"] == "skipped"
    assert report["reason"] == "audit_already_running"


@pytest.mark.asyncio
async def test_meta_cognition_correction_failure_returns_false():
    from core.orchestrator.meta_cognition_shard import MetaCognitionShard

    class BrokenOrchestrator:
        def add_correction_shard(self, _hint):
            error = RuntimeError("correction bus unavailable")
            raise error

    shard = MetaCognitionShard(BrokenOrchestrator())

    pushed = await shard._push_correction("latency_mitigation", "slow")

    assert pushed is False
    assert shard.get_status()["recent_corrections"] == []
