"""Internal bookkeeping writes must succeed under the live governance runtime.

Live incident (July 2026): with governance active, every cognitive-trace save
and every learning-example append was refused as a GOVERNANCE VIOLATION —
spawning incidents, inflating resilience frustration, and silently dropping
the write. Internal maintenance writers must establish their own
local_internal_governed_scope because they are invoked from arbitrary
contexts, including bare threads with no inherited scope.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time

import pytest

from core.container import ServiceContainer


@contextlib.contextmanager
def _governance_runtime_forced_active(monkeypatch):
    monkeypatch.delenv("AURA_GOVERNANCE_MODE", raising=False)
    monkeypatch.delenv("AURA_REQUIRE_GOVERNANCE", raising=False)
    saved_services = dict(ServiceContainer._services)
    saved_aliases = dict(ServiceContainer._aliases)
    saved_locked = ServiceContainer._registration_locked
    try:
        ServiceContainer._services = {}
        ServiceContainer._aliases = {}
        ServiceContainer._registration_locked = True
        yield
    finally:
        ServiceContainer._services = saved_services
        ServiceContainer._aliases = saved_aliases
        ServiceContainer._registration_locked = saved_locked


def test_cognitive_trace_save_is_governed(monkeypatch, tmp_path):
    from core.governance_context import governance_runtime_active
    from core.meta.cognitive_trace import CognitiveTrace

    with _governance_runtime_forced_active(monkeypatch):
        assert governance_runtime_active() is True
        trace = CognitiveTrace(trace_id="governed-test")
        trace.log_dir = str(tmp_path)
        trace.record_step("reason", "governed write check")
        trace.save()

        saved = tmp_path / "trace_governed-test.json"
        assert saved.exists(), "trace save must not be refused under live governance"
        payload = json.loads(saved.read_text())
        assert payload["steps"][0]["content"] == "governed write check"


def test_learning_pipeline_record_is_governed_from_background_thread(monkeypatch, tmp_path):
    from core.governance_context import governance_runtime_active
    from core.learning.genuine_learning_pipeline import ExperienceBuffer

    with _governance_runtime_forced_active(monkeypatch):
        assert governance_runtime_active() is True
        pipeline = ExperienceBuffer(db_path=str(tmp_path / "examples.jsonl"))

        # The append runs on a fresh daemon thread; intercept thread start to
        # run it synchronously so the assertion window is deterministic while
        # still exercising the no-inherited-scope path.
        started: list[threading.Thread] = []
        real_thread = threading.Thread

        class ImmediateThread(real_thread):
            def start(self):
                started.append(self)
                self.run()

        monkeypatch.setattr(threading, "Thread", ImmediateThread)

        accepted = pipeline.record(
            system_prompt="sys",
            user_input="hello",
            response="a substantive reply that satisfies quality scoring here",
            quality_score=0.95,
        )

        assert accepted is True
        assert started, "background persist thread never started"
        assert pipeline.db_path.exists(), (
            "learning example append must not be refused under live governance"
        )
        line = pipeline.db_path.read_text().strip()
        record = json.loads(line)
        assert record["_meta"]["quality"] == pytest.approx(0.95)


def test_research_history_owns_a_narrow_governed_append(monkeypatch, tmp_path):
    from core.autonomy.research_history import HISTORY_SCHEMA, ResearchHistory
    from core.governance_context import GovernanceViolation, get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    target = tmp_path / "research" / "cycle_history.jsonl"
    observed = []
    gateway = get_file_write_gateway()
    real_append = gateway.append_text

    def observing_append(path, text, **kwargs):
        observed.append(get_active_governance())
        return real_append(path, text, **kwargs)

    monkeypatch.setattr(gateway, "append_text", observing_append)

    with _governance_runtime_forced_active(monkeypatch):
        with pytest.raises(GovernanceViolation):
            real_append(target, "unguarded\n", source="test.unguarded")

        digest = ResearchHistory(target).append({"record_id": "research-1"})

    assert target.exists()
    assert len(observed) == 1
    token = observed[0]
    assert token is not None
    assert token.domain == "memory_write"
    assert token.source == "autonomy.research_cycle.history"
    constraints = dict(token.constraints)
    assert constraints["artifact"] == HISTORY_SCHEMA
    assert constraints["operation"] == "append_only"
    assert constraints["record_sha256"] == digest


@pytest.mark.asyncio
async def test_experience_consolidation_owns_async_governed_writes(
    monkeypatch,
    tmp_path,
):
    from core.consciousness import experience_consolidator as module
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    narrative_path = tmp_path / "identity" / "self_narrative.json"
    log_path = tmp_path / "identity" / "consolidation_log.jsonl"
    monkeypatch.setattr(module, "NARRATIVE_PATH", narrative_path)
    monkeypatch.setattr(module, "CONSOL_LOG_PATH", log_path)

    consolidator = module.ExperienceConsolidator(cognitive_engine=None)
    narrative = module.IdentityNarrative(
        version=4,
        signature_phrase="I retain measured changes in a durable self-model.",
    )
    consolidator._narrative = narrative

    gateway = get_file_write_gateway()
    real_write = gateway.write_text_async
    real_append = gateway.append_text_async
    observed = []

    async def observing_write(path, text, **kwargs):
        observed.append(("write", get_active_governance()))
        await real_write(path, text, **kwargs)

    async def observing_append(path, text, **kwargs):
        observed.append(("append", get_active_governance()))
        await real_append(path, text, **kwargs)

    monkeypatch.setattr(gateway, "write_text_async", observing_write)
    monkeypatch.setattr(gateway, "append_text_async", observing_append)

    with _governance_runtime_forced_active(monkeypatch):
        await consolidator._save_narrative()
        await consolidator._log_consolidation(
            narrative,
            {"experiences": [{"type": "test"}]},
        )

    assert narrative_path.exists()
    assert log_path.exists()
    assert [kind for kind, _token in observed] == ["write", "append"]
    save_token = observed[0][1]
    log_token = observed[1][1]
    assert save_token is not None
    assert save_token.source == "experience_consolidator.save_narrative"
    assert save_token.domain == "state_mutation"
    assert dict(save_token.constraints)["version"] == 4
    assert log_token is not None
    assert log_token.source == "experience_consolidator.log_consolidation"
    assert log_token.domain == "memory_write"
    assert dict(log_token.constraints)["operation"] == "append_only"


def test_degradation_is_not_recorded_for_governed_bookkeeping(monkeypatch, tmp_path):
    """The live failure mode: refused writes spawned incidents every turn."""
    from core.meta.cognitive_trace import CognitiveTrace

    degradations: list[str] = []
    monkeypatch.setattr(
        "core.meta.cognitive_trace.record_degradation",
        lambda subsystem, exc, **kw: degradations.append(f"{subsystem}: {exc}"),
    )

    with _governance_runtime_forced_active(monkeypatch):
        trace = CognitiveTrace(trace_id=f"clean-{int(time.time())}")
        trace.log_dir = str(tmp_path)
        trace.save()

    assert degradations == [], f"governed save still degraded: {degradations}"


def test_mhaf_shutdown_checkpoint_establishes_local_state_authority(monkeypatch, tmp_path):
    from core.consciousness import mhaf_field
    from core.governance_context import get_active_governance

    observed = []

    class Gateway:
        @staticmethod
        def write_text(_path, _payload, **_kwargs):
            observed.append(get_active_governance())

    monkeypatch.setattr(mhaf_field, "_DATA_PATH", tmp_path / "mhaf.json")
    monkeypatch.setattr(
        "core.runtime.file_write_gateway.get_file_write_gateway",
        lambda: Gateway(),
    )
    field = mhaf_field.MycelialHypergraphAttractorField()

    with _governance_runtime_forced_active(monkeypatch):
        field._save()

    assert len(observed) == 1
    assert observed[0] is not None
    assert observed[0].domain == "state_mutation"
    assert observed[0].source == "mhaf_field.persistence"


def test_epistemic_humility_checkpoint_establishes_local_state_authority(
    monkeypatch,
    tmp_path,
):
    from core.adaptation import epistemic_humility as humility_module
    from core.governance_context import get_active_governance

    observed = []

    class Gateway:
        @staticmethod
        def write_text(_path, _payload, **_kwargs):
            observed.append(get_active_governance())

    humility = humility_module.EpistemicHumility(orchestrator=None)
    humility.data_path = tmp_path / "humility.json"
    monkeypatch.setattr(humility_module, "get_file_write_gateway", lambda: Gateway())

    with _governance_runtime_forced_active(monkeypatch):
        humility._save()

    assert len(observed) == 1
    assert observed[0] is not None
    assert observed[0].domain == "state_mutation"
    assert observed[0].source == "epistemic_humility.persistence"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_factory", "running_attr"),
    [
        (
            lambda: __import__(
                "core.consciousness.mhaf_field",
                fromlist=["MycelialHypergraphAttractorField"],
            ).MycelialHypergraphAttractorField(),
            "_running",
        ),
        (
            lambda: __import__(
                "core.adaptation.epistemic_humility",
                fromlist=["EpistemicHumility"],
            ).EpistemicHumility(orchestrator=None),
            "running",
        ),
    ],
)
async def test_internal_state_owner_quiesces_loop_before_shutdown_checkpoint(
    monkeypatch,
    owner_factory,
    running_attr,
):
    events: list[str] = []

    async def active_loop() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            events.append("loop_stopped")

    owner = owner_factory()
    setattr(owner, running_attr, True)
    owner._task = asyncio.create_task(active_loop())
    monkeypatch.setattr(owner, "_save", lambda: events.append("checkpoint_saved"))
    await asyncio.sleep(0)

    await owner.stop()

    assert events == ["loop_stopped", "checkpoint_saved"]
