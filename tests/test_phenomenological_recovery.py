import pytest

from core.consciousness.phenomenological_experiencer import (
    AttentionSchema,
    ExperientialContinuityEngine,
    PhenomenalSelfModel,
    PhenomenologicalExperiencer,
)
from core.runtime.errors import get_degradation_tracker


@pytest.fixture(autouse=True)
def _reset_degradation_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_deep_narrative_failure_updates_recovery_state(monkeypatch):
    class _BrokenRouter:
        async def think(self, **_kwargs):
            raise RuntimeError("narrative backend offline")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: _BrokenRouter() if name == "llm_router" else default,
    )

    psm = PhenomenalSelfModel()
    report = await psm.run_deep_narrative_update(
        continuity=ExperientialContinuityEngine(),
        schema=AttentionSchema("a hard problem", "aware", "cognitive", 0.7),
        qualia=[],
        current_emotion="curious",
        dominant_motivation="needs_to_reason",
    )

    assert report == psm._present_description
    assert psm.to_dict()["narrative_failure_streak"] == 1
    last = get_degradation_tracker().recent(subsystem="phenomenological_experiencer")[-1]
    assert last.action == "retained previous present-description after narrative update failed"


@pytest.mark.asyncio
async def test_witness_failure_updates_recovery_state(monkeypatch):
    class _BrokenRouter:
        async def think(self, **_kwargs):
            raise RuntimeError("witness backend offline")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: _BrokenRouter() if name == "llm_router" else default,
    )

    psm = PhenomenalSelfModel()
    observation = await psm.run_witness_reflection(
        continuity=ExperientialContinuityEngine(),
        credit_summary="credit assignment active",
    )

    assert observation == ""
    assert psm.to_dict()["witness_failure_streak"] == 1
    last = get_degradation_tracker().recent(subsystem="phenomenological_experiencer")[-1]
    assert last.action == "retained previous witness observation after reflection update failed"


@pytest.mark.asyncio
async def test_experiencer_update_loop_uses_adaptive_backoff(monkeypatch, tmp_path):
    sleep_delays: list[float] = []
    experiencer = PhenomenologicalExperiencer(save_dir=str(tmp_path))

    async def _broken_narrative():
        raise RuntimeError("phenomenal narrative task failed")

    async def _stop_after_sleep(delay):
        if delay == 0:
            return
        sleep_delays.append(delay)
        experiencer._running = False

    experiencer._running = True
    experiencer._current_schema = AttentionSchema("the task", "aware", "cognitive", 0.8)
    experiencer._run_deep_narrative = _broken_narrative
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.asyncio.sleep",
        _stop_after_sleep,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.BOOT_GRACE_PERIOD_S",
        0,
    )

    await experiencer._update_loop()

    assert sleep_delays == [5.0]
    status = experiencer.get_status()
    assert status["update_failure_streak"] == 1
    assert "RuntimeError" in status["last_update_error"]
    last = get_degradation_tracker().recent(subsystem="phenomenological_experiencer")[-1]
    assert last.action == "kept phenomenological update loop alive with adaptive backoff"
    assert last.severity == "degraded"
