import asyncio
from types import SimpleNamespace

import pytest

from core.kernel.aura_kernel import AuraKernel


def test_foreground_response_phases_get_extra_headroom():
    # The initial estimate remains stable; owned response completion can
    # outlive it without changing background or scientific deadlines.
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={})
    assert kernel._phase_timeout_seconds("UnitaryResponsePhase", priority=True) == 180.0
    assert kernel._phase_timeout_seconds("ResponseGenerationPhase", priority=True) == 180.0


def test_deep_handoff_response_phases_get_solver_headroom():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={"deep_handoff": True})
    assert kernel._phase_timeout_seconds("UnitaryResponsePhase", priority=True) == 210.0


def test_non_response_phase_timeouts_remain_stable():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={})
    assert kernel._phase_timeout_seconds("MemoryRetrievalPhase", priority=True) == 10.0
    assert kernel._phase_timeout_seconds("MemoryRetrievalPhase", priority=False) == 45.0


def test_background_response_phases_timeout_quickly_to_protect_foreground_turns():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={})
    assert kernel._phase_timeout_seconds("UnitaryResponsePhase", priority=False) == 12.0
    assert kernel._phase_timeout_seconds("ResponseGenerationPhase", priority=False) == 12.0


def test_background_only_phases_timeout_quickly_under_background_load():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={})
    assert kernel._phase_timeout_seconds("EternalMemoryPhase", priority=False) == 10.0
    assert kernel._phase_timeout_seconds("EternalGrowthEngine", priority=False) == 10.0


def test_priority_turns_keep_skill_phase_when_explicit_tool_intent_is_present():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={"intent_type": "SKILL"})

    assert kernel._should_skip_priority_phase("GodModeToolPhase", priority=True) is False


def test_priority_turns_skip_skill_phase_for_plain_chat():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={"intent_type": "CHAT"})

    assert kernel._should_skip_priority_phase("GodModeToolPhase", priority=True) is True


def test_priority_turns_skip_heavy_post_response_phases():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={"intent_type": "CHAT"})

    assert kernel._should_skip_priority_phase("PhiConsciousnessPhase", priority=True) is True
    assert kernel._should_skip_priority_phase("MemoryConsolidationPhase", priority=True) is True
    assert kernel._should_skip_priority_phase("LearningPhase", priority=True) is True


def test_owned_response_waits_for_endpoint_completion():
    from core.runtime import turn_outcome

    async def run():
        async def response():
            await asyncio.sleep(0.03)
            return "complete"

        with turn_outcome.bind_turn(turn_outcome.TurnOutcome(origin="user")):
            return await AuraKernel._await_phase_completion(
                asyncio.create_task(response()), phase_name="UnitaryResponsePhase",
                priority=True, origin="user", budget_s=0.001,
            )

    assert asyncio.run(run()) == "complete"


@pytest.mark.parametrize("phase,priority,origin", [
    ("MemoryRetrievalPhase", True, "user"),
    ("UnitaryResponsePhase", False, "user"),
    ("UnitaryResponsePhase", True, "benchmark"),
])
def test_other_phase_waits_still_cancel_and_drain(phase, priority, origin):
    stopped = []

    async def run():
        async def work():
            try:
                await asyncio.sleep(10)
            finally:
                stopped.append(True)

        with pytest.raises(TimeoutError):
            await AuraKernel._await_phase_completion(
                asyncio.create_task(work()), phase_name=phase,
                priority=priority, origin=origin, budget_s=0.001,
            )

    asyncio.run(run())
    assert stopped == [True]


def test_owned_response_caller_cancellation_drains_work():
    from core.runtime import turn_outcome

    async def run():
        stopped = asyncio.Event()

        async def work():
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()

        with turn_outcome.bind_turn(turn_outcome.TurnOutcome(origin="user")):
            waiting = asyncio.create_task(AuraKernel._await_phase_completion(
                asyncio.create_task(work()), phase_name="UnitaryResponsePhase",
                priority=True, origin="user", budget_s=0.001,
            ))
            await asyncio.sleep(0.03)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert stopped.is_set()

    asyncio.run(run())
