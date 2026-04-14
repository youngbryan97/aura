from types import SimpleNamespace

from core.kernel.aura_kernel import AuraKernel


def test_foreground_response_phases_get_extra_headroom():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={})
    assert kernel._phase_timeout_seconds("UnitaryResponsePhase", priority=True) == 120.0
    assert kernel._phase_timeout_seconds("ResponseGenerationPhase", priority=True) == 120.0


def test_deep_handoff_response_phases_get_solver_headroom():
    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = SimpleNamespace(response_modifiers={"deep_handoff": True})
    assert kernel._phase_timeout_seconds("UnitaryResponsePhase", priority=True) == 180.0


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
