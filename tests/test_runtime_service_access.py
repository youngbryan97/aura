from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.autonomy.personhood_engine import PersonhoodEngine
from core.runtime import conversation_support, service_access
from core.state.aura_state import AuraState


def test_resolve_state_repository_prefers_orchestrator_alias(service_container):
    repo_from_orchestrator = object()
    repo_from_container = object()
    service_container.register_instance("state_repository", repo_from_container, required=False)

    orchestrator = SimpleNamespace(state_repo=repo_from_orchestrator)

    assert service_access.resolve_state_repository(orchestrator) is repo_from_orchestrator


def test_resolve_state_repository_falls_back_to_legacy_service_alias(service_container):
    repo = object()
    service_container.register_instance("state_repo", repo, required=False)

    assert service_access.resolve_state_repository() is repo


def test_resolve_llm_router_falls_back_to_kernel_interface(service_container):
    llm = object()
    kernel_interface = SimpleNamespace(
        kernel=SimpleNamespace(
            organs={
                "llm": SimpleNamespace(get_instance=lambda: llm),
            }
        )
    )
    service_container.register_instance("kernel_interface", kernel_interface, required=False)

    assert service_access.resolve_llm_router() is llm


def test_resolve_dialogue_cognition_uses_factory_when_service_missing(monkeypatch):
    import sys

    dialogue = object()
    fake_module = ModuleType("core.social.dialogue_cognition")
    fake_module.get_dialogue_cognition = lambda: dialogue
    monkeypatch.setitem(sys.modules, "core.social.dialogue_cognition", fake_module)

    assert service_access.resolve_dialogue_cognition() is dialogue


def test_build_conversational_context_blocks_aggregates_registered_services(service_container):
    state = AuraState.default()
    state.world.known_entities = {"Bryan": {"name": "Bryan"}}

    service_container.register_instance(
        "conversational_profiler",
        SimpleNamespace(get_context_injection=lambda user_id: f"profile:{user_id}"),
        required=False,
    )
    service_container.register_instance(
        "dialogue_cognition",
        SimpleNamespace(
            default_source_ids=lambda: ["recent"],
            get_context_injection=lambda user_id, current_text, source_ids: (
                f"dialogue:{user_id}:{current_text}:{source_ids[0]}"
            ),
        ),
        required=False,
    )
    service_container.register_instance(
        "humor_engine",
        SimpleNamespace(
            get_humor_guidance=lambda user_id: f"humor:{user_id}",
            get_banter_directive=lambda: "banter",
        ),
        required=False,
    )
    service_container.register_instance(
        "conversation_intelligence",
        SimpleNamespace(get_context_injection=lambda: "conversation"),
        required=False,
    )
    service_container.register_instance(
        "relational_intelligence",
        SimpleNamespace(get_context_injection=lambda user_id: f"relation:{user_id}"),
        required=False,
    )
    service_container.register_instance(
        "social_imagination",
        SimpleNamespace(
            get_context_injection=lambda user_id, current_text: f"social:{user_id}:{current_text}"
        ),
        required=False,
    )

    blocks = conversation_support.build_conversational_context_blocks(
        state,
        objective="steady continuity",
    )

    assert blocks == [
        "profile:bryan",
        "dialogue:bryan:steady continuity:recent",
        "humor:bryan",
        "banter",
        "conversation",
        "relation:bryan",
        "social:bryan:steady continuity",
    ]


def test_build_conversational_context_blocks_prioritizes_coding_and_task_context(monkeypatch, service_container):
    state = AuraState.default()
    state.world.known_entities = {"Bryan": {"name": "Bryan"}}

    service_container.register_instance(
        "conversational_profiler",
        SimpleNamespace(get_context_injection=lambda user_id: f"profile:{user_id}"),
        required=False,
    )

    monkeypatch.setattr(
        conversation_support,
        "build_coding_context_block",
        lambda objective: "## CODING WORKING SET\n- Files in play: core/runtime/conversation_support.py",
    )
    monkeypatch.setattr(
        "core.agency.task_commitment_verifier.get_task_commitment_verifier",
        lambda: SimpleNamespace(
            get_context_block=lambda objective="": "## TASK CONTINUITY\n- [task-1] Fix failing pytest — status: running_async"
        ),
    )

    blocks = conversation_support.build_conversational_context_blocks(
        state,
        objective="Keep going on the failing pytest in core/runtime/conversation_support.py",
    )

    assert blocks[0].startswith("## CODING WORKING SET")
    assert blocks[1].startswith("## TASK CONTINUITY")
    assert "profile:bryan" in blocks


def test_build_conversational_context_blocks_includes_goal_engine_state(monkeypatch, service_container):
    state = AuraState.default()
    state.world.known_entities = {"Bryan": {"name": "Bryan"}}

    service_container.register_instance(
        "goal_engine",
        SimpleNamespace(
            get_context_block=lambda objective="": "## GOAL EXECUTION STATE\n- Immediate execution: stabilize the runtime"
        ),
        required=False,
    )

    blocks = conversation_support.build_conversational_context_blocks(
        state,
        objective="Keep going on the project roadmap",
    )

    assert any(block.startswith("## GOAL EXECUTION STATE") for block in blocks)


@pytest.mark.asyncio
async def test_personhood_engine_uses_shared_router_resolution(monkeypatch):
    llm = SimpleNamespace(
        think=AsyncMock(return_value=SimpleNamespace(content="Shared route response."))
    )
    monkeypatch.setattr(
        "core.runtime.service_access.resolve_llm_router",
        lambda default=None: llm,
    )

    engine = PersonhoodEngine(SimpleNamespace())

    result = await engine._generate_thought(AuraState.default(), "Check continuity.")

    assert result == "Shared route response."
    llm.think.assert_awaited_once()
