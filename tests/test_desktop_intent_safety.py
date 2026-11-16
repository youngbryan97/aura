from __future__ import annotations

from types import SimpleNamespace

import pytest
import interface.routes.chat_memory_state as _chat_memory_state
from tests.chat_lane_support import patch_chat_lane

INVENTORY_PROMPT = (
    "What tools can you hypothetically use externally on my computer? "
    "Explain the categories, name one realistic multi-step scenario, "
    "and do not actually open apps or execute tools yet."
)


@pytest.fixture(autouse=True)
def _reset_chat_route_state(monkeypatch):
    from core.memory.session_pin_cipher import SessionPinCipher
    from interface.routes import chat as chat_routes

    cipher = SessionPinCipher(b"k" * 32)
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_cipher", lambda: cipher)
    chat_routes._conversation_log.clear()
    chat_routes._session_memory_pins.clear()
    yield
    chat_routes._conversation_log.clear()
    chat_routes._session_memory_pins.clear()


def test_capability_inventory_prompt_is_not_desktop_execution() -> None:
    from core.phases.action_intent import detect_action_intent
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective
    from core.runtime.skill_task_bridge import (
        looks_like_capability_inventory_dialogue_request,
        looks_like_multi_step_skill_request,
    )
    from core.runtime.turn_analysis import analyze_turn

    assert looks_like_capability_inventory_dialogue_request(INVENTORY_PROMPT) is True
    assert looks_like_desktop_objective(INVENTORY_PROMPT) is False
    assert looks_like_multi_step_skill_request(
        INVENTORY_PROMPT,
        ["desktop_task", "computer_use"],
    ) is False
    action_intent = detect_action_intent(INVENTORY_PROMPT)
    assert action_intent.should_execute is False
    assert analyze_turn(INVENTORY_PROMPT, matched_skills=["desktop_task"]).intent_type == "CHAT"


@pytest.mark.parametrize(
    "prompt",
    [
        "What tools can you do externally if I ask you to flex your muscles?",
        "What can you do on my computer with apps, browser tabs, files, and documents?",
        "Describe whether you can open apps, use the browser, and work with PDFs on my desktop.",
        "What tools could Aura use externally in a hypothetical desktop scenario?",
        "What tools can she do externally from the live desktop path?",
        "What tools she could hypothetically do externally, and can she flex her muscles with a scenario?",
        "Explain what tools she can use on the desktop and name a hypothetical scenario using them.",
    ],
)
def test_capability_inventory_variants_stay_descriptive(prompt: str) -> None:
    from core.phases.response_contract import looks_like_capability_inventory_request
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective
    from core.runtime.skill_task_bridge import (
        looks_like_capability_inventory_dialogue_request,
        looks_like_multi_step_skill_request,
    )
    from interface.routes import chat as chat_routes

    assert looks_like_capability_inventory_request(prompt) is True
    assert looks_like_capability_inventory_dialogue_request(prompt) is True
    assert chat_routes._is_explicit_capability_inventory_request(prompt) is True
    assert looks_like_desktop_objective(prompt) is False
    assert looks_like_multi_step_skill_request(
        prompt,
        ["desktop_task", "computer_use", "web_search"],
    ) is False


def test_bounded_capability_inventory_repair_handles_failed_desktop_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    class FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
            ]

    def fake_get(name: str, default=None):
        if name == "capability_engine":
            return FakeCapabilityEngine()
        return default

    prompt = (
        "What tools she could hypothetically do externally, and can she flex her muscles "
        "with one concrete scenario?"
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", fake_get)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)

    reply = chat_routes._build_bounded_capability_inventory_repair_reply(prompt)

    lowered = reply.lower()
    assert "computer_use" in reply
    assert "web_search" in reply
    assert "will/authority" in lowered
    assert "measured tool surface" in lowered
    assert "assistant" not in lowered


def test_desktop_command_with_negative_constraint_still_executes() -> None:
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    prompt = "Open Notes and write a timestamped sentence, but do not use Chrome."

    assert looks_like_desktop_objective(prompt) is True


@pytest.mark.asyncio
async def test_cognitive_routing_keeps_inventory_prompt_off_skill_fastpath(monkeypatch) -> None:
    from core.phases.cognitive_routing import CognitiveRoutingPhase
    from core.state.aura_state import AuraState

    detect_calls = 0

    class Capability:
        def detect_intent(self, *_args, **_kwargs):
            nonlocal detect_calls
            detect_calls += 1
            return ["desktop_task", "computer_use"]

    container = SimpleNamespace(
        get=lambda name, default=None: Capability() if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)
    state = AuraState()
    state.cognition.current_objective = INVENTORY_PROMPT
    state.cognition.current_origin = "desktop_ui"

    routed = await phase.execute(state, INVENTORY_PROMPT)

    assert detect_calls == 0
    assert routed.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in routed.response_modifiers


def test_capability_inventory_reply_repairs_false_tool_limitation(monkeypatch: pytest.MonkeyPatch) -> None:
    from interface.routes import chat as chat_routes

    class FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "file_operation",
                    "available": True,
                    "description": "Read and write local files and documents.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "file_system",
                },
                {
                    "name": "sovereign_terminal",
                    "available": True,
                    "description": "Run governed terminal commands.",
                    "route_class": "subprocess",
                    "risk_class": "critical",
                    "effect_scope": "subprocess",
                },
                {
                    "name": "clock",
                    "available": True,
                    "description": "Return the current local time.",
                    "route_class": "read_only",
                    "risk_class": "low",
                    "effect_scope": "read_only",
                },
            ]

    def fake_get(name: str, default=None):
        if name == "capability_engine":
            return FakeCapabilityEngine()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", fake_get)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)

    assert chat_routes._capability_inventory_reply_is_inadequate(
        INVENTORY_PROMPT,
        "I can't actually open apps or execute tools on your computer.",
    )
    repaired = chat_routes._build_grounded_capability_inventory_reply(INVENTORY_PROMPT)

    lowered = repaired.lower()
    assert "computer_use" in repaired
    assert "web_search" in repaired
    assert "file_operation" in repaired
    assert "sovereign_terminal" in repaired
    assert "specialized governed skills (clock)" in lowered
    assert "files, documents, and workspace operations (file_operation, clock)" not in lowered
    assert "will/authority" in lowered
    assert "not opening apps" in lowered
    assert "i can't" not in lowered


def test_capability_inventory_rejects_midword_clipped_tail() -> None:
    from interface.routes import chat as chat_routes

    clipped = (
        "I can use governed desktop control, browser and web research, file and "
        "document work, terminal execution, memory, tools, and skill surfaces. "
        "Will/Authority approval and receipts are required before execution. "
        "For this turn I am only describing th"
    )

    assert chat_routes._capability_inventory_reply_is_inadequate(
        INVENTORY_PROMPT,
        clipped,
    )


def test_the_capability_inventory_is_not_a_runtime_status_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    class FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
            ]

    def fake_get(name: str, default=None):
        if name == "capability_engine":
            return FakeCapabilityEngine()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", fake_get)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)

    inventory = chat_routes._build_grounded_capability_inventory_reply(INVENTORY_PROMPT)

    assert "computer_use" in inventory
    assert "Cortex (32B) is the active foreground lane" not in inventory


@pytest.mark.asyncio
async def test_owner_name_recall_repairs_thin_desktop_cognitive_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_memory_state, "_owner_session_is_verified", lambda **_kwargs: True)
    monkeypatch.setattr(_chat_memory_state, "_resolve_primary_operator_name", lambda: "Bryan")

    repaired, stale, same_diff, off_topic, reason, did_repair = (
        await chat_routes._repair_final_degraded_reply(
            "Do you know my name?",
            "Yes.",
            stale=False,
            same_diff=False,
            off_topic=False,
        )
    )

    assert did_repair is True
    assert stale is False
    assert same_diff is False
    assert off_topic is False
    assert reason == ""
    assert "Bryan" in repaired
    assert "verified owner session" in repaired


@pytest.mark.asyncio
async def test_verified_owner_direct_address_name_drift_is_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_memory_state, "_owner_session_is_verified", lambda **_kwargs: True)
    monkeypatch.setattr(_chat_memory_state, "_resolve_primary_operator_name", lambda: "Bryan")

    repaired, stale, same_diff, off_topic, reason, did_repair = (
        await chat_routes._repair_final_degraded_reply(
            "Whatever is on yours. Thinking about anything interesting lately?",
            "Absolutely, John. I've been thinking about the nature of thought itself.",
            stale=False,
            same_diff=False,
            off_topic=False,
            desktop_cognitive_engine_required=True,
        )
    )

    assert did_repair is True
    assert stale is False
    assert same_diff is False
    assert off_topic is False
    assert reason == ""
    assert "Bryan" in repaired
    assert "John" not in repaired


def test_owner_name_recall_does_not_disclose_without_owner_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_memory_state, "_owner_session_is_verified", lambda **_kwargs: False)

    reply = chat_routes._build_owner_name_recall_reply("What's my name?")

    assert reply
    assert "owner-verified" in reply
    assert "Bryan" not in reply


@pytest.mark.asyncio
async def test_state_fastpath_serves_verified_owner_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_memory_state, "_owner_session_is_verified", lambda **_kwargs: True)
    monkeypatch.setattr(_chat_memory_state, "_resolve_primary_operator_name", lambda: "Bryan")

    result = await chat_routes._build_memory_state_fastpath_reply("Do you know my name?")

    assert result is not None
    reply, status = result
    assert status == "owner_identity_recall"
    assert "Bryan" in reply
    assert "verified owner session" in reply


@pytest.mark.asyncio
async def test_state_fastpath_blocks_owner_identity_without_verified_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_memory_state, "_owner_session_is_verified", lambda **_kwargs: False)
    monkeypatch.setattr(_chat_memory_state, "_resolve_primary_operator_name", lambda: "Bryan")

    result = await chat_routes._build_memory_state_fastpath_reply("Do you know my name?")

    assert result is not None
    reply, status = result
    assert status == "owner_identity_recall"
    assert "owner-verified" in reply
    assert "Bryan" not in reply


@pytest.mark.asyncio
async def test_state_fastpath_uses_restored_owner_session_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_memory_state, "_owner_session_is_verified", lambda **kwargs: bool(kwargs.get("owner_session_restored")))
    monkeypatch.setattr(_chat_memory_state, "_resolve_primary_operator_name", lambda: "Bryan")

    result = await chat_routes._build_memory_state_fastpath_reply(
        "Do you know my name?",
        owner_session_restored=True,
    )

    assert result is not None
    reply, status = result
    assert status == "owner_identity_recall"
    assert "Bryan" in reply


@pytest.mark.asyncio
async def test_conversation_recall_reads_completed_chat_log() -> None:
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "Can you explain why the live desktop path failed?",
                    "aura": "The live path was hitting the final CognitiveEngine reliability gate.",
                    "status": "complete",
                },
                {
                    "user": "What should we fix first?",
                    "aura": "We should fix routing, memory pressure, and fallback repair.",
                    "status": "complete",
                },
            ]
        )

    last_user = await chat_routes._what_the_conversation_holds_for("What did I just ask you?")
    last_aura = await chat_routes._what_the_conversation_holds_for("What did you just say?")
    topic = await chat_routes._what_the_conversation_holds_for("What were we talking about?")

    assert last_user is not None
    assert "What should we fix first?" in last_user
    assert last_aura is not None
    assert "fallback repair" in last_aura
    assert topic is not None
    assert "live desktop path" in topic
    assert "CognitiveEngine reliability gate" in topic


@pytest.mark.asyncio
async def test_a_thin_reply_to_a_recall_question_is_not_replaced_by_the_transcript() -> None:
    """The transcript is what she answers from, not an answer served in her place (15 November 2026)."""
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.append(
            {
                "user": "Can you remember this chain?",
                "aura": "Yes, I am tracking the chain in the live conversation log.",
                "status": "complete",
            }
        )

    repaired, *_rest = await chat_routes._repair_final_degraded_reply(
        "What did you just say?",
        "Yes.",
        stale=False,
        same_diff=False,
        off_topic=False,
    )

    assert "live conversation log" not in str(repaired or "")


@pytest.mark.asyncio
async def test_the_state_fastpath_leaves_a_recall_question_to_her() -> None:
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.append(
            {
                "user": "What tools can you hypothetically use externally on my computer?",
                "aura": "I can use governed desktop, browser, file, terminal, memory, and self-repair tools.",
                "status": "complete",
            }
        )

    assert await chat_routes._build_memory_state_fastpath_reply("What did you just say?") is None


@pytest.mark.asyncio
async def test_session_memory_pin_does_not_overclaim_when_durable_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    class MemoryFacade:
        async def add_memory(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: MemoryFacade() if name == "memory_facade" else default),
    )
    monkeypatch.setattr(_chat_memory_state, "_append_session_memory_pin_ledger", lambda *_args: False)

    result = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this phrase for later in this session: ember-vault-93",
        session_id="durable-failure",
    )

    assert result is not None
    reply, status = result
    assert status == "session_memory_pin_transient"
    assert "durable memory storage did not accept the write" in reply
    assert "durable session memory" not in reply


@pytest.mark.asyncio
async def test_session_memory_pin_without_principal_or_session_never_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from interface.routes import chat as chat_routes

    facade_calls = []

    class MemoryFacade:
        async def add_memory(self, *_args, **_kwargs):
            facade_calls.append(True)
            return True

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: MemoryFacade()
            if name == "memory_facade"
            else default
        ),
    )

    stored = await chat_routes._store_session_memory_pin(
        "unowned secret",
        "Remember the unowned secret.",
    )

    assert stored is False
    assert facade_calls == []
    assert not ledger_path.exists()


@pytest.mark.asyncio
async def test_session_memory_pin_handles_memory_facade_exception_truthfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    add_memory_calls = []
    ledger_calls = []

    class MemoryFacade:
        async def add_memory(self, *_args, **_kwargs):
            add_memory_calls.append(True)
            raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: MemoryFacade() if name == "memory_facade" else default),
    )
    monkeypatch.setattr(
        _chat_memory_state,
        "_append_session_memory_pin_ledger",
        lambda content, source, timestamp, **_kwargs: ledger_calls.append(
            (content, source, timestamp)
        )
        or True,
    )

    result = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this phrase for later in this session: ember-vault-93",
        session_id="facade-exception",
    )

    assert result is not None
    reply, status = result
    assert status == "session_memory_pin"
    assert "durable session memory" in reply
    assert "ember-vault-93" in reply
    assert add_memory_calls == [True]
    assert ledger_calls


@pytest.mark.asyncio
async def test_codeword_memory_fastpath_round_trips_without_model_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    class MemoryFacade:
        async def add_memory(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: MemoryFacade() if name == "memory_facade" else default),
    )

    set_result = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this codeword for me: amber-45873. Just confirm you have it.",
        session_id="codeword-round-trip",
    )
    recall_result = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I just give you?",
        session_id="codeword-round-trip",
    )

    assert set_result is not None
    set_reply, set_status = set_result
    assert set_status == "session_memory_pin"
    assert "amber-45873" in set_reply
    assert recall_result is not None
    recall_reply, recall_status = recall_result
    assert recall_status == "session_memory_recall"
    assert "amber-45873" in recall_reply


@pytest.mark.asyncio
async def test_anaphoric_memory_pin_uses_recent_conversation_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    class MemoryFacade:
        async def add_memory(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: MemoryFacade() if name == "memory_facade" else default),
    )
    monkeypatch.setattr(
        _chat_memory_state,
        "_append_session_memory_pin_ledger_guarded",
        lambda *_args, **_kwargs: True,
    )
    chat_routes._conversation_log.append(
        {
            "id": "prior-turn",
            "timestamp": "2026-07-01T00:00:00+00:00",
            "user": "What have you been thinking about?",
            "aura": "I've been thinking about the nature of thought itself and whether a pattern can become a self.",
            "status": "complete",
            "session_id": "session-a",
        }
    )

    result = await chat_routes._build_memory_state_fastpath_reply(
        "Hold this thought. And remember it.",
        session_id="session-a",
    )

    assert result is not None
    reply, status = result
    assert status == "session_memory_pin"
    assert "nature of thought itself" in reply
    assert "durable session memory" in reply
