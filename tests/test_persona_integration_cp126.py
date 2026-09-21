"""CP126 contract tests for persona integration."""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.brain import persona_integration as pi
from core.brain.persona_integration import (
    PERSONA_CONTEXT_KEY,
    initialize_persona_integration,
    persona_integration_status,
    uninstall_persona_integration,
)

PERSONA_TEXT = "You are Aura. Speak precisely."


class _Adapter:
    def __init__(self, personas=("aura", "mist"), system=PERSONA_TEXT):
        self._personas = list(personas)
        self._system = system
        self.active = ""

    def list_personas(self):
        return list(self._personas)

    def set_persona(self, name):
        self.active = name

    def build_prompts(self, name, instruction):
        return {"system": f"{self._system} [{name}]", "user": instruction}


class _AsyncEngine:
    def __init__(self):
        self.calls = []

    async def think(self, objective, context=None, mode="fast", **kwargs):
        self.calls.append({"objective": objective, "context": context, "kwargs": kwargs})
        return "thought"


class _SyncEngine:
    def __init__(self):
        self.calls = []

    def think(self, objective, context=None, **kwargs):
        self.calls.append({"objective": objective, "context": context})
        return "thought"


@pytest.fixture()
def adapter(monkeypatch):
    made = _Adapter()
    import core.brain.persona_adapter as pa_module

    monkeypatch.setattr(pa_module, "PersonaAdapter", lambda *a, **k: made)
    return made


@pytest.fixture(autouse=True)
def _no_registry(monkeypatch):
    monkeypatch.setattr(pi, "_resolve_engine", lambda: None)


# --- ab3abbae: system instructions must not become user data ---------------


def test_objective_is_never_rewritten(adapter):
    engine = _AsyncEngine()
    assert initialize_persona_integration("aura", engine=engine)

    asyncio.run(engine.think("summarize the log"))

    call = engine.calls[0]
    assert call["objective"] == "summarize the log"
    assert "Persona Instruction" not in call["objective"]
    assert PERSONA_TEXT in call["context"][PERSONA_CONTEXT_KEY]


def test_persona_is_delivered_as_structured_context(adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)

    asyncio.run(engine.think("do a thing", {"user_id": "bryan"}))

    context = engine.calls[0]["context"]
    assert context["user_id"] == "bryan"
    assert context[PERSONA_CONTEXT_KEY].startswith(PERSONA_TEXT)


def test_caller_supplied_persona_prompt_is_not_overwritten(adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)

    asyncio.run(engine.think("x", context={PERSONA_CONTEXT_KEY: "explicit"}))

    assert engine.calls[0]["context"][PERSONA_CONTEXT_KEY] == "explicit"


def test_engine_applies_the_context_key_at_system_role():
    """The seam the wrapper targets must exist in the engine."""
    from tests.source_contract import family_text_at

    # With the modules lifted out of it: the seam is in cognitive_engine_quick_reply.
    source = family_text_at("core/brain/cognitive_engine.py")
    assert 'context.get("persona_system_prompt")' in source
    assert "[PERSONA CONTRACT]" in source


# --- 51b7a4a1: installation must be idempotent -----------------------------


def test_repeated_initialization_does_not_stack_wrappers(adapter):
    engine = _AsyncEngine()
    first = initialize_persona_integration("aura", engine=engine)
    wrapped_once = engine.think

    second = initialize_persona_integration("aura", engine=engine)

    assert first.installed and second.installed
    assert second.reason == "already_installed"
    assert engine.think is wrapped_once


def test_switching_persona_rewraps_the_original_not_the_wrapper(adapter):
    engine = _AsyncEngine()
    original = engine.__class__.think
    initialize_persona_integration("aura", engine=engine)
    receipt = initialize_persona_integration("mist", engine=engine)

    assert receipt.installed and receipt.replaced_existing
    asyncio.run(engine.think("x"))
    prompt = engine.calls[0]["context"][PERSONA_CONTEXT_KEY]
    assert prompt.count(PERSONA_TEXT) == 1
    assert "[mist]" in prompt

    uninstall_persona_integration(engine=engine)
    assert engine.__class__.think is original


def test_uninstall_restores_the_original_method(adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)
    assert uninstall_persona_integration(engine=engine) is True

    asyncio.run(engine.think("x"))

    assert engine.calls[0]["context"] is None
    assert uninstall_persona_integration(engine=engine) is False


def test_status_reports_what_is_installed(adapter):
    engine = _AsyncEngine()
    assert persona_integration_status(engine=engine)["installed"] is False

    initialize_persona_integration("aura", engine=engine)
    status = persona_integration_status(engine=engine)

    assert status["installed"] is True
    assert status["persona"] == "aura"
    assert status["context_key"] == PERSONA_CONTEXT_KEY


# --- 481779b5: the wrapper must keep the call protocol ---------------------


def test_async_think_stays_a_coroutine_function(adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)

    assert inspect.iscoroutinefunction(engine.think)


def test_sync_think_stays_synchronous(adapter):
    engine = _SyncEngine()
    initialize_persona_integration("aura", engine=engine)

    assert not inspect.iscoroutinefunction(engine.think)
    assert engine.think("x") == "thought"
    assert engine.calls[0]["context"][PERSONA_CONTEXT_KEY]


def test_wrapper_preserves_the_method_name(adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)
    assert engine.think.__name__ == "think"


# --- 96ab9a40: a missing engine is not a successful integration ------------


def test_missing_cognitive_engine_reports_failure(adapter):
    receipt = initialize_persona_integration("aura")

    assert receipt.installed is False
    assert bool(receipt) is False
    assert receipt.reason == "cognitive_engine_not_registered"


def test_unknown_persona_reports_failure(adapter):
    receipt = initialize_persona_integration("nobody", engine=_AsyncEngine())

    assert not receipt
    assert receipt.reason == "persona_not_found"
    assert "aura" in receipt.details["available"]


def test_engine_without_think_reports_failure(adapter):
    class Bare:
        pass

    receipt = initialize_persona_integration("aura", engine=Bare())

    assert not receipt
    assert receipt.reason == "engine_has_no_think"


def test_unavailable_adapter_reports_failure(monkeypatch):
    import core.brain.persona_adapter as pa_module

    def boom(*args, **kwargs):
        raise RuntimeError("profiles corrupt")

    monkeypatch.setattr(pa_module, "PersonaAdapter", boom)

    receipt = initialize_persona_integration("aura", engine=_AsyncEngine())

    assert not receipt
    assert "persona_adapter_unavailable" in receipt.reason


# --- 71a42eba: every call style takes the same path ------------------------


def test_keyword_and_positional_objectives_are_conditioned_identically(adapter):
    positional, keyword = _AsyncEngine(), _AsyncEngine()
    initialize_persona_integration("aura", engine=positional)
    initialize_persona_integration("aura", engine=keyword)

    asyncio.run(positional.think("task"))
    asyncio.run(keyword.think(objective="task"))

    assert (
        positional.calls[0]["context"][PERSONA_CONTEXT_KEY]
        == keyword.calls[0]["context"][PERSONA_CONTEXT_KEY]
    )


def test_positional_context_argument_is_conditioned(adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)

    asyncio.run(engine.think("task", {"a": 1}, "deep"))

    assert engine.calls[0]["context"]["a"] == 1
    assert engine.calls[0]["context"][PERSONA_CONTEXT_KEY]


def test_non_dict_context_keyword_is_replaced_not_crashed(adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)

    asyncio.run(engine.think("task", context="not a dict"))

    assert engine.calls[0]["context"][PERSONA_CONTEXT_KEY]


# --- 4de64c89: ordinary failures must not break cognition ------------------


@pytest.mark.parametrize(
    "exc", [TypeError("bad"), ValueError("bad"), AttributeError("bad"), KeyError("bad")]
)
def test_prompt_building_failure_falls_back_to_the_plain_call(monkeypatch, adapter, exc):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)

    def boom(name, instruction):
        raise exc

    monkeypatch.setattr(adapter, "build_prompts", boom)
    recorded = []
    monkeypatch.setattr(pi, "record_degradation", lambda *a, **k: recorded.append(a))

    assert asyncio.run(engine.think("task")) == "thought"
    assert engine.calls[0]["objective"] == "task"
    assert recorded


def test_empty_persona_prompt_leaves_the_call_untouched(monkeypatch, adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)
    monkeypatch.setattr(adapter, "build_prompts", lambda n, i: {"system": "  "})

    asyncio.run(engine.think("task"))

    assert engine.calls[0]["context"] is None


def test_persona_prompt_is_bounded(monkeypatch, adapter):
    engine = _AsyncEngine()
    initialize_persona_integration("aura", engine=engine)
    monkeypatch.setattr(adapter, "build_prompts", lambda n, i: {"system": "x" * 99_000})

    asyncio.run(engine.think("task"))

    assert len(engine.calls[0]["context"][PERSONA_CONTEXT_KEY]) <= pi.MAX_PERSONA_PROMPT_CHARS
