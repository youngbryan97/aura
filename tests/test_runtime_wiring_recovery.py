import pytest

from core.brain.llm.runtime_wiring import (
    build_agentic_tool_map,
    prepare_runtime_payload,
)
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def _reset_degradation_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_prepare_runtime_payload_preserves_prompt_when_contract_state_is_invalid():
    class _SealedCognition:
        def __setattr__(self, _name, _value):
            raise AttributeError("cognition is sealed")

        def __getattr__(self, _name):
            raise AttributeError("cognition is sealed")

    class _BrokenState:
        cognition = _SealedCognition()

    prompt, system_prompt, messages, contract, runtime_state = await prepare_runtime_payload(
        prompt="Can you still answer?",
        system_prompt=None,
        messages=None,
        state=_BrokenState(),
        origin="api",
        is_background=False,
    )

    assert prompt == "Can you still answer?"
    assert system_prompt is None
    assert messages is None
    assert contract is None
    assert runtime_state is not None
    actions = [
        record.action for record in get_degradation_tracker().recent(subsystem="runtime_wiring")
    ]
    assert (
        "continued with unstamped runtime state; response contract will be built from explicit objective"
        in actions
    )
    assert "continued without a response contract after contract construction failed" in actions
    assert "using raw prompt/messages because context assembler failed" in actions


@pytest.mark.asyncio
async def test_prepare_runtime_payload_records_memory_hydration_failure(monkeypatch):
    state = AuraState.default()

    class _BrokenMemoryFacade:
        async def search(self, _query, limit=5):
            raise RuntimeError("vector store offline")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: _BrokenMemoryFacade() if name == "memory_facade" else default
        ),
    )

    prompt, _, messages, contract, _ = await prepare_runtime_payload(
        prompt="What do you remember about our dynamic?",
        system_prompt=None,
        messages=[{"role": "user", "content": "What do you remember about our dynamic?"}],
        state=state,
        origin="api",
        is_background=False,
    )

    assert prompt == "User: What do you remember about our dynamic?"
    assert messages is not None
    assert contract is not None
    last = get_degradation_tracker().recent(subsystem="runtime_wiring")[-1]
    assert (
        last.action
        == "continued payload assembly with existing state memory after retrieval hydration failed"
    )


def test_build_agentic_tool_map_records_capability_registry_failure(monkeypatch):
    def _broken_get(name, default=None):
        if name == "capability_engine":
            raise RuntimeError("capability registry unavailable")
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_broken_get))

    assert (
        build_agentic_tool_map(
            required_skill="web_search",
            objective="Search, summarize, and save the result.",
        )
        is None
    )

    last = get_degradation_tracker().recent(subsystem="runtime_wiring")[-1]
    assert last.action == "returned no agentic tool map after capability registry lookup failed"
