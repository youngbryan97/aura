from types import SimpleNamespace

import pytest


class _Decision:
    receipt_id = "will-receipt-test"
    outcome = SimpleNamespace(value="refuse")
    reason = "test"

    def __init__(self, approved: bool):
        self._approved = approved
        self.outcome = SimpleNamespace(value="proceed" if approved else "refuse")

    def is_approved(self):
        return self._approved


@pytest.mark.asyncio
async def test_identity_reflection_blocks_identity_mutation_without_will_approval(monkeypatch):
    from core.phases.identity_reflection import IdentityReflectionPhase
    from core.state.aura_state import AuraState
    import core.will as will_module

    state = AuraState.default()
    state.version = 20
    state.identity.narrative_version = 3

    monkeypatch.setattr(
        will_module,
        "get_will",
        lambda: SimpleNamespace(decide=lambda **_kwargs: _Decision(False)),
    )

    result = await IdentityReflectionPhase(container=None).execute(state)

    assert result.identity.narrative_version == 3


@pytest.mark.asyncio
async def test_identity_reflection_records_will_receipt_when_mutating(monkeypatch):
    from core.phases.identity_reflection import IdentityReflectionPhase
    from core.state.aura_state import AuraState
    import core.will as will_module

    state = AuraState.default()
    state.version = 20
    state.identity.narrative_version = 3

    monkeypatch.setattr(
        will_module,
        "get_will",
        lambda: SimpleNamespace(decide=lambda **_kwargs: _Decision(True)),
    )

    result = await IdentityReflectionPhase(container=None).execute(state)

    assert result.identity.narrative_version == 4
    assert result.response_modifiers["identity_reflection_will_receipt"] == "will-receipt-test"


@pytest.mark.asyncio
async def test_identity_reflection_never_truncates_long_form_speech():
    from core.phases.identity_reflection import IdentityReflectionPhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    long_answer = "A complete technical paragraph.\n" * 240
    assert len(long_answer) > 5000
    state.cognition.working_memory.append(
        {"role": "assistant", "content": long_answer}
    )

    result = await IdentityReflectionPhase(container=None).execute(state)

    assert result.cognition.working_memory[-1]["content"] == long_answer
