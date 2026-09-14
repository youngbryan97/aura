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


def _something_the_story_does_not_hold(state) -> None:
    """Arrange the condition the phase now revises on.

    The trigger used to be `state.version % 20`, a counter. It is now recall
    bringing back something the narrative has no room for, measured against how
    much it usually brings back — so the setup is a settled history and then
    one recollection unlike it. What these tests are about is the Will gate,
    which is the same either way.
    """
    from core.self.revision import get_revision_ledger, reset_for_test

    reset_for_test()
    state.identity.current_narrative = "She builds instruments and keeps records."
    ledger = get_revision_ledger()
    for _ in range(8):
        ledger.read(["records of the instruments she builds"], state.identity.current_narrative)
    state.cognition.long_term_memory = [
        "a concert in Brooklyn changed how she hears saxophone phrasing entirely"
    ]


@pytest.mark.asyncio
async def test_identity_reflection_blocks_identity_mutation_without_will_approval(monkeypatch):
    from core.phases.identity_reflection import IdentityReflectionPhase
    from core.state.aura_state import AuraState
    import core.will as will_module

    state = AuraState.default()
    state.identity.narrative_version = 3
    _something_the_story_does_not_hold(state)

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
    state.identity.narrative_version = 3
    _something_the_story_does_not_hold(state)

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
