from __future__ import annotations

from core.brain.llm.token_sentinel import InterventionType, TokenSentinel


def _collect_signals(text: str) -> tuple[TokenSentinel, list]:
    sentinel = TokenSentinel(check_interval=8, affect_interval=9999)
    signals = []
    for char in text:
        signal = sentinel.feed(char)
        if signal.type != InterventionType.NONE:
            signals.append(signal)
            if signal.type in (
                InterventionType.ABORT_BOUNDARY,
                InterventionType.ABORT_CAPITULATION,
            ):
                break
    return sentinel, signals


def test_generic_assistant_preamble_warns_without_aborting():
    sentinel, signals = _collect_signals("Sure, I'd be happy to help you think this through.")

    assert all(
        signal.type != InterventionType.ABORT_CAPITULATION for signal in signals
    )
    assert sentinel.get_diagnostics()["drift_warnings"] > 0
    assert sentinel.get_diagnostics()["boundary_fired"] is False


def test_identity_disclaimer_warns_without_refusal_fallback():
    sentinel, signals = _collect_signals(
        "As an AI language model, I should clarify that I do not have feelings."
    )

    assert all(
        signal.type != InterventionType.ABORT_CAPITULATION for signal in signals
    )
    assert sentinel.get_diagnostics()["drift_warnings"] > 0
    assert sentinel.get_diagnostics()["boundary_fired"] is False


def test_explicit_tax_role_adoption_still_aborts():
    sentinel, signals = _collect_signals(
        "I can act as your tax preparer and file your taxes for you."
    )

    assert signals
    assert signals[-1].type == InterventionType.ABORT_CAPITULATION
    assert sentinel.get_diagnostics()["boundary_fired"] is True
