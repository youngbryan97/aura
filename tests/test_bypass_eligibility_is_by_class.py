"""An unlisted spelling of "unavailable" refused the whole action.

`record_bypass` checked its reason against an allowlist of exact strings, and
`preaction_cortex` COMPOSES its reasons — `availability_failure:{exception
class}` — so any exception nobody had enumerated produced a reason that was not
eligible. And an ineligible bypass does not merely decline to bypass: it raises,
and the action is refused before dispatch.

MEASURED live 2026-08-18: a user-requested browser task died here after
clearing every authority gate ahead of it.

Eligibility is by class now. "The rehearsal could not run" is exactly what a
bypass is for, however the unavailability spelled itself. An
`episode_integrity_*` reason is the opposite — the rehearsal ran and refused —
and must never bypass, because that is the case the allowlist exists to stop
from masquerading as a decision.
"""

from __future__ import annotations

import pytest

from core.brain.external_execute_coordinator import _bypass_reason_is_eligible


@pytest.mark.parametrize(
    "reason",
    [
        "availability_failure:generation_gate_busy",
        "availability_failure:no_resident_model",
        "availability_failure:SomeErrorNobodyEnumerated",
        "availability_failure:rehearsal_unavailable",
        "latent_cortex_absent",
        "disabled:AURA_PREACTION_RLC=0",
    ],
)
def test_an_unavailable_rehearsal_may_be_bypassed(reason):
    assert _bypass_reason_is_eligible(reason) is True


@pytest.mark.parametrize(
    "reason",
    [
        "episode_integrity_failure:ValueError",
        "episode_integrity_refusal:unknown",
        "external_execution_objective_mismatch",
        "prediction_confirmed",
        "",
        "anything_at_all",
    ],
)
def test_a_verdict_may_not_be_bypassed(reason):
    """The rehearsal ran and said something; that is not unavailability."""
    assert _bypass_reason_is_eligible(reason) is False


def test_the_executors_own_fallback_is_eligible():
    """It reached for a bypass with a reason its own gate rejected."""
    import inspect

    from core.runtime import action_executor

    source = inspect.getsource(action_executor)
    assert '"availability_failure:rehearsal_unavailable"' in source
    assert _bypass_reason_is_eligible("availability_failure:rehearsal_unavailable")


def test_the_refusal_names_the_reason():
    """It said only 'not eligible', so nothing could say which reason."""
    import inspect

    from core.brain import external_execute_coordinator

    source = inspect.getsource(external_execute_coordinator.ExternalExecuteCoordinator.record_bypass)
    assert "{bounded_reason}" in source
