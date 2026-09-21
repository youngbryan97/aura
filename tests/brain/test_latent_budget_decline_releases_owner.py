"""Being priced out of an optional stage must not cost the turn.

LIVE, 2026-08-10, on the desktop, one turn after a restart:

    Recursive Latent Cortex exhausted the single resident owner
    (latent_phase_failed:RuntimeError:compute budget cannot afford window
    [0:16) for 9 slots); refusing a late ordinary generation.
    stage=latent_optimization

    CognitiveEngine retained single ownership of the failed turn; skipping a
    duplicate route-level model call.

and the person got "I couldn't get to an answer I'd stand behind on that one."

Three correct components producing a wrong outcome together:

* recurrence.LoopCore.run raised a bare RuntimeError when the budget could not
  afford a window. Everywhere else in engine.py an unaffordable budget is a
  graceful `if not budget.can_afford(...)` skip; only this one raised.
* the engine mapped every _LATENT_PHASE_ERRORS exception to the single class
  ``latent_phase_failed``, so "declined to spend" and "a numerical invariant
  blew up mid-decode" arrived at the caller as the same string.
* latent_owner_exhausted — whose actual question is "could a second decode
  collide with a still-cleaning worker?" — falls through to "the episode has an
  id and reached a stage", which is true of both.

The window was refused BEFORE any layer ran. Nothing was in flight, the
resident model was clean, and one ordinary generation would have answered the
turn. Instead the answer was refused to protect a worker that was not busy.

So the refusal is typed, kept distinct through the engine, and releases the
owner like a soft cancel does.
"""

from __future__ import annotations

import inspect

import pytest

from core.brain.foreground_latent_runtime import latent_owner_exhausted
from tests.source_contract import family_text


def _receipt(**overrides):
    """A receipt that the fallthrough would read as owner-exhausted."""
    receipt = {
        "episode_id": "ep-8817",
        "last_stage": "latent_optimization",
        "input_token_count": 411,
    }
    receipt.update(overrides)
    return receipt


def test_a_budget_decline_releases_the_resident_owner() -> None:
    """The live failure, expressed as the outcome that must not repeat."""
    reason = (
        "latent_budget_declined:ComputeBudgetUnaffordable:"
        "compute budget cannot afford window [0:16) for 9 slots"
    )

    assert latent_owner_exhausted(reason, _receipt()) is False


def test_the_same_receipt_still_holds_the_owner_for_a_real_phase_failure() -> None:
    """The guard must keep guarding — this is not a blanket release."""
    reason = "latent_phase_failed:LoopCoreError:recurrent state is non-finite at shared_update_output"

    assert latent_owner_exhausted(reason, _receipt()) is True


@pytest.mark.parametrize(
    "reason",
    [
        "latent_timeout:foreground",
        "latent_integrity:receipt_mismatch",
        "worker_identity_failed:adapter_stack",
    ],
)
def test_the_existing_release_and_hold_rules_are_unchanged(reason: str) -> None:
    assert latent_owner_exhausted(reason, _receipt()) is True


def test_soft_cancel_still_releases() -> None:
    assert latent_owner_exhausted("soft_cancel:user", _receipt()) is False


# ── The type has to survive the trip ───────────────────────────────────────

def test_the_loop_core_raises_a_typed_refusal() -> None:
    """A bare RuntimeError is indistinguishable from a decode blowing up."""
    from core.brain.llm.latent_cortex import recurrence
    from core.brain.llm.latent_cortex.loop_core import ComputeBudgetUnaffordable

    source = inspect.getsource(recurrence.WindowRunner.run)

    assert "ComputeBudgetUnaffordable" in source
    assert "raise RuntimeError(" not in source
    assert issubclass(ComputeBudgetUnaffordable, RuntimeError)


def test_the_engine_keeps_the_refusal_distinct() -> None:
    """Mapping it back onto latent_phase_failed would undo the whole fix."""
    from core.brain.llm.latent_cortex import engine

    source = family_text(engine)

    assert '"latent_budget_declined"' in source
    assert "isinstance(exc, ComputeBudgetUnaffordable)" in source


def test_a_budget_refusal_is_still_a_subclass_of_the_caught_errors() -> None:
    """It must land in the engine's handler, not escape as an unknown crash."""
    from core.brain.llm.latent_cortex.engine import _LATENT_PHASE_ERRORS
    from core.brain.llm.latent_cortex.loop_core import ComputeBudgetUnaffordable

    assert issubclass(ComputeBudgetUnaffordable, _LATENT_PHASE_ERRORS)
